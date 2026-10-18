# RunPod serverless handler — seed-vc singing voice conversion (SVC).
#
# Input  (event.input) — audio arrives by signed URL, or base64 for small clips:
#   source_url / source_b64 : the sung performance (WAV) — melody/lyrics/timing
#   target_url / target_b64 : the target-voice reference (WAV) — the timbre
#   result_put_url          : optional; PUT the WAV here instead of inlining it
#   semi_tone_shift         : optional int, -12..12 (pitch shift for key matching)
#   diffusion_steps         : optional int, 10..100 (default 40; 30-50 best for singing)
#   model                   : optional, "svc" (default — the 44.1 kHz F0-conditioned
#                             singing model, /voice-cover), "speech" (v1 22.05 kHz
#                             speech model) or "v2" (seed-vc v2 CFM, timbre only —
#                             /voice-gender). "speech" and "v2" ignore
#                             semi_tone_shift / auto_f0_adjust: they take the register
#                             from the reference itself.
# Output: { sample_rate, gen_seconds, engine, watermarked, bytes } plus either
#         `audio_b64` (inline) or `uploaded: true` — or { error, detail }.
# Why not base64: see svc_io.py. RunPod caps a /run body at 10 MiB, which a mono
# 44.1 kHz WAV reaches after ~89 seconds. Songs are longer than that.
#
# The model is loaded ONCE at worker boot (module scope) so every warm request is
# fast. Audio prep (decode/normalise) stays on the Laravel side; this worker is a
# thin GPU executor. Backs the /voice-cover tool on Timbrica.
#
# seed-vc is GPLv3. It runs ONLY here, on our own GPU, and is never distributed to
# users — so no copyleft obligation is triggered (GPLv3 has no network clause).

import io
import os
import sys
import time
import traceback
import types

import numpy as np
import runpod
import soundfile as sf

import gpu_probe
import svc_io

SEED_VC_DIR = os.environ.get("SEED_VC_DIR", "/seed-vc")
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(64 * 1024 * 1024)))
# model="v2": sources longer than this run on a shortened reference (see handler)
V2_LONG_SOURCE_S = float(os.environ.get("V2_LONG_SOURCE_S", "60"))
V2_LONG_REF_S = float(os.environ.get("V2_LONG_REF_S", "12"))

# app_svc resolves configs relative to the repo root and imports `modules.*`.
sys.path.insert(0, SEED_VC_DIR)
os.chdir(SEED_VC_DIR)

# app_svc does `import gradio as gr` at module scope but only touches `gr` inside
# main() (the web UI), which we never call. Stub it so the ~500 MB Gradio stack
# stays out of the image. Verified against upstream: no module-level gr usage.
sys.modules.setdefault("gradio", types.ModuleType("gradio"))

# ---- load model once ---------------------------------------------------------
_READY = False
_LOAD_ERR = None
app_svc = None


class _DiscardedSegment:
    """Stand-in for pydub.AudioSegment inside app_svc.voice_conversion().

    That function is written for a streaming Gradio UI and mp3-encodes each chunk
    before yielding it; the handler keeps only the final waveform, so the encode is
    pure waste (and drags in ffmpeg at request time). `.export(...).read()` must
    still return bytes, hence the empty buffer.
    """

    def __init__(self, *args, **kwargs):
        pass

    def export(self, *args, **kwargs):
        return io.BytesIO(b"")


def _load():
    global _READY, _LOAD_ERR, app_svc
    from types import SimpleNamespace

    import torch
    import app_svc as _m

    # UPSTREAM TRAP: app_svc declares `device = None` at module scope and only
    # assigns it inside `if __name__ == "__main__"`. Importing the module (which we
    # must, to reuse load_models + voice_conversion) leaves it None, and the first
    # `.to(device)` dies with "'NoneType' object has no attribute 'type'".
    # Caught on a real GPU before ever building the image.
    _m.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # voice_conversion() is written for a streaming web UI: it mp3-encodes EVERY
    # chunk through pydub (→ an ffmpeg subprocess) and yields it. We discard those
    # chunks — only the final float waveform matters. Stub the encoder out: drops an
    # ffmpeg round-trip per chunk (real time on a 4-minute song) and removes a
    # runtime dependency. Found by running the handler on a real GPU.
    _m.AudioSegment = _DiscardedSegment

    args = SimpleNamespace(checkpoint=None, config=None, share=False, fp16=True, gpu=0)
    (
        _m.model_f0, _m.semantic_fn, _m.vocoder_fn, _m.campplus_model,
        _m.to_mel_f0, _m.mel_fn_args, _m.f0_fn,
    ) = _m.load_models(args)

    # main() derives these from the module globals load_models sets; do the same
    # here (defensively falling back to mel_fn_args) instead of launching Gradio.
    sr = getattr(_m, "sr", None) or _m.mel_fn_args.get("sampling_rate")
    hop = getattr(_m, "hop_length", None) or _m.mel_fn_args.get("hop_size")
    _m.sr = int(sr)
    _m.hop_length = int(hop)
    _m.max_context_window = _m.sr // _m.hop_length * 30
    _m.overlap_wave_len = _m.overlap_frame_len * _m.hop_length

    app_svc = _m
    _READY = True
    print(f"[boot] seed-vc ready sr={_m.sr} hop={_m.hop_length}", flush=True)


# ---- the SPEECH model, loaded on its first request -----------------------------
# Measured 2026-10-18 on 36 pairs of real Russian speech (FLEURS, 6 male + 6 female
# readers, our four catalogue voices as targets, same commit as this image):
#
#                                 target sim   source leak   UTMOS   WER    F0 miss
#   svc 44.1 kHz, 40 steps (old)     0.819        0.187        2.56   7.9 %  1.11 st
#   speech 22.05 kHz, 30 steps       0.870        0.160        2.75   8.4 %  0.59 st
#   v2 CFM, 30 steps                 0.833        0.134        3.29   7.5 %  0.79 st
#
# The singing model keeps the speaker's own F0 contour shifted by whole semitones;
# on speech that leaves the result less like the target and less natural than the
# INPUT (source UTMOS 2.82). The speech models take the register from the
# reference. v2 is the most natural of all (34 of 36 pairs over "speech") and lets
# the least of the speaker through; "speech" lands closest to the exact target.
# Both load lazily so /voice-cover's boot and memory stay as they were.
_SPEECH = None
_SPEECH_ERR = None
_V2 = None
_V2_ERR = None
_V2_ARGS = None


def _ensure_v2():
    global _V2, _V2_ERR, _V2_ARGS
    if _V2 is not None or _V2_ERR:
        return
    with _LOAD_LOCK:
        if _V2 is not None or _V2_ERR:
            return
        try:
            from types import SimpleNamespace

            import inference_v2 as _i

            _V2_ARGS = SimpleNamespace(
                diffusion_steps=30, length_adjust=1.0, compile=False,
                intelligibility_cfg_rate=0.7, similarity_cfg_rate=0.7, top_p=0.9,
                temperature=1.0, repetition_penalty=1.0, convert_style=False,
                anonymization_only=False, ar_checkpoint_path=None, cfm_checkpoint_path=None,
            )
            _i.vc_wrapper_v2 = _i.load_v2_models(_V2_ARGS)
            _V2 = _i
            print("[load] seed-vc v2 ready", flush=True)
        except Exception as e:  # noqa: BLE001
            _V2_ERR = f"v2_model_load_failed: {repr(e)[:300]}"
            print(f"[load] {_V2_ERR}", flush=True)
            traceback.print_exc()


def _ensure_speech():
    global _SPEECH, _SPEECH_ERR
    if _SPEECH is not None or _SPEECH_ERR:
        return
    with _LOAD_LOCK:
        if _SPEECH is not None or _SPEECH_ERR:
            return
        try:
            from types import SimpleNamespace

            import torch
            import app_vc as _v

            _v.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _v.AudioSegment = _DiscardedSegment
            args = SimpleNamespace(checkpoint=None, config=None, share=False, fp16=True, gpu=0)
            (_v.model, _v.semantic_fn, _v.vocoder_fn, _v.campplus_model,
             _v.to_mel, _v.mel_fn_args) = _v.load_models(args)
            sr = getattr(_v, "sr", None) or _v.mel_fn_args.get("sampling_rate")
            hop = getattr(_v, "hop_length", None) or _v.mel_fn_args.get("hop_size")
            _v.sr = int(sr)
            _v.hop_length = int(hop)
            _v.max_context_window = _v.sr // _v.hop_length * 30
            _v.overlap_wave_len = _v.overlap_frame_len * _v.hop_length
            _SPEECH = _v
            print(f"[load] seed-vc speech model ready sr={_v.sr}", flush=True)
        except Exception as e:  # noqa: BLE001
            _SPEECH_ERR = f"speech_model_load_failed: {repr(e)[:300]}"
            print(f"[load] {_SPEECH_ERR}", flush=True)
            traceback.print_exc()


# ⚠️⚠️ NOTHING that touches the GPU may run at import. Boot only NAMES the host.
#
# This is the fix for the outage of 16–18 August 2026. The model used to be loaded
# here, at module scope, i.e. BEFORE runpod.serverless.start() — so the worker
# announced itself ready only after the load finished. Any GPU that was slow, or
# that hung, or that could not run our kernels, left the worker in `initializing`
# for as long as it lasted. Measured consequence: three days of `ready: 0`, jobs
# queued and never dispatched, $19.88 billed for 28.7 worker-hours and ZERO
# completed jobs — and no way to see the cause, because RunPod shows worker logs
# only in its console and the container never got far enough to answer anything.
#
# A control run settled it: the same endpoint booted the stem-split image — which
# loads its model lazily, on the first request — in 425 seconds. Same endpoint,
# same hosts, same account. The difference was WHEN the model is loaded.
#
# So: become ready first, load on the first request (see _ensure_loaded). A GPU
# problem then arrives as a job error carrying the host report, which is a thing
# a person can read, instead of an invisible hang that bills by the hour.
print(f"[boot] host={gpu_probe.host_report()}", flush=True)

_GPU_ERR = None      # set by the first request, not at boot
_LOAD_LOCK = __import__("threading").Lock()


def _ensure_loaded() -> None:
    """Load the model on FIRST USE. Idempotent, and safe under concurrent calls."""
    global _LOAD_ERR, _GPU_ERR
    if _READY or _LOAD_ERR:
        return
    with _LOAD_LOCK:
        if _READY or _LOAD_ERR:
            return
        try:
            # Prove the machine can launch a kernel before spending a minute on a
            # model that would never run on it.
            gpu_probe.assert_gpu_usable()
        except gpu_probe.GpuUnusable as e:
            _GPU_ERR = str(e)[:200]
            _LOAD_ERR = f"gpu_unusable: {_GPU_ERR}"
            print(f"[load] gpu UNUSABLE: {_GPU_ERR}", flush=True)
            return
        try:
            _load()
        except Exception as e:  # noqa: BLE001
            _LOAD_ERR = f"model_load_failed: {repr(e)[:300]}"
            print(f"[load] {_LOAD_ERR}", flush=True)
            traceback.print_exc()


# ---- inaudible provenance watermark ------------------------------------------
# Perth (Resemble) neural watermark — marks the output as AI-generated and keeps it
# traceable. Required for EU AI Act Art. 50 machine-readable marking (and it is our
# own anti-abuse measure). Best-effort: never a gate on a paid result.
_WM = None
try:
    import perth
    _WM = perth.PerthImplicitWatermarker()
    print("[boot] perth watermarker ready", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[boot] perth unavailable: {repr(e)[:200]}", flush=True)


def handler(event):
    inp = event.get("input") or {}

    # A free question, answerable even on a machine that cannot compute: who are
    # you, and can you? Costs no model, no download, no tokens.
    if inp.get("probe"):
        return gpu_probe.probe_result()

    # First real request pays for the model load — the worker is already `ready`,
    # so a slow or hostile GPU costs THIS job a wait instead of hanging the whole
    # container in `initializing` forever.
    _ensure_loaded()

    # A REAL job on a host that cannot launch a kernel must NOT be answered with an
    # error: that spends the caller's one paid attempt proving the machine is
    # broken, and the retry lands on the same warm worker. Exiting hands the job
    # back to RunPod, which reschedules it somewhere that works, and this container
    # is offered no more work.
    if _GPU_ERR is not None:
        print(f"[gpu-unusable] {_GPU_ERR} host={gpu_probe.host_report()}", flush=True)
        os._exit(1)

    if not _READY:
        return {"error": _LOAD_ERR or "model_unavailable", "host": gpu_probe.host_report()}

    steps = max(10, min(100, int(inp.get("diffusion_steps") or 40)))
    shift = max(-12, min(12, int(inp.get("semi_tone_shift") or 0)))
    # auto_f0_adjust: retune the SOURCE melody into the REFERENCE voice's register.
    # OFF by default (mode A "sing yourself" — the source IS the user's own melody,
    # keep their pitch). The caller sets it True for modes B/C, where the source is
    # a catalogue/composed melody sung by someone else and must be transposed into
    # the user's range — without it a woman singing a male-register melody (or vice
    # versa) comes out badly (aksinia «голос ужасный», 2026-07-22; A/B-confirmed by ear).
    auto_f0 = bool(inp.get("auto_f0_adjust", False))

    model = str(inp.get("model") or "svc")
    if model not in ("svc", "speech", "v2"):
        return {"error": "unknown_model", "detail": model[:40]}
    if model == "speech":
        _ensure_speech()
        if _SPEECH is None:
            return {"error": _SPEECH_ERR or "speech_model_unavailable", "host": gpu_probe.host_report()}
    if model == "v2":
        _ensure_v2()
        if _V2 is None:
            return {"error": _V2_ERR or "v2_model_unavailable", "host": gpu_probe.host_report()}

    src_path = tgt_path = trimmed_ref = None
    try:
        src_path = svc_io.write_tmp(svc_io.fetch_audio(inp, "source", MAX_AUDIO_BYTES), "svc_src_")
        tgt_path = svc_io.write_tmp(svc_io.fetch_audio(inp, "target", MAX_AUDIO_BYTES), "svc_ref_")

        t0 = time.time()
        full = None
        # voice_conversion is a generator: it streams mp3 chunks and yields the
        # complete waveform as (sr, np.float32[]) on its final chunk.
        # auto_f0_adjust: False (mode A, keep the user's own melody pitch) or
        # True (mode B/C, transpose the catalogue/composed melody into the user's
        # register). Default False keeps the old behaviour for any caller that
        # doesn't pass the field.
        if model == "v2":
            # v2 works in 30-second windows and spends up to 25 s of each on the
            # reference, leaving ~5 s of source per pass: a 3-minute take was
            # 467 s on an RTX 4060 laptop (RTF 2.6). A 12-second reference leaves
            # 18 s per pass (36 pairs: target sim 0.833 → 0.811, UTMOS 3.29 →
            # 3.21). Short takes — 90 % of real ones are under 35 s — keep the full
            # reference; long ones trade a little likeness for finishing inside the
            # caller's poll window.
            ref_for_run = tgt_path
            try:
                src_s = sf.info(src_path).duration
            except Exception:  # noqa: BLE001
                src_s = 0.0
            if src_s > V2_LONG_SOURCE_S:
                ref, ref_sr = sf.read(tgt_path, dtype="float32", always_2d=True)
                ref_for_run = svc_io.write_tmp(b"", "svc_ref12_")
                sf.write(ref_for_run, ref[: int(ref_sr * V2_LONG_REF_S)].mean(axis=1), ref_sr, format="WAV")
                trimmed_ref = ref_for_run
            _V2_ARGS.diffusion_steps = steps
            full = _V2.convert_voice_v2(src_path, ref_for_run, _V2_ARGS)
        else:
            if model == "speech":
                gen = _SPEECH.voice_conversion(src_path, tgt_path, steps, 1.0, 0.7)
            else:
                gen = app_svc.voice_conversion(src_path, tgt_path, steps, 1.0, 0.7, auto_f0, shift)
            for _chunk, full_out in gen:
                if full_out is not None:
                    full = full_out
        if full is None:
            return {"error": "conversion_produced_nothing"}

        sr_out, wave = full
        wave = np.asarray(wave, dtype="float32")
        gen_s = round(time.time() - t0, 2)

        watermarked = False
        if _WM is not None:
            try:
                wave = _WM.apply_watermark(wave, watermark=None, sample_rate=int(sr_out))
                watermarked = True
            except Exception as e:  # noqa: BLE001
                print(f"[wm] watermark failed: {repr(e)[:200]}", flush=True)

        buf = io.BytesIO()
        sf.write(buf, wave, int(sr_out), format="WAV", subtype="PCM_16")
        return svc_io.deliver(buf.getvalue(), inp, {
            "sample_rate": int(sr_out),
            "gen_seconds": gen_s,
            "engine": "seedvc",
            "model": model,
            "watermarked": watermarked,
        })
    except svc_io.TransferError as e:
        # Moving the audio failed, not the model. Keep them apart in the logs.
        traceback.print_exc()
        return {"error": "transfer_failed", "detail": str(e)[:300]}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return {"error": "convert_failed", "detail": repr(e)[:300]}
    finally:
        for p in (src_path, tgt_path, trimmed_ref):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# Guarded so the module can be IMPORTED and driven directly (GPU verification,
# unit-style checks) without spawning the serverless loop. RunPod runs this file as
# `python -u /handler.py`, i.e. __main__, so production behaviour is unchanged.
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
