# RunPod serverless handler — seed-vc singing voice conversion (SVC).
#
# Input  (event.input) — audio arrives by signed URL, or base64 for small clips:
#   source_url / source_b64 : the sung performance (WAV) — melody/lyrics/timing
#   target_url / target_b64 : the target-voice reference (WAV) — the timbre
#   result_put_url          : optional; PUT the WAV here instead of inlining it
#   semi_tone_shift         : optional int, -12..12 (pitch shift for key matching)
#   diffusion_steps         : optional int, 10..100 (default 40; 30-50 best for singing)
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

import svc_io

SEED_VC_DIR = os.environ.get("SEED_VC_DIR", "/seed-vc")
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(64 * 1024 * 1024)))

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


try:
    _load()
except Exception as e:  # noqa: BLE001
    _LOAD_ERR = f"model_load_failed: {repr(e)[:300]}"
    print(f"[boot] {_LOAD_ERR}", flush=True)
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
    if not _READY:
        return {"error": _LOAD_ERR or "model_unavailable"}

    steps = max(10, min(100, int(inp.get("diffusion_steps") or 40)))
    shift = max(-12, min(12, int(inp.get("semi_tone_shift") or 0)))

    src_path = tgt_path = None
    try:
        src_path = svc_io.write_tmp(svc_io.fetch_audio(inp, "source", MAX_AUDIO_BYTES), "svc_src_")
        tgt_path = svc_io.write_tmp(svc_io.fetch_audio(inp, "target", MAX_AUDIO_BYTES), "svc_ref_")

        t0 = time.time()
        full = None
        # voice_conversion is a generator: it streams mp3 chunks and yields the
        # complete waveform as (sr, np.float32[]) on its final chunk.
        # auto_f0_adjust=False — the source's own pitch contour IS the melody.
        for _chunk, full_out in app_svc.voice_conversion(
            src_path, tgt_path, steps, 1.0, 0.7, False, shift
        ):
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
        for p in (src_path, tgt_path):
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
