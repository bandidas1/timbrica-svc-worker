# RunPod serverless handler — SoulX-Singer SVC (Apache-2.0 code AND weights).
# Same contract as handler_seedvc.py:
#   in : {source_url|source_b64, target_url|target_b64, result_put_url?,
#         semi_tone_shift, diffusion_steps}
#   out: {sample_rate, gen_seconds, engine, watermarked, bytes} + audio_b64 | uploaded
# Audio moves out of band by signed URL — RunPod caps a /run body at 10 MiB. See svc_io.py.
#
# Unlike seed-vc, SoulX needs an explicit F0 contour for BOTH audios (its CLI loads
# precomputed .npy). We compute them per request with the same RMVPE extractor the
# upstream preprocess pipeline uses, at its defaults (24 kHz grid, hop 480) — the
# grid the checkpoint was trained against.
#
# Naming trap: upstream calls the VOICE reference `prompt/pt` and the audio to be
# converted `target/gt`. Our contract is the other way round, so map carefully.
import io, os, sys, time, traceback
import numpy as np, soundfile as sf, torch

import svc_io

SOULX_DIR = os.environ.get("SOULX_DIR", "/soulx")
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(64 * 1024 * 1024)))
sys.path.insert(0, SOULX_DIR)
os.chdir(SOULX_DIR)

_READY = False; _LOAD_ERR = None
_model = _config = _f0x = _device = _load_wav = None

def _load():
    global _READY, _LOAD_ERR, _model, _config, _f0x, _device, _load_wav
    from soulxsinger.utils.file_utils import load_config
    from soulxsinger.utils.audio_utils import load_wav
    from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
    from preprocess.tools import F0Extractor

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _config = load_config("soulxsinger/config/soulxsinger.yaml")
    _load_wav = load_wav

    m = SoulXSingerSVC(_config).to(_device)
    ckpt = torch.load("pretrained_models/SoulX-Singer/model-svc.pt", weights_only=False, map_location="cpu")
    m.load_state_dict(ckpt["state_dict"], strict=True)
    if _device == "cuda":
        m.half(); m.mel.float()
    m.eval().to(_device)
    _model = m

    # Defaults on purpose: they define the F0 grid the checkpoint expects.
    _f0x = F0Extractor(
        model_path="pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt",
        device=_device, is_half=False, verbose=False,
    )
    _READY = True
    print(f"[boot] soulx ready sr={_config.audio.sample_rate} device={_device}", flush=True)

try:
    _load()
except Exception as e:
    _LOAD_ERR = f"model_load_failed: {repr(e)[:300]}"
    print(f"[boot] {_LOAD_ERR}", flush=True); traceback.print_exc()

_WM = None
try:
    import perth; _WM = perth.PerthImplicitWatermarker(); print("[boot] perth ready", flush=True)
except Exception as e:
    print(f"[boot] perth unavailable: {repr(e)[:200]}", flush=True)

def handler(event):
    inp = event.get("input") or {}
    if not _READY: return {"error": _LOAD_ERR or "model_unavailable"}
    n_steps = max(10, min(100, int(inp.get("diffusion_steps") or 32)))
    # "0 semitones" MUST mean "do not touch the key" — the caller mixes our vocal back
    # under THEIR instrumental, so a silent transposition returns an out-of-tune mix.
    # SoulX's auto_shift retunes the performance into the REFERENCE clip's register; the
    # reference is a short spoken liveness phrase, whose pitch has nothing to do with the
    # song. seed-vc passes auto_f0_adjust=False for exactly this reason — match it.
    # Auto-matching stays available, but only when explicitly asked for.
    raw_shift = inp.get("semi_tone_shift") or 0
    if isinstance(raw_shift, str) and raw_shift.strip().lower() == "auto":
        auto_shift, shift = True, 0
    else:
        auto_shift, shift = False, max(-12, min(12, int(raw_shift)))

    src = tgt = None
    try:
        # the sung performance -> gt_* ;  the target voice -> pt_*  (upstream's names are swapped)
        src = svc_io.write_tmp(svc_io.fetch_audio(inp, "source", MAX_AUDIO_BYTES), "svc_src_")
        tgt = svc_io.write_tmp(svc_io.fetch_audio(inp, "target", MAX_AUDIO_BYTES), "svc_ref_")
        sr = _config.audio.sample_rate
        t0 = time.time()
        pt_wav = _load_wav(tgt, sr).to(_device)
        gt_wav = _load_wav(src, sr).to(_device)
        pt_f0 = torch.from_numpy(_f0x.process(tgt)).unsqueeze(0).to(_device)
        gt_f0 = torch.from_numpy(_f0x.process(src)).unsqueeze(0).to(_device)
        with torch.no_grad():
            audio, _shift = _model.infer(
                pt_wav=pt_wav, gt_wav=gt_wav, pt_f0=pt_f0, gt_f0=gt_f0,
                auto_shift=auto_shift, pitch_shift=shift,
                n_steps=n_steps, cfg=_config.infer.cfg, use_fp16=(_device == "cuda"),
            )
        audio = audio.squeeze().float().cpu().numpy()
        gen_s = round(time.time() - t0, 2)

        watermarked = False
        if _WM is not None:
            try:
                audio = _WM.apply_watermark(np.asarray(audio, dtype="float32"), watermark=None, sample_rate=int(sr))
                watermarked = True
            except Exception as e:
                print(f"[wm] failed: {repr(e)[:200]}", flush=True)

        buf = io.BytesIO(); sf.write(buf, audio, int(sr), format="WAV", subtype="PCM_16")
        return svc_io.deliver(buf.getvalue(), inp, {
            "sample_rate": int(sr), "gen_seconds": gen_s, "engine": "soulx",
            "auto_shift": auto_shift, "pitch_shift": shift, "watermarked": watermarked})
    except svc_io.TransferError as e:
        # Moving the audio failed, not the model. Keep them apart in the logs.
        traceback.print_exc()
        return {"error": "transfer_failed", "detail": str(e)[:300]}
    except Exception as e:
        traceback.print_exc()
        return {"error": "convert_failed", "detail": repr(e)[:300]}
    finally:
        for p in (src, tgt):
            if p:
                try: os.unlink(p)
                except OSError: pass

if __name__ == "__main__":
    import runpod  # only the serverless entrypoint needs the runtime
    runpod.serverless.start({"handler": handler})
