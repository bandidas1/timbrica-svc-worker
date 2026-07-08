# RunPod serverless handler — SoulX-Singer SVC (Apache-2.0 code AND weights).
# Same contract as handler_seedvc.py:
#   in : {source_b64, target_b64, semi_tone_shift, diffusion_steps}
#   out: {audio_b64 (WAV), sample_rate, gen_seconds, engine, watermarked}
#
# Unlike seed-vc, SoulX needs an explicit F0 contour for BOTH audios (its CLI loads
# precomputed .npy). We compute them per request with the same RMVPE extractor the
# upstream preprocess pipeline uses, at its defaults (24 kHz grid, hop 480) — the
# grid the checkpoint was trained against.
#
# Naming trap: upstream calls the VOICE reference `prompt/pt` and the audio to be
# converted `target/gt`. Our contract is the other way round, so map carefully.
import base64, io, os, sys, tempfile, time, traceback
import numpy as np, soundfile as sf, torch

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

def _tmp(b64, prefix):
    raw = base64.b64decode(b64, validate=True)
    if len(raw) > MAX_AUDIO_BYTES: raise ValueError("audio_too_large")
    if len(raw) < 1000: raise ValueError("audio_too_small")
    f = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", delete=False)
    f.write(raw); f.close(); return f.name

def handler(event):
    inp = event.get("input") or {}
    if not _READY: return {"error": _LOAD_ERR or "model_unavailable"}
    src_b64, tgt_b64 = inp.get("source_b64"), inp.get("target_b64")
    if not src_b64 or not isinstance(src_b64, str): return {"error": "source_b64_required"}
    if not tgt_b64 or not isinstance(tgt_b64, str): return {"error": "target_b64_required"}
    n_steps = max(10, min(100, int(inp.get("diffusion_steps") or 32)))
    shift = max(-12, min(12, int(inp.get("semi_tone_shift") or 0)))

    src = tgt = None
    try:
        src = _tmp(src_b64, "svc_src_")   # the sung performance  -> gt_*
        tgt = _tmp(tgt_b64, "svc_ref_")   # the target voice      -> pt_*
        sr = _config.audio.sample_rate
        t0 = time.time()
        pt_wav = _load_wav(tgt, sr).to(_device)
        gt_wav = _load_wav(src, sr).to(_device)
        pt_f0 = torch.from_numpy(_f0x.process(tgt)).unsqueeze(0).to(_device)
        gt_f0 = torch.from_numpy(_f0x.process(src)).unsqueeze(0).to(_device)
        with torch.no_grad():
            audio, _shift = _model.infer(
                pt_wav=pt_wav, gt_wav=gt_wav, pt_f0=pt_f0, gt_f0=gt_f0,
                # No explicit shift asked for -> let it match the target key itself.
                auto_shift=(shift == 0), pitch_shift=shift,
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
        return {"audio_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "sample_rate": int(sr), "gen_seconds": gen_s,
                "engine": "soulx", "watermarked": watermarked}
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
