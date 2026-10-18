# Build-time weight bake: pull every checkpoint seed-vc's SVC path needs into the
# image's HF cache (DiT f0 model, campplus, BigVGAN vocoder, Whisper encoder, RMVPE
# f0 extractor) so a cold worker never downloads at request time.
# Runs on the CPU builder → fp16=False. Non-fatal by design (see Dockerfile).
import os
import sys
import types

sys.path.insert(0, "/seed-vc")
os.chdir("/seed-vc")
sys.modules.setdefault("gradio", types.ModuleType("gradio"))

from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

import app_svc  # noqa: E402

# app_svc only assigns its module-level `device` inside `if __name__ == "__main__"`,
# so an import leaves it None (see handler_seedvc._load). The builder is CPU-only.
app_svc.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app_svc.load_models(SimpleNamespace(checkpoint=None, config=None, share=False, fp16=False, gpu=0))
print("seed-vc weights baked", flush=True)

# The SPEECH model (/voice-gender, model="speech"): its DiT checkpoint, the 22 kHz
# BigVGAN and Whisper-small. Same non-fatal contract — a failed bake only means the
# first speech request downloads them.
try:
    import app_vc  # noqa: E402

    app_vc.device = app_svc.device
    app_vc.load_models(SimpleNamespace(checkpoint=None, config=None, share=False, fp16=False, gpu=0))
    print("seed-vc speech weights baked", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"speech weight bake skipped: {e!r}"[:300], flush=True)

# seed-vc v2 (model="v2"): CFM + AR checkpoints, the two content extractors, the
# style encoder and its vocoder. load_v2_models() also moves them to the device —
# on the CPU builder that is just memory.
try:
    import inference_v2  # noqa: E402

    inference_v2.device = app_svc.device
    inference_v2.dtype = torch.float32
    inference_v2.load_v2_models(SimpleNamespace(
        diffusion_steps=30, length_adjust=1.0, compile=False, intelligibility_cfg_rate=0.7,
        similarity_cfg_rate=0.7, top_p=0.9, temperature=1.0, repetition_penalty=1.0,
        convert_style=False, anonymization_only=False, ar_checkpoint_path=None, cfm_checkpoint_path=None))
    print("seed-vc v2 weights baked", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"v2 weight bake skipped: {e!r}"[:300], flush=True)
