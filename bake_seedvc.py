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

import app_svc  # noqa: E402

app_svc.load_models(SimpleNamespace(checkpoint=None, config=None, share=False, fp16=False, gpu=0))
print("seed-vc weights baked", flush=True)
