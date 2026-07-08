# Build-time weight bake for the SoulX-Singer SVC worker: the SVC checkpoint
# (~2.8 GB) and the RMVPE F0 extractor, so a cold worker never downloads at request
# time. Non-fatal by design (see Dockerfile).
#
# NB: we call snapshot_download directly rather than the `hf` CLI. The CLI lives in
# huggingface_hub >= 1.0, and installing it would break transformers 4.41.2, which
# requires hub < 1.0. Learned the hard way on the 2026-07-08 spike.
from huggingface_hub import snapshot_download

snapshot_download("Soul-AILab/SoulX-Singer",
                  local_dir="/soulx/pretrained_models/SoulX-Singer")
snapshot_download("Soul-AILab/SoulX-Singer-Preprocess",
                  local_dir="/soulx/pretrained_models/SoulX-Singer-Preprocess")
print("soulx weights baked", flush=True)
