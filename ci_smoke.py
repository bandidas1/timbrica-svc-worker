"""Does this image actually start?

Run INSIDE the built image (see .github/workflows/build.yml):

    docker run --rm --entrypoint python <image> /ci_smoke.py

Why it exists. On 2026-08-18 both SVC endpoints spent three days with workers
that never reached "ready" — jobs queued, nothing dispatched, $19.88 billed for
28.7 worker-hours and zero completed jobs. A control run proved the endpoint and
the hosts were fine: the same endpoint booted a different image in 425 s. The
image could not start, and there was nowhere to see why — RunPod shows worker
logs only in its console, and a container that dies before the SDK starts leaves
no trace in any API.

So the check lives here, where it is free and repeatable. It imports the engine's
own package — the exact place a torch or CUDA change explodes (pinned deps built
against another ABI) — WITHOUT loading weights, so it costs seconds and no GPU.

It prints a sentinel on success. The workflow greps for that sentinel, because a
step that produces NO output must never count as a pass: the first version of
this check was a shell heredoc, it silently ran nothing, and the build went
green while the image was still unbootable.
"""

import os
import sys
import traceback
import types

SENTINEL = "IMAGE-STARTS-OK"


def step(name, fn):
    try:
        fn()
    except BaseException:
        print("  FAIL %s" % name, flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("  ok   %s" % name, flush=True)


def main() -> None:
    import numpy
    import soundfile  # noqa: F401
    import torch
    import torchaudio

    print("torch=%s torchaudio=%s cuda=%s numpy=%s"
          % (torch.__version__, torchaudio.__version__, torch.version.cuda, numpy.__version__),
          flush=True)

    sys.path.insert(0, "/")
    step("gpu_probe", lambda: __import__("gpu_probe"))
    step("svc_io", lambda: __import__("svc_io"))

    root = os.environ.get("SEED_VC_DIR") or os.environ.get("SOULX_DIR")
    if root and os.path.isdir(root):
        sys.path.insert(0, root)
        os.chdir(root)
        # seed-vc imports gradio at module scope but only uses it in the web UI,
        # which the handler never calls — the handler stubs it the same way.
        sys.modules.setdefault("gradio", types.ModuleType("gradio"))
        mod = "app_svc" if "seed" in root else "soulxsinger"
        step("%s (engine package)" % mod, lambda: __import__(mod))
    else:
        print("  ??   engine dir not found (%r) — engine import NOT verified" % root, flush=True)
        sys.exit(1)

    print(SENTINEL, flush=True)


if __name__ == "__main__":
    main()
