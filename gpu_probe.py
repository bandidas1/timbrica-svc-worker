# Can this machine actually run our kernels, and which machine is it?
#
# Why this exists. On 2026-08-18 the stem-split worker was taught to name its own
# host and immediately answered a month-old mystery: RunPod satisfies an endpoint's
# "24 GB" GPU CLASSES with slices of "NVIDIA RTX PRO 6000 Blackwell Server Edition
# MIG 1g.24gb" — compute capability sm_120. Images built against CUDA ≤ 12.4 have no
# kernels for it, so every job that lands there dies at the first kernel launch with
# "CUDA error: no kernel image is available for execution on the device".
#
# Both SVC images are in that band (seed-vc torch 2.4/cu124, SoulX torch 2.2/cu121),
# and /voice-cover logged 22 `svc_timeout` failures in August — 21 of them on the two
# days the stem splitter started failing the same way. That correlation is the reason
# this module is here; it is NOT yet proof, and proving it is exactly what the module
# is for. Until a worker can say what it is running on, every such failure looks like
# "the GPU was slow" and teaches nobody anything.
#
# Deliberately stdlib + torch only: the seed-vc and SoulX dependency stacks are
# brittle and mutually incompatible (same reasoning as svc_io.py).

__all__ = ["host_report", "GpuUnusable", "assert_gpu_usable", "probe_result"]


def host_report() -> dict:
    """Everything needed to tell one machine from another, best-effort."""
    rep = {}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"torch_import_error": str(exc)[:160]}

    rep["torch"] = torch.__version__
    rep["cuda"] = torch.version.cuda
    try:
        rep["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # noqa: BLE001
        rep["device"] = "unknown"
        rep["available_error"] = str(exc)[:160]
    if rep.get("device") != "cuda":
        return rep
    try:
        rep["gpu"] = torch.cuda.get_device_name(0)
        rep["capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
        rep["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception as exc:  # noqa: BLE001
        rep["probe_error"] = str(exc)[:160]
    try:
        with open("/proc/driver/nvidia/version") as fh:
            rep["nvidia"] = fh.readline().strip()[:120]
    except Exception:  # noqa: BLE001
        pass
    return rep


class GpuUnusable(RuntimeError):
    """This host cannot launch a kernel from this image. Not the job's fault."""


def assert_gpu_usable() -> None:
    """
    Launch the smallest possible real kernel and read the result back.

    `.item()` is the point: it synchronises, so a CUDA error that would otherwise
    surface minutes later — in the middle of a paid conversion — surfaces here, in
    about a second, before anything has been downloaded or charged.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return  # CPU box (local mock): nothing to prove
        a = torch.ones(64, 64, device="cuda")
        (a @ a).sum().item()
    except GpuUnusable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GpuUnusable(str(exc)[:160]) from exc


def probe_result() -> dict:
    """Answer for `{"input": {"probe": true}}` — who am I, and can I compute.

    A free question. Before it existed, the only way to ask a host anything was to
    spend a user's paid job on it.
    """
    rep = {"probe": True, "host": host_report()}
    try:
        assert_gpu_usable()
        rep["usable"] = True
    except GpuUnusable as exc:
        rep["usable"] = False
        rep["reason"] = str(exc)[:160]
    return rep
