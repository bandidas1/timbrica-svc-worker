# Bulk-audio transport for the SVC workers.
#
# RunPod's job API is a CONTROL plane, and it enforces that: the gateway rejects any
# /run body over 10 MiB ("bad request: body: exceeded max body size of 10MiB" —
# returned as 400, or racily as a 502 from its edge, for the very same body), and the
# runpod SDK warns that a job result above 20 MB belongs in storage rather than in the
# result JSON. A 44.1 kHz mono WAV crosses 10 MiB at ~89 seconds of audio. A cover is
# a whole song. Base64-in-JSON therefore cannot carry this tool's payload, and never
# could: every production cover failed at submit until this module existed.
#
# So audio travels out of band. Timbrica mints short-lived signed URLs and passes them
# in `event.input`: one GET per input audio, one PUT for the result. The JSON stays a
# few hundred bytes in each direction and there is no size ceiling on the audio.
#
# `*_b64` in / `audio_b64` out remain supported: short clips, the local mock path, and
# anything that predates this contract keep working unchanged.
#
# stdlib only, on purpose. The seed-vc (torch 2.4) and SoulX (torch 2.2 + NeMo) images
# have brittle, mutually incompatible dependency stacks; an HTTP GET and an HTTP PUT
# do not justify adding a package to either.

import base64
import tempfile
import time
import urllib.error
import urllib.request

__all__ = ["TransferError", "fetch_audio", "write_tmp", "deliver"]

_RETRIES = 3
_BACKOFF_S = 1.5
_GET_TIMEOUT_S = 120
_PUT_TIMEOUT_S = 300
_UA = "timbrica-svc-worker/1"


class TransferError(RuntimeError):
    """Audio could not be moved in or out. Distinct from a conversion failure:
    the caller releases the token hold either way, but the two want different
    operator responses (network/signature vs model)."""


def _retrying(what, fn):
    last = None
    for attempt in range(1, _RETRIES + 1):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            # 4xx is a verdict (expired signature, wrong job): retrying cannot help.
            if e.code < 500:
                raise TransferError(f"{what}_http_{e.code}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        if attempt < _RETRIES:
            time.sleep(_BACKOFF_S * attempt)
    raise TransferError(f"{what}_unreachable: {repr(last)[:120]}")


def fetch_audio(inp, key, max_bytes):
    """Return the raw bytes of input `key`, from `{key}_url` (preferred) or `{key}_b64`.

    Reads at most max_bytes + 1 so an oversized body is rejected instead of filling
    the worker's disk.
    """
    url = inp.get(f"{key}_url")
    if url:
        if not isinstance(url, str) or not url.startswith("https://"):
            raise TransferError(f"{key}_url_not_https")

        def _get():
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=_GET_TIMEOUT_S) as r:
                return r.read(max_bytes + 1)

        raw = _retrying(f"{key}_fetch", _get)
    else:
        b64 = inp.get(f"{key}_b64")
        if not b64 or not isinstance(b64, str):
            raise TransferError(f"{key}_required")
        raw = base64.b64decode(b64, validate=True)

    if len(raw) > max_bytes:
        raise TransferError(f"{key}_too_large")
    if len(raw) < 1000:
        raise TransferError(f"{key}_too_small")
    return raw


def write_tmp(raw, prefix):
    f = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", delete=False)
    f.write(raw)
    f.close()
    return f.name


def deliver(wav_bytes, inp, extra):
    """Hand the converted WAV back: PUT to `result_put_url` when Timbrica supplied
    one, else inline base64 (small clips / mock). Returns the handler's result dict.

    The `bytes` field lets the caller verify that what landed on its disk is what the
    worker produced — a truncated upload must not be served as a finished cover.
    """
    put_url = inp.get("result_put_url")
    if not put_url:
        return dict(extra, audio_b64=base64.b64encode(wav_bytes).decode("ascii"), bytes=len(wav_bytes))

    if not isinstance(put_url, str) or not put_url.startswith("https://"):
        raise TransferError("result_put_url_not_https")

    def _put():
        req = urllib.request.Request(
            put_url,
            data=wav_bytes,
            method="PUT",
            headers={
                "Content-Type": "audio/wav",
                "Content-Length": str(len(wav_bytes)),
                "User-Agent": _UA,
            },
        )
        with urllib.request.urlopen(req, timeout=_PUT_TIMEOUT_S) as r:
            return r.status

    _retrying("result_put", _put)
    return dict(extra, uploaded=True, bytes=len(wav_bytes))
