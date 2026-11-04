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

# ── Retry policy (2026-11-04) ────────────────────────────────────────────────
# A finished cover exists only on this worker until the PUT lands on Timbrica,
# and until 2026-11-04 the policy was «3 tries, 1.5 s and 3 s apart, every 4xx
# is a verdict»: a 30-second blip on the way to the origin — or nginx's 408 for
# a body that went silent — failed the job after the GPU work was done, and the
# app refunded it. The same class lost a finished /video-upscaler file on
# 2026-11-01 (the network from RunPod to the RU origin failed for half an hour)
# and showed up as nginx 408s on every 50th request to that box on 28.10.
# The rules, shared with the upscale and stem workers:
#   · silence is noticed in _STALL_S — nginx drops a silent body at 60 s, so a
#     longer socket wait only waits on a request that is already dead;
#   · no answer at all, 408, 425, 429, any 5xx and a body the app saw cut short
#     (400 …truncated) are retried; every other 4xx is the app's final word;
#   · retries go on until the app's own deadline (`deadline_at` in the input,
#     epoch seconds — past it the app has stopped waiting and refunded), and
#     never longer than _PATIENCE_S when an older app sends no deadline.
_STALL_S = 60
_PATIENCE_S = 600
_GET_TIMEOUT_S = _STALL_S
_PUT_TIMEOUT_S = _STALL_S
_UA = "timbrica-svc-worker/1"
# Indirection for the self-check (svc_delivery_selftest.py): a fake clock.
_clock, _sleep, _now = time.monotonic, time.sleep, time.time


class TransferError(RuntimeError):
    """Audio could not be moved in or out. Distinct from a conversion failure:
    the caller releases the token hold either way, but the two want different
    operator responses (network/signature vs model)."""


def _retryable(code, body=""):
    if code == 0 or code in (408, 425, 429) or code >= 500:
        return True
    return code == 400 and "truncated" in (body or "")


def _pause(attempt):
    return (2, 5, 10, 20)[attempt] if attempt < 4 else 30


def _deadline_of(inp):
    try:
        return float((inp or {}).get("deadline_at") or 0)
    except (TypeError, ValueError):
        return 0.0


def _retrying(what, fn, deadline_at=0.0, ok_codes=()):
    """Run `fn` until it answers, the app's deadline is near, or patience is out.
    An HTTP code in `ok_codes` counts as success (returned as the status)."""
    started = _clock()
    attempt = 0
    last = None
    while True:
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code in ok_codes:
                return e.code
            try:
                body = e.read(300).decode("utf-8", "replace")
            except Exception:
                body = ""
            if not _retryable(e.code, body):
                # 403 signature, 404/410 job gone, 413 size, 422 wrong file:
                # sending the same bytes again cannot change the answer.
                raise TransferError(f"{what}_http_{e.code}") from e
            last = f"http_{e.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = repr(e)[:120]
        pause = _pause(attempt)
        attempt += 1
        if _clock() - started + pause >= _PATIENCE_S:
            raise TransferError(f"{what}_unreachable: {last}")
        # Never start a try that, silent to the end, would outlive the deadline.
        if deadline_at and _now() + pause + _STALL_S + 5 >= deadline_at:
            raise TransferError(f"{what}_past_deadline: {last}")
        print(f"[svc_io] {what} try {attempt} failed ({last}); again in {pause}s", flush=True)
        _sleep(pause)


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

        raw = _retrying(f"{key}_fetch", _get, _deadline_of(inp))
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

    # 409 «result_already_uploaded»: an earlier try landed and only its answer
    # was lost — the result is on the app's disk, which is delivery.
    _retrying("result_put", _put, _deadline_of(inp), ok_codes=(409,))
    return dict(extra, uploaded=True, bytes=len(wav_bytes))
