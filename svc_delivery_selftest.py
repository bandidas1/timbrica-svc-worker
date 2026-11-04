"""Self-check of svc_io's transfer retries — no GPU, no network.

Why: a finished cover exists only on this worker until its PUT lands, and until
2026-11-04 the policy was three tries 1.5 s and 3 s apart with every 4xx a
verdict. A short blip or nginx's 408 failed the job after the GPU work was
done. This pins the new policy against fake requests and a fake clock:

  python svc_delivery_selftest.py
"""
import io
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import svc_io  # noqa: E402  (stdlib-only by design)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def mono(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)

    def wall(self):
        return 1_800_000_000 + self.t


FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def http_error(code, body=b""):
    return urllib.error.HTTPError("https://x", code, "x", {}, io.BytesIO(body))


def run(script, deadline_in=None, ok_codes=()):
    """`script(n)` → 'ok' | 'drop' | HTTPError — the fate of the n-th call."""
    c = Clock()
    svc_io._clock, svc_io._sleep, svc_io._now = c.mono, c.sleep, c.wall
    calls = {"n": 0}

    def fn():
        n = calls["n"]
        calls["n"] += 1
        f = script(n)
        if f == "ok":
            return 200
        if f == "drop":
            c.sleep(svc_io._STALL_S)
            raise urllib.error.URLError(TimeoutError("timed out"))
        raise f

    try:
        out, err = svc_io._retrying("result_put", fn, (c.wall() + deadline_in) if deadline_in else 0.0, ok_codes), None
    except svc_io.TransferError as e:
        out, err = None, str(e)
    return out, err, calls["n"], c.t - 1000


out, err, n, t = run(lambda n: "ok")
check("clean: one call", out == 200 and n == 1)

out, err, n, t = run(lambda n: http_error(408) if n == 0 else "ok")
check("408 is a hiccup: retried", out == 200 and n == 2, (out, err))

out, err, n, t = run(lambda n: http_error(400, b"result_truncated") if n == 0 else "ok")
check("400 …truncated is retried", out == 200, (out, err))

out, err, n, t = run(lambda n: http_error(409), ok_codes=(409,))
check("409 result_already_uploaded on the PUT = delivered", out == 409 and n == 1, (out, err))

out, err, n, t = run(lambda n: http_error(503) if n < 6 else "ok")
check("503 storm is ridden out (old policy: dead on the third)", out == 200 and n == 7, (out, err, n))

out, err, n, t = run(lambda n: "drop" if n < 3 else "ok")
check("three silent sockets, then the link is back", out == 200, (out, err))

for code in (403, 404, 410, 413, 422):
    out, err, n, t = run(lambda n, code=code: http_error(code))
    check(f"{code} is final at once", err == f"result_put_http_{code}" and n == 1, (err, n))

out, err, n, t = run(lambda n: "drop")
check("endless outage gives up", err and err.startswith("result_put_unreachable"), err)
check("…not before patience", t >= svc_io._PATIENCE_S - 31, t)

out, err, n, t = run(lambda n: "drop", deadline_in=300)
check("the app's deadline ends the retries in time", err and err.startswith("result_put_past_deadline") and t <= 300, (err, t))

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all svc transfer checks passed'}")
sys.exit(1 if FAILS else 0)
