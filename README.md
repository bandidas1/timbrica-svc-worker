# timbrica-svc-worker

RunPod **serverless** workers for **singing voice conversion (SVC)** — the engine
behind the `/voice-cover` tool ("sing a song in your own voice") on
[Timbrica](https://timbrica.com).

Takes a **sung performance** (melody, timing, lyrics) plus a **target-voice
reference** (timbre) and returns the performance re-sung in that voice.

## Engines

One image per engine — their dependency stacks conflict (seed-vc needs torch 2.4,
SoulX needs torch 2.2 + NeMo), so they can't share a venv.

| Tag | Model | License | Output |
|---|---|---|---|
| `:seedvc` | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | GPLv3 | 44.1 kHz, full band |
| `:soulx` | [Soul-AILab/SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) | Apache-2.0 | 24 kHz |

Both were verified on real audio before this worker was written (rented RTX 3090,
2026-07-08): each converts singing while preserving melody, phrasing and timing
from a 7–11 s zero-shot reference.

**GPLv3 note:** seed-vc runs only here, on our own GPU, and is never distributed to
users. GPLv3 (unlike AGPL) has no network clause, so no copyleft obligation is
triggered by offering it as a service. SoulX is Apache-2.0 (code *and* weights).

**SoulX specifics.** Its CLI expects a precomputed F0 contour (`*.npy`) for both
audios, so the handler runs the upstream RMVPE extractor per request, at upstream's
defaults (24 kHz grid, hop 480) — the grid the checkpoint was trained on. Watch the
naming: upstream calls the *voice reference* `prompt/pt` and the *audio to convert*
`target/gt`, which is the opposite of this worker's `target_b64` / `source_b64`.
`requirements-soulx.txt` is a **slim** set: upstream additionally pins nemo_toolkit,
sageattention, gradio, torchcodec, webrtcvad and the g2p/MIDI tooling, none of which
appear in `sys.modules` after the model + RMVPE load — verified on a real GPU, and
the handler runs green without them.

## Contract

**Input** (`event.input`):

| field | type | notes |
|---|---|---|
| `source_b64` | string | the sung performance, WAV bytes, base64 |
| `target_b64` | string | the target-voice reference, WAV bytes, base64 |
| `semi_tone_shift` | int | optional, −12…12 (key matching), default 0 |
| `diffusion_steps` | int | optional, 10…100, default 40 (30–50 best for singing) |

**Output**: `{ audio_b64 (WAV, base64), sample_rate, gen_seconds, engine, watermarked }`
or `{ error, detail }`.

Audio prep (decode, mono, loudness) stays on the Laravel side (`VoiceCoverJob`);
this worker is a thin GPU executor. The model loads **once** at boot, so warm
requests skip the load.

## Provenance watermark

Every output carries an inaudible [Perth](https://github.com/resemble-ai/perth)
neural watermark marking it as AI-generated and keeping it traceable. This is our
anti-abuse measure and satisfies the machine-readable marking required by **EU AI
Act Art. 50** (enforceable 2026-08-02). Best-effort: the watermark never gates a
paid result. See `docs/voice-cover-legal.md` in the main repo.

## Build

GitHub Actions builds on every push to `main` and publishes to
`ghcr.io/<owner>/timbrica-svc-worker:seedvc`. The RunPod serverless endpoint pulls
that image. No local Docker needed. Checkpoints are baked into the image for
reliable cold starts.


## Operations — hard-won endpoint settings

Verified live on 2026-07-08. Getting these wrong costs hours of misdiagnosis.

| Setting | Value | Why |
|---|---|---|
| `workersMin` | `0` | Anything higher bills continuously against the **shared** RunPod balance that the live `voice-clone` endpoint also draws from. |
| `idleTimeout` | **≥ 30 s** | At `15 s` RunPod reaped the `ready` worker before the scheduler handed it the queued job: `workers.idle=1, jobs.inQueue=1` forever, and the caller only ever saw a poll timeout. Two production runs failed this way before the timeout was raised. |
| `gpuTypeIds` | only types with real capacity | `A40 / A4000 / A4500` reported `od=None` (no capacity) and produced `workers.throttled=1`. Check `gpuTypes.lowestPrice.uninterruptablePrice != null` before listing a type. |
| image ref | pinned by **digest** | A moving tag must never change what a paid conversion produces. |

**Cold start** is ~50–100 s (the image is ~8–9 GB). The conversion itself is ~3–5 s
for a 28 s vocal on a 3090 (≈5× realtime), so a warm worker answers in seconds. The
Laravel side polls for up to `poll_max_s = 480 s`, which comfortably covers a cold
pull; a job that outlives it refunds the user in full.

`workersStandby` mirrors `workersMax` and is **not** settable via REST or GraphQL —
it is derived, not an always-on worker count. Throttled workers do not bill.
