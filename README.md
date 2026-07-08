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
| `:soulx` *(planned)* | [Soul-AILab/SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) | Apache-2.0 | 24 kHz |

Both were verified on real audio before this worker was written (rented RTX 3090,
2026-07-08): each converts singing while preserving melody, phrasing and timing
from a 7–11 s zero-shot reference.

**GPLv3 note:** seed-vc runs only here, on our own GPU, and is never distributed to
users. GPLv3 (unlike AGPL) has no network clause, so no copyleft obligation is
triggered by offering it as a service.

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
