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

**Audio never travels inside the JSON.** RunPod's gateway rejects any `/run` body
over **10 MiB** (`bad request: body: exceeded max body size of 10MiB` — served as a
400, or racily as a 502 from its edge, for the identical body), and the SDK warns
that a result over **20 MB** belongs in storage. A mono 44.1 kHz WAV crosses 10 MiB
at **~89 seconds**; a cover is a whole song. Timbrica therefore passes short-lived
signed URLs and the worker streams the bytes itself (`svc_io.py`, stdlib only).

**Input** (`event.input`):

| field | type | notes |
|---|---|---|
| `source_url` | string | https URL to GET the sung performance (WAV) |
| `target_url` | string | https URL to GET the target-voice reference (WAV) |
| `result_put_url` | string | optional; https URL to PUT the converted WAV |
| `source_b64` / `target_b64` | string | fallback for short clips when no `*_url` is given |
| `semi_tone_shift` | int | optional, −12…12 (key matching), default 0 |
| `diffusion_steps` | int | optional, 10…100, default 40 (30–50 best for singing) |

`*_url` wins over `*_b64`. Fetches retry 3× on 5xx/network; a 4xx is a verdict
(expired signature, wrong job) and fails immediately.

**Output**: `{ sample_rate, gen_seconds, engine, watermarked, bytes }` plus either
`uploaded: true` (the WAV was PUT to `result_put_url`) or `audio_b64` (inlined).
On failure: `{ error, detail }` — `transfer_failed` for the network/signature class,
`convert_failed` for the model.

Audio prep (decode, mono, loudness) stays on the Laravel side (`VoiceCoverJob`);
this worker is a thin GPU executor. The model loads **once** at boot, so warm
requests skip the load.

## Pitch semantics — `0` means `0`, in every engine

The caller mixes our vocal back under **their own instrumental**, so any transposition we
apply that they did not ask for returns an out-of-tune mix. Therefore:

| `semi_tone_shift` | behaviour |
|---|---|
| `0` (default) | key untouched — the source's own pitch contour is the melody |
| `-12`..`12` | shift by exactly that many semitones |
| `"auto"` | engine matches the reference clip's register (SoulX only) |

**Timbre transfer is register-coupled in SoulX, and it is not in seed-vc.** Same input
pair (male performance 167 Hz, female reference 333 Hz), f0-invariant cepstral envelope,
cosine similarity to the reference (baseline source↔reference = +0.383):

| run | output f0 | key | →source | →reference |
|---|---|---|---|---|
| seed-vc, `0` | 167 Hz | in key | +0.568 | **+0.893** |
| SoulX, `0` | 165 Hz | in key | +0.889 | +0.505 |
| SoulX, `12` | 333 Hz | +12 st | +0.420 | **+0.980** |

Explicit `12` reproduces the old `auto_shift` result exactly — `auto_shift` was only ever
"pick +12". So SoulX transfers timbre well *only when the registers line up*; seed-vc does
not need that. In the product, reference and performance are the same person, so they do
line up — but this is why `seedvc` is the default engine.

SoulX shipped with `auto_shift=(shift == 0)`, i.e. the *default* silently retuned the
performance into the **reference clip's** register. Our reference is a short spoken
liveness phrase whose pitch has nothing to do with the song, so seed-vc returned the
song in key and SoulX returned it up an octave — same UI, same `0`, different music.
Caught by measuring median f0 of both outputs (167 Hz vs 333 Hz) against the source.
seed-vc always passed `auto_f0_adjust=False`; SoulX now matches it.

## Provenance watermark

Every output carries an inaudible [Perth](https://github.com/resemble-ai/perth)
neural watermark marking it as AI-generated and keeping it traceable. This is our
anti-abuse measure and satisfies the machine-readable marking required by **EU AI
Act Art. 50** (enforceable 2026-08-02). Best-effort: the watermark never gates a
paid result. See `docs/voice-cover-legal.md` in the main repo.

## Build

GitHub Actions builds a matrix — one image per engine — and publishes
`ghcr.io/<owner>/timbrica-svc-worker:{seedvc,soulx}`. Each RunPod serverless endpoint
pulls its own tag. No local Docker needed. Checkpoints are baked into the images, so
a cold worker never downloads weights at request time.


## Operations — hard-won endpoint settings

Verified live on 2026-07-08. Getting these wrong costs hours of misdiagnosis.

| Setting | Value | Why |
|---|---|---|
| `workersMin` | `0` | Anything higher bills continuously against the **shared** RunPod balance that the live `voice-clone` endpoint also draws from. |
| `idleTimeout` | **≥ 30 s** | At `15 s` RunPod reaped the `ready` worker before the scheduler handed it the queued job: `workers.idle=1, jobs.inQueue=1` forever, and the caller only ever saw a poll timeout. Two production runs failed this way before the timeout was raised. |
| `gpuTypeIds` | only types with real capacity | `A40 / A4000 / A4500` reported `od=None` (no capacity) and produced `workers.throttled=1`. Check `gpuTypes.lowestPrice.uninterruptablePrice != null` before listing a type. |
| image ref | pinned by **digest** | A moving tag must never change what a paid conversion produces. |

**Cold start dominates, and it differs per engine** — all measured on live endpoints:

| Engine | Image | Cold start | Conversion (28 s vocal) | `poll_max_s` |
|---|---|---|---|---|
| `seedvc` | ~8 GB | ~98 s | 3–5 s | 480 s |
| `soulx` | ~13 GB (2.8 GB ckpt) | **313 / 587 / 751 s** | 12–19 s | **1080 s** |

The SoulX cold pull is **not a stable number** — three fresh hosts gave 313 s, 587 s and
751 s. At the original 780 s deadline the slowest of those cleared by 28 s; one slower
pull and the caller is refunded instead of sung to. Do not tune this deadline against a
single lucky measurement.

A warm worker answers in seconds; the pull is the cost. Laravel gives each engine its
own deadline (`config/paid-tools.php` → `voice_cover.engines.<id>.poll_max_s`, merged
over the shared default) and `VoiceCoverJob::$timeout` must exceed the largest of
them, or a dying queue worker strands the user's token hold. A job that outlives its
deadline refunds the user in full.

Shrinking the images (weights on a network volume instead of baked in) is the obvious
next lever if cold-start latency ever becomes the complaint.

`workersStandby` mirrors `workersMax` and is **not** settable via REST or GraphQL —
it is derived, not an always-on worker count. Throttled workers do not bill.
