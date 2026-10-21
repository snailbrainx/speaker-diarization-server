# Speaker Diarization Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

GPU-accelerated speaker diarization, transcription, and emotion recognition with a REST API, WebSocket streaming, and an MCP server for AI agents. Combines pyannote.audio + faster-whisper or NVIDIA Parakeet + emotion2vec into one FastAPI service with persistent speaker identity across conversations.

## Screenshots

The server itself is headless: everything goes through the REST API, the WebSocket stream and the MCP server.
The screenshots below are from an older version of the separate example
[Next.js frontend](https://github.com/snailbrainx/speaker_identity_nextjs) and may not match the current API or features.

<table>
  <tr>
    <td align="center"><img src="screenshots/1.png" width="300" alt="Voice Profile Management"/><br/>Voice Profile Management</td>
    <td align="center"><img src="screenshots/2.png" width="300" alt="Process Audio"/><br/>Process Audio</td>
    <td align="center"><img src="screenshots/3.png" width="300" alt="Conversation Detail"/><br/>Conversation Detail</td>
  </tr>
  <tr>
    <td align="center"><img src="screenshots/4.png" width="300" alt="Conversations List"/><br/>Conversations List</td>
    <td align="center"><img src="screenshots/5.png" width="300" alt="Speaker Management"/><br/>Speaker Management</td>
    <td align="center"><img src="screenshots/6.png" width="300" alt="Live Recording"/><br/>Live Recording</td>
  </tr>
</table>

## Features

- **Persistent speaker identity** — enroll a voice once, recognized across every future recording.
- **Retroactive updates** — identify one segment and every past segment with that voice re-labels automatically.
- **Transcription** — faster-whisper (`large-v3` default, any Whisper variant, 99 languages) or NVIDIA Parakeet-TDT (`ASR_BACKEND=parakeet`: English, ~4× faster speech-to-text, lower WER on meetings), word-level timestamps.
- **Dual-detector emotion recognition** — emotion2vec+ (general) combined with per-speaker voice profiles (personalized). 9 categories.
- **Live streaming** — WebSocket audio ingestion with VAD-gated segment flushing.
- **Batch processing** — multi-GPU script for bulk ingest (`run_batch.sh`).
- **REST API** at `/api/v1/*` with interactive Swagger docs at `/docs`.
- **MCP server** at `/mcp` (JSON-RPC 2.0 / HTTP) exposing 11 tools for Claude Desktop, Flowise, or any MCP client.
- **Backup/restore** — export speakers + segments + settings as JSON profiles and checkpoints.
- **SQLite WAL** — handles thousands of conversations with concurrent reads.

## Tech Stack

| Component       | Implementation                                                              |
|-----------------|-----------------------------------------------------------------------------|
| Diarization     | `pyannote/speaker-diarization-community-1` (pyannote.audio 4.0.1)           |
| Speaker embed   | `pyannote/embedding` (512-D)                                                |
| Transcription   | faster-whisper 1.2.1 / CTranslate2, `large-v3` default; or Parakeet-TDT 0.6B v2 via NVIDIA NeMo 3.0 (`ASR_BACKEND=parakeet`) |
| Emotion         | emotion2vec_plus_large via FunASR (1024-D, 9 categories)                    |
| API             | FastAPI 0.141.1 + WebSockets + uvicorn                                      |
| DB              | SQLAlchemy 2.0 / SQLite (WAL)                                               |
| ML runtime      | PyTorch 2.10 + CUDA 12.8 (Blackwell/RTX 5090 `sm_120` ready)                |
| MCP transport   | JSON-RPC 2.0 over HTTP (+ optional SSE), no external MCP library            |

## Requirements

- **GPU:** NVIDIA with CUDA 12.x. Tested on RTX 3090 (24 GB) and RTX 5090. Runs on 6 GB+ cards if you pick a smaller Whisper model.
- **VRAM ballpark:** 2–3 GB diarization/embedding + 2 GB emotion2vec + Whisper (0.5–4 GB depending on model). `large-v3` + emotion ≈ 8–9 GB total (turbo ≈ 6–7 GB).
- **OS:** Linux (tested on Ubuntu). macOS/Windows via Docker Desktop + WSL2.
- **Other:** ffmpeg, git. Python 3.11 or 3.12 if running outside Docker.
- **HuggingFace token** with pyannote terms accepted (see below).

## Quick Start

### 1. HuggingFace access

Create an account + token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then accept the terms for:
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [pyannote/embedding](https://huggingface.co/pyannote/embedding)

### 2. Docker (recommended)

```bash
git clone https://github.com/snailbrainx/speaker-diarization-server.git
cd speaker-diarization-server
cp .env.example .env           # then edit .env and set HF_TOKEN
docker compose up --build      # first run downloads ~3-5 GB of models
```

### 3. Local development

```bash
sudo apt-get install -y ffmpeg git
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # edit HF_TOKEN
./run_local.sh
```

### 4. Verify

- Swagger UI: `http://<host>:8418/docs`
- Health: `curl http://<host>:8418/api/v1/status`
- MCP: `http://<host>:8418/mcp`

### Remote access

No built-in auth. Use an SSH tunnel for remote hosts:

```bash
ssh -L 8418:localhost:8418 user@host
```

For network deployments put it behind a reverse proxy (nginx + HTTPS + auth). WebSocket microphone capture in browsers **requires HTTPS/WSS** — place `cert.pem`/`key.pem` in `certs/` and the server starts in HTTPS mode automatically.

## Configuration

All configuration is environment variables (see `.env.example` for the curated list). Everything is optional except `HF_TOKEN`.

### Required

| Variable    | Purpose                                   |
|-------------|-------------------------------------------|
| `HF_TOKEN`  | HuggingFace token for pyannote model access |

### Storage & networking

| Variable         | Default                                | Purpose                                                        |
|------------------|----------------------------------------|----------------------------------------------------------------|
| `DATABASE_URL`   | `sqlite:////app/volumes/speakers.db`   | SQLAlchemy URL. Defaults assume the Docker volume layout.      |
| `DATA_PATH`      | `./data`                               | Root for recordings, stream segments, temp files.              |
| `VOLUMES_PATH`   | `./volumes`                            | Root for DB + model cache.                                     |
| `PORT`           | `8418`                                 | HTTP/HTTPS port.                                               |
| `CORS_ORIGINS`   | *(unset)*                              | Comma-separated origins. Leave empty for internal-only use.    |
| `LOG_LEVEL`      | `INFO`                                 | Python logging level.                                          |

### Live streaming limits

| Variable                          | Default                          | Purpose                                                   |
|-----------------------------------|----------------------------------|-----------------------------------------------------------|
| `MAX_STREAM_BUFFER_SECONDS`       | `120` (`300` in docker-compose)  | Longest speech buffered before a segment is force-flushed. |
| `STREAM_LONG_TURN_SECONDS`        | `15`                             | Speech with no pause this long (noisy rooms, crosstalk) is cut at its quietest point and processed instead of waiting for a pause. `0` = wait for a pause. |
| `SEGMENT_HANDLER_TIMEOUT_SECONDS` | `600`                            | How long a live segment may take to process.              |
| `WS_SEND_TIMEOUT_SECONDS`         | `10` (`5` in docker-compose)     | Timeout for sending a message to a WebSocket client.      |

### Voice & emotion tuning (also editable at runtime via `/api/v1/settings/voice`)

| Variable                 | Default  | Purpose                                                        |
|--------------------------|----------|----------------------------------------------------------------|
| `SPEAKER_THRESHOLD`      | `0.30`   | Cosine threshold for speaker match. Higher → stricter.         |
| `CONTEXT_PADDING`        | `0.15`   | Seconds of audio added around a segment before embedding.      |
| `SILENCE_DURATION`       | `0.5`    | Streaming: seconds of silence before flushing a segment.       |
| `FILTER_HALLUCINATIONS`  | `true`   | Drop Whisper filler phrases ("thank you.", etc.).              |
| `EMOTION_THRESHOLD`      | `0.6`    | Global emotion-match sensitivity. Higher → stricter.           |

### Models

| Variable                       | Default                          | Purpose                                             |
|--------------------------------|----------------------------------|-----------------------------------------------------|
| `WHISPER_MODEL`                | `large-v3`                 | Any faster-whisper model ID. See `.env.example`.    |
| `WHISPER_LANGUAGE`             | `en`                             | `auto` for 99-language detection, or ISO-639 code.  |
| `ASR_BACKEND`                  | `whisper`                        | Speech-to-text engine: `whisper` or `parakeet` (see [Speech-to-text](#speech-to-text-whisper-or-parakeet)). |
| `PARAKEET_MODEL`               | `nvidia/parakeet-tdt-0.6b-v2`    | `ASR_BACKEND=parakeet` model. v2 is English-only; `nvidia/parakeet-tdt-0.6b-v3` covers 25 European languages (detected automatically). |
| `EMOTION_MODEL`                | `iic/emotion2vec_plus_large`     | FunASR emotion model ID.                            |
| `ENABLE_PERSONALIZED_EMOTIONS` | `true`                           | Use per-speaker voice profiles in emotion matching. |
| `OFFLINE_MODE`                 | `false`                          | Force cached models only; never hit the hub.        |

### GPU

| Variable                    | Default       | Purpose                                                         |
|-----------------------------|---------------|-----------------------------------------------------------------|
| `CUDA_VISIBLE_DEVICES`      | *(unset)*     | Select physical GPUs. `0`, `0,2`, etc.                          |
| `CUDA_DEVICE_ORDER`         | `PCI_BUS_ID`  | Pinned by `run_local.sh` and `docker-compose.yml` so indices match `nvidia-smi` instead of CUDA's "fastest first" reshuffle. |
| `CLEANUP_VRAM_THRESHOLD_GB` | `12`          | Only run the VRAM cleanup loop when allocated memory exceeds this. |

### Speed & speaker assignment

On by default. To get the original behaviour back, set `WHISPER_BATCH_SIZE=0`, `WORD_LEVEL_SPEAKERS=false`,
`PYANNOTE_ALLOW_TF32=false`, `SPEAKER_ASSIGNMENT=segment` and `SNAP_TO_TURNS=false`.

| Variable                    | Default   | Purpose                                                                                     |
|-----------------------------|-----------|---------------------------------------------------------------------------------------------|
| `WHISPER_BATCH_SIZE`        | `4` on GPU, `0` on CPU | Batched faster-whisper (VAD chunks in batches): ~8x faster transcription on long audio, no end-of-file repeat hallucination. `0` = sequential decoding. Larger batches use more VRAM (peak ~8 GB at 4, ~10 GB at 8, ~13 GB at 16 on long recordings) without being faster, since diarization is the slower half. |
| `WORD_LEVEL_SPEAKERS`       | `true`    | Each word gets the diarization speaker at its midpoint; segments split where the speaker changes. `false` = one speaker per Whisper segment. |
| `PYANNOTE_ALLOW_TF32`       | `true`    | Keep TF32 on for pyannote (it disables it for reproducibility). Faster on Ampere and newer.   |
| `SPEAKER_ASSIGNMENT`        | `cluster` | `cluster`: identify each diarization speaker **once per file** from all of its speech, one-to-one against stored speakers; an unmatched speaker becomes one `Unknown_*`. Uploaded/reprocessed files only; live chunks keep per-segment matching. `segment` = match every segment on its own (original). |
| `CLUSTER_SPEAKER_THRESHOLD` | `0.70`    | Cluster mode's match threshold (used instead of `SPEAKER_THRESHOLD`, which is tuned for single segments). Higher → stricter. |
| `SNAP_TO_TURNS`             | `true`    | Widen each segment to its speaker's diarization turn, never into the neighbouring segments. Word timestamps cover only the voiced core of each word, so segments that span first to last word leave the speech around them unlabelled. `false` = first to last word (original). |

Measured on public test sets (RTX 3090; AMI = 16 test meetings, VoxConverse = 40 dev clips):

| | original | speed settings only | current defaults (+ cluster) |
|---|---|---|---|
| Speed, AMI (× real time, server-side) | 11.7× | 23.5× | same or slightly faster¹ |
| WER, LibriSpeech test-clean (40 clips) | 2.9% | 1.0% | 1.0% |
| WER, AMI (mixed headset audio) | 24.7% | 24.6% | 24.6% |
| Speaker confusion, VoxConverse / AMI | 11.9% / 24.8% | 10.6% / 18.7% | **2.6% / 2.1%** |
| AMI words attributed to the wrong speaker | 41.2% | 33.8% | **13.3%** |
| Speaker labels per 4-person AMI meeting, nobody enrolled (median, range) | 48 (10–120) | 33 (18–52) | **4 (3–5)** |
| Enrolled speakers in later meetings: right / wrong / unknown² | – | 86.8% / 8.5% / 4.7% | **97.5% / 2.2% / 0.2%** |
| Speech of unenrolled speakers given an enrolled name² | – | 10.3% | **0%** |

¹ 6% less processing time than the speed settings alone, measured side by side on the same GPU. (Measured with
`WHISPER_BATCH_SIZE=16`; batch 4, the default, is equally fast and gives the same transcripts.)
² Each AMI series has the same four people in four meetings: speakers were named in the first meeting with the
identify endpoint, then the other three meetings were processed; a meeting from another series stood in for
unenrolled speakers.

### Speech-to-text: Whisper or Parakeet

`ASR_BACKEND=parakeet` runs NVIDIA's [Parakeet-TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
through NeMo instead of faster-whisper. It returns the same segments (text, word timestamps, word confidences), so
speaker assignment, emotion and storage are unchanged. Long recordings are cut at pauses into chunks of up to 60 s that
cover the whole file, transcribed 8 at a time; sentences are joined into segments like Whisper's. v2 is English-only
(`PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3` covers 25 European languages); Whisper stays the default for its 99
languages.

This release against the previous one, on the same GPU. Speed is server-side processing time, with each file sent to
each version in turn; LibriSpeech = 40 test-clean clips, VoxConverse = 40 dev clips, AMI = the 16 test meetings (9.1 h).

| | previous release | this release, Whisper | this release, Parakeet |
|---|---|---|---|
| Speed, LibriSpeech (median per clip) | 9.2× (0.83 s) | 19.7× (0.34 s) | **52.8× (0.12 s)** |
| Speed, VoxConverse | 32× | 36× | **76×** |
| Speed, AMI | 27× | 45× | **79×** |
| WER, LibriSpeech test-clean | **1.0%** | **1.0%** | 1.4% |
| WER, AMI | 24.7% | 24.7% | **21.2%** |
| DER, VoxConverse | 14.9% | 11.7% | **11.2%** |
| DER, AMI | 31.8% | 25.0% | **23.9%** |
| Speaker confusion, VoxConverse / AMI | **2.6% / 2.1%** | 2.9% / 2.6% | 3.0% / 2.3% |
| AMI words in a segment of the right speaker, of all words | 79.1% | 83.8% | **83.9%** |
| Enrolled speakers' speech in later meetings given their own name³ | 82.4% | – | **89.2%** |

Held-out sets, never used to choose a setting (previous release / this release with Parakeet): WER on 200
LibriSpeech test-clean clips 1.69% / 1.94%, on 200 test-other clips 3.24% / **2.76%**, on the 18 AMI dev meetings
23.7% / **21.8%**; DER on 40 other VoxConverse clips 12.5% / **8.7%**, on AMI dev 30.6% / **24.1%**.

Live, six 150 s VoxConverse clips with 3–5 speakers streamed in real time: median delay from speech to a labelled line
17.6 s → **4.5 s**, worst 116 s → **15 s**; DER 23.2% → **21.7%**, speaker confusion 12.6% → 15.2%.

Speaker confusion rises a little because turn edges label ~8 points more of the speech, and those edges are where
pyannote itself is least sure; DER, which counts confusion, falls 3–8 points. Parakeet trails Whisper on clean read
speech (mostly names and compound words) and leads on harder speech and meetings.

³ Of all their speech, in the identification test above; 2.0% / 2.2% was given another person's name.

## How it Works

```
upload / WS chunk
      │
      ▼
┌─────────────────┐   ┌──────────────────┐
│ Whisper/Parakeet│   │ pyannote         │
│ transcription   │   │ diarization      │   run in parallel
│ (text + words)  │   │ (speaker turns)  │   (ThreadPoolExecutor)
└────────┬────────┘   └────────┬─────────┘
         └────────────┬────────┘
                      ▼
        segment alignment by timestamp
                      │
                      ▼
        pyannote embedding (512-D) per segment
                      │
                      ▼
        cosine match vs known speaker profiles
        (files: once per diarization speaker, from its centroid;
         live chunks: per segment)
        │                              │
        ▼                              ▼
   known speaker                 Unknown_NN  (auto-enrolled)
                      │
                      ▼
        emotion2vec (1024-D) + optional voice-profile emotion match
                      │
                      ▼
             ConversationSegment  →  SQLite (WAL)
```

Key behaviors:

- **Parallel speech-to-text + pyannote** — ~2× faster than sequential.
- **Decoded once** — each file is decoded and resampled to 16 kHz once, and Parakeet, pyannote and the voice embeddings all work on that copy in memory. Given a file path, pyannote re-reads the file for every window it embeds: 2.3× slower on a 17-minute meeting, with identical results.
- **Turn edges** (`SNAP_TO_TURNS`) — a segment's start and end are widened to its speaker's diarization turn (word timestamps cover only the voiced core of the words). Voice embeddings still come from the words' own span, so this changes timing only, never who is identified.
- **One-speaker mode** — `diarize=false` (upload form field, or the live start message) skips pyannote when one person is talking; the speech is still matched against stored voices.
- **Live turns** — speech is processed at each pause; a turn with no pause for `STREAM_LONG_TURN_SECONDS` (noisy rooms, crosstalk) is cut at its quietest point instead of being held until the speakers stop.
- **Context padding** — 0.15 s of audio on each side of a segment makes the embedding robust to background noise.
- **Auto-enrollment** — unidentified voices get an `Unknown_N` profile immediately so they accumulate samples. Rename/identify and every past `Unknown_N` segment re-labels.
- **Cluster mode** (`SPEAKER_ASSIGNMENT=cluster`, the default for files) — instead of matching each segment's embedding on its own, each pyannote speaker is matched once from the centroid of all its segments, so one person gets one label per file (one `Unknown_N` if not enrolled) and a single identify names them everywhere.
- **Misidentified-flag** — segments flagged `is_misidentified` are excluded from embedding averaging, keeping profiles clean.
- **Dual-detector emotion** — emotion2vec always runs; if the speaker has ≥ 3 voice samples for an emotion, a voice-profile detector also runs. Best match wins with deterministic tie-break rules.
- **Personalized learning** — user emotion corrections store both the 1024-D emotion embedding and the 512-D voice embedding, merged with weighted averaging into the speaker's profile.

## Emotion Recognition

### Categories (9)

emotion2vec+ large classifies each segment into one of these labels:

`angry` · `disgusted` · `fearful` · `happy` · `neutral` · `other` · `sad` · `surprised` · `unknown`

### Dual-detector decision

Every segment runs the emotion2vec base detector. If the speaker has **≥ 3 user-corrected samples** for any emotion, a second per-speaker voice-profile detector runs in parallel. The two scores are combined:

| Case                                        | Final emotion                     |
|---------------------------------------------|-----------------------------------|
| Only emotion2vec has a confident match      | emotion2vec winner                |
| emotion2vec returns `neutral` or `<unk>`    | `neutral`                         |
| Both detectors agree                        | agreed emotion (higher confidence)|
| Detectors disagree                          | `neutral` (conservative fallback) |
| Voice-profile detector wins (≥ threshold)   | profile winner                    |

The server returns the raw scores from both detectors so clients can override the fallback if they want.

### Personalization flow

1. User uploads audio → segment 7 is labeled `neutral` by emotion2vec with 0.94 confidence.
2. User disagrees, `POST /api/v1/conversations/{id}/segments/7/correct-emotion` with `{"emotion_category": "surprised"}`.
3. Server stores both the **1024-D emotion2vec embedding** and the **512-D pyannote voice embedding** into the speaker's `surprised` profile, averaged with any previous samples (EWMA-style).
4. Once that profile reaches 3 samples, the voice-profile detector activates for this speaker's future segments.
5. When a later segment has similar vocal characteristics, the voice-profile detector returns `surprised` above threshold and wins over the emotion2vec guess.

Corrections also set `emotion_corrected=true` on the segment, so they are kept out of personalization training loops (no feedback contamination).

### Example — dual-detector output

Segment from a laughing 10 s clip with no prior voice profile for this speaker:

```json
{
  "text": "Oh my god that is funny. Oh wow. Oh that's brilliant.",
  "emotion_category": "neutral",
  "emotion_confidence": 0.937,
  "detector_breakdown": {
    "emotion2vec_detector": { "emotion": "surprised", "confidence": 0.937 },
    "voice_profile_detector": null,
    "final_decision": {
      "emotion": "neutral",
      "confidence": 0.937,
      "reason": "Disagree: surprised vs neutral",
      "voice_profile_available": false
    }
  }
}
```

emotion2vec said *surprised* but the personalized detector is unavailable (no 3+ samples yet), so the server falls back to `neutral`. After three corrections to `surprised` for this speaker, the profile detector would agree and the final decision flips.

## REST API

Full interactive docs at `/docs` (Swagger UI). The OpenAPI JSON is at `/openapi.json`. Endpoint groups:

- `GET /api/v1/status` — health + GPU.
- `GET /api/v1/speakers`, `POST /api/v1/speakers/enroll`, `PATCH /api/v1/speakers/{id}/rename`, `DELETE /api/v1/speakers/{id}`, `DELETE /api/v1/speakers/unknown/all` — list, enroll, rename, delete.
- `GET/DELETE /api/v1/conversations/speakers/{id}/emotion-profiles`, `GET/PATCH .../emotion-threshold`, `PATCH .../emotion-profiles/{emotion}/{threshold,voice-threshold}` — emotion profiles and thresholds.
- `GET /api/v1/conversations`, `GET/PATCH/DELETE /api/v1/conversations/{id}` — list, read, rename, delete conversations.
- `POST /api/v1/conversations/{id}/reprocess`, `POST /api/v1/conversations/{id}/recalculate-emotions` — re-run pipelines.
- `POST /api/v1/conversations/{id}/segments/{sid}/{identify,correct-emotion}`, `PATCH .../{misidentified,emotion-misidentified}` — segment corrections.
- `GET /api/v1/conversations/segments/{sid}/audio` — extract segment audio.
- `POST /api/v1/process` — upload and process a whole file (form field `diarize=false` skips speaker diarization when one person is talking: faster, one speaker, still matched against the stored voices).
- `GET/POST /api/v1/settings/voice`, `POST /api/v1/settings/voice/reset` — runtime-tunable voice settings.
- `GET/POST /api/v1/profiles`, checkpoints, download, import, restore — backup/restore.
- `WS /api/v1/streaming/ws` — live audio ingestion: a `{"type": "start"}` message, then binary PCM chunks (48 kHz float32 by default; `"sample_rate": 16000, "format": "int16"` in the start message is a sixth of the bandwidth; `"diarize": false` skips diarization per chunk), then `{"type": "stop"}`. Speech is processed at each pause, and a turn that runs 15 s without one is cut at its quietest point.

### Example — upload + process

```bash
curl -X POST http://localhost:8418/api/v1/process \
  -F "audio_file=@meeting.wav"
```

Response (trimmed to one segment for readability):

```json
{
  "id": 42,
  "title": "Uploaded: meeting.wav",
  "duration": 137.39,
  "status": "completed",
  "num_segments": 24,
  "num_speakers": 2,
  "transcript_segments": [
    {
      "id": 911,
      "conversation_id": 42,
      "speaker_id": 17,
      "speaker_name": "Unknown_1776798145477718",
      "text": "Call me Ishmael. Some years ago, never mind how long precisely...",
      "start_offset": 0.05,
      "end_offset": 6.71,
      "confidence": 1.0,
      "emotion_category": "neutral",
      "emotion_confidence": 1.0,
      "emotion_corrected": false,
      "emotion_misidentified": false,
      "detector_breakdown": null,
      "words": [
        { "word": " Call",    "start": 0.05, "end": 0.57, "probability": 0.91 },
        { "word": " me",      "start": 0.57, "end": 0.75, "probability": 0.99 },
        { "word": " Ishmael.","start": 0.75, "end": 1.37, "probability": 0.99 }
      ]
    }
  ]
}
```

Unidentified voices are auto-enrolled as `Unknown_<ts>` so they accumulate samples. Rename retroactively:

```bash
# Identify an unknown speaker (retroactively labels every past segment with that voice)
curl -X POST http://localhost:8418/api/v1/conversations/42/segments/911/identify \
  -H 'Content-Type: application/json' \
  -d '{"speaker_name": "Alice", "enroll": true}'

# Correct an emotion (feeds personalization)
curl -X POST http://localhost:8418/api/v1/conversations/42/segments/911/correct-emotion \
  -H 'Content-Type: application/json' \
  -d '{"emotion_category": "surprised"}'
```

## MCP Server

JSON-RPC 2.0 over HTTP at `/mcp`. Compatible with Claude Desktop, Flowise, and any client that speaks the spec.

```json
{
  "mcpServers": {
    "speaker-diarization": {
      "url": "http://localhost:8418/mcp",
      "transport": "http"
    }
  }
}
```

Eleven tools: `list_conversations`, `get_conversation`, `get_latest_segments`, `list_speakers`, `rename_speaker`, `delete_speaker`, `delete_all_unknown_speakers`, `identify_speaker_in_segment`, `update_conversation_title`, `reprocess_conversation`, `search_conversations_by_speaker`. All mutations (identify / rename) retroactively update past segments so the AI doesn't have to reconcile identities manually.

## Data Persistence

```
speaker-diarization-server/
├── data/
│   ├── recordings/           # permanent audio (uploads + concatenated streams)
│   ├── stream_segments/      # per-conversation live-segment WAVs
│   └── temp/                 # transient
├── volumes/
│   ├── speakers.db           # SQLite (WAL)
│   └── huggingface_cache/    # pyannote + Whisper + emotion2vec
└── backups/
    ├── profile_<name>.json            # full speaker + segment + settings snapshot
    └── checkpoint_<name>_<ts>.json    # timestamped state capture
```

Docker bind-mounts `./volumes`, `./data`, `./backups`, and `./certs` (read-only) into the container. Nothing else is persisted between runs.

## Troubleshooting

**"HuggingFace token not found"** — confirm `HF_TOKEN` in `.env`, no quotes/spaces, model terms accepted.

**"Unable to load libcudnn_cnn.so.9"** — `run_local.sh` sets `LD_LIBRARY_PATH` automatically. Manual: `pip install nvidia-cudnn-cu12==9.* nvidia-cublas-cu12`.

**Docker permission errors on `data/`, `volumes/`, `backups/`** — the container runs as uid 1000. If your host uid differs, `sudo chown -R 1000:1000 data/ volumes/ backups/` once.

**Docker GPU not visible** — verify `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` works; if not, reinstall the NVIDIA Container Toolkit.

**Wrong GPU selected** — set `CUDA_VISIBLE_DEVICES` to match the `nvidia-smi` index you want; `CUDA_DEVICE_ORDER=PCI_BUS_ID` is already pinned for you.

**"CUDA out of memory"** — switch `WHISPER_MODEL` to a smaller variant (`small` or `base`), or lower `CLEANUP_VRAM_THRESHOLD_GB` so cleanup runs more often. With `ASR_BACKEND=parakeet`, speech-to-text alone peaks at ~5.5 GB on long recordings.

**`ASR_BACKEND=parakeet needs NVIDIA NeMo`** — the Docker image includes NeMo; outside Docker, `pip install 'nemo_toolkit[asr]==3.0.0'` (it is in `requirements.txt`). Parakeet v2 transcribes English only: for other languages use `PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3` (25 European languages) or `ASR_BACKEND=whisper`.

**Speakers named before upgrading are recognised less often** — profiles auto-enrolled by the old per-segment
matching are often built from a single short segment, which may not reach `CLUSTER_SPEAKER_THRESHOLD`. Identify the
speaker once in a recent conversation (the profile is then rebuilt from all of their segments), or set
`SPEAKER_ASSIGNMENT=segment` to keep the old behaviour.

**Speaker not recognized** — enroll with 10–30 s of clean audio. Lower `SPEAKER_THRESHOLD` to ~0.20 for challenging audio (movies, music). In cluster mode the threshold is `CLUSTER_SPEAKER_THRESHOLD` (0.70); speakers with only a few seconds of enrolled speech may stay `Unknown` until identified in more recordings.

**Browser microphone not working** — WebSocket mic capture requires HTTPS. Generate or drop a cert/key into `certs/`.

## License

MIT — see `LICENSE`. All bundled dependencies (pyannote.audio, faster-whisper, FastAPI, PyTorch, SQLAlchemy, Pydantic) are under MIT or BSD; NVIDIA NeMo is Apache-2.0. The pyannote **models** require HuggingFace account + token + terms acceptance; the software remains open-source. The Parakeet-TDT models (`ASR_BACKEND=parakeet`) are released by NVIDIA under CC-BY-4.0.

## Credits

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — Hervé Bredin et al.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — SYSTRAN
- [Whisper](https://github.com/openai/whisper) — OpenAI
- [NeMo](https://github.com/NVIDIA/NeMo) / [Parakeet-TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) — NVIDIA
- [emotion2vec](https://github.com/ddlBoJack/emotion2vec) / [FunASR](https://github.com/modelscope/FunASR) — ACL 2024
- [FastAPI](https://github.com/tiangolo/fastapi) — Sebastián Ramírez

## Planned

- AI-generated conversation summaries and titles on finalize.
- Vector-search over transcripts for semantic conversation lookup.

## Disclaimer

Provided as-is, no warranty. Speaker recognition is probabilistic — verify important identifications manually and obtain consent before recording others. Portions of this codebase were pair-programmed with an AI assistant; review before deploying in critical settings.
