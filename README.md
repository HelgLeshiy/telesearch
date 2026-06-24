# telesearch

Local, private **AI search over a large Telegram conversation** — including the
photos and videos — powered entirely by **open-weight models** running on your
own GPU (built and tuned with an **NVIDIA RTX PRO 6000 / 96 GB** in mind).

You point it at a Telegram chat export. It reads the text, *describes the
photos*, *summarizes and transcribes the videos*, *transcribes voice messages*,
*reads text out of images (OCR)*, and *extracts the contents of attached
documents* (PDF, Word, Excel, PowerPoint, text/code/CSV) — then puts everything
into one searchable index. Then you can:

- **Search** semantically + by keyword: `docker compose run --rm telesearch search "the receipt from the sushi place"`
- **Ask** questions in natural language (RAG): `docker compose run --rm telesearch ask "what hotel did we book in Rome?"`

Everything runs locally. Nothing leaves your machine.

---

## How it works (the approach)

The core idea: **turn every modality into searchable text, embed it all into one
vector space, and search with hybrid (semantic + keyword) retrieval — then
optionally let a local LLM answer questions over the retrieved context (RAG).**

```
Telegram export (result.json + media/)
          │
          ▼
   ┌──────────────┐
   │  ingest      │  parse messages, resolve media paths
   └──────┬───────┘
          ▼
   ┌──────────────────────────────────────────────┐
   │  media understanding (open-weight models)      │
   │   • photos  → VLM caption (Qwen2.5-VL)         │
   │   • videos  → sampled frames → VLM summary     │
   │               + audio → Whisper transcript     │
   │   • voice   → Whisper transcript               │
   │   • files   → document text extraction          │
   │               (PDF, Office, text/code/CSV)      │
   └──────┬─────────────────────────────────────────┘
          ▼
   ┌──────────────┐
   │  embed       │  text embeddings (bge-m3) → vectors
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │  LanceDB     │  vectors + BM25 full-text, on disk
   └──────┬───────┘
          ▼
   hybrid (RRF) ─► rerank (bge-reranker-v2-m3) ─► search results
                                               └─► ask (RAG, local chat model)
```

### Which model does what?

There is **no generative LLM doing the search itself**. Retrieval is handled by
embeddings + keyword matching + a reranker. Generative models are only used to
*understand media at index time* and to *write answers* in `ask`:

| Stage | Model | Type | When |
|---|---|---|---|
| **Search / retrieval** | `bge-m3` + BM25 | embedding + lexical | every query (fast) |
| **Rerank** | `bge-reranker-v2-m3` | cross-encoder | every query (top ~50 → top-k) |
| **Image / video captioning + OCR** | `Qwen2.5-VL` | vision-language (generative) | once, at index time |
| **Voice / video transcription** | Whisper `large-v3` | speech-to-text | once, at index time |
| **Document extraction** | pypdf / python-docx / openpyxl / python-pptx | parsers (no model) | once, at index time |
| **`ask` answer synthesis** | `Qwen2.5-VL` / `Qwen2.5-72B` | chat (generative) | only for `telesearch ask` |

Why this design:

- **Conversation context, not isolated messages.** Chat messages are short and
  fragmented ("ok", "the sushi place", "yes book it"), so embedding each one
  alone loses meaning and rarely matches a full question. Alongside per-message
  chunks, telesearch indexes overlapping **windows of consecutive messages**, so
  retrieval sees the surrounding conversation. Replies are also stitched to a
  snippet of the message they answer. At answer time, `ask` additionally pulls
  in the messages just **before/after** each hit and uses **HyDE** (drafting a
  hypothetical answer to retrieve with) so it still finds things even when your
  question shares almost no words with the message that answers it.
- **Photos/videos become first-class search targets.** A vision-language model
  writes a detailed caption ("two people at a beach at sunset holding a
  cocktail menu") which is then embedded just like a typed message. So "that
  beach photo" finds the *image*, not just messages mentioning a beach.
- **Hybrid retrieval** (semantic embeddings + BM25 keyword) via Reciprocal Rank
  Fusion catches both fuzzy/conceptual queries *and* exact strings (names,
  numbers, links), then a **cross-encoder reranker** re-scores the top
  candidates by reading query+document together for much sharper precision.
- **OCR as a first-class field.** On-image text (screenshots, receipts,
  documents, memes) is transcribed verbatim and indexed as its own chunk, so a
  query can match what's *written in* a picture independently of its caption.
- **Document attachments are searched by content.** PDFs, Word/Excel/PowerPoint
  files and any text-based file (code, CSV, JSON, logs, subtitles, ...) are
  extracted to text and split into overlapping chunks, so you can find a phrase
  *inside* a file someone shared. Genuinely binary files are skipped.
- **Embedded vector DB (LanceDB)** = no separate database server to run; the
  index is just files on disk, which scales fine to very large chats.
- **Open-weight, OpenAI-compatible serving.** The VLM/chat model runs in the
  `vllm` Compose service and is reached through a standard
  `/v1/chat/completions` endpoint — swap models via `.env` without changing
  application code.

---

## Recommended open-weight models for an RTX PRO 6000 (96 GB)

96 GB of VRAM is a lot — you can comfortably run a strong VLM *and* the
embedding/ASR models together, or a 70B-class chat model quantized.

| Job | Recommended | Notes |
|---|---|---|
| **Vision-language (image/video captioning)** | `Qwen/Qwen2.5-VL-32B-Instruct` | Excellent captioning + OCR; ~70 GB in bf16, or run AWQ/FP8 to leave room for other models. `Qwen2.5-VL-7B` is a great lighter option. |
| **Chat / RAG answers** | `Qwen/Qwen2.5-VL-32B-Instruct` (reuse) or `Qwen2.5-72B-Instruct` / `Llama-3.3-70B-Instruct` (AWQ 4-bit ≈ 40 GB) | The VLM can double as the chat model, so you only need one served endpoint. |
| **Text embeddings (search)** | `BAAI/bge-m3` | Multilingual, strong retrieval, long context. Alternatives: `Qwen3-Embedding`, `intfloat/multilingual-e5-large`. ~2 GB. |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Lightweight multilingual cross-encoder; big precision win on noisy chats. ~2 GB. |
| **(Optional) image-text embeddings** | `jinaai/jina-clip-v2` or SigLIP | For pure visual similarity ("find similar-looking photos"). The default text pipeline already makes images searchable via captions. |
| **Speech-to-text** | Whisper `large-v3` via `faster-whisper` | Transcribes voice notes + video audio. ~3 GB in float16. |

Sizing tip: VLM-32B (FP8/AWQ) + bge-m3 + Whisper large-v3 all fit together in 96
GB with headroom. If you'd rather use a 72B/70B chat model *and* a separate VLM,
quantize both to 4-bit AWQ.

---

## Setup

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose v2
- NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- A GPU available at **index 5** on the host (all services are pinned to that device — edit `device_ids` in `docker-compose.yml` to use a different GPU)
- A recent NVIDIA driver. The `telesearch` image ships PyTorch built against
  **CUDA 12.8**, required for Blackwell GPUs (e.g. RTX PRO 6000, `sm_120`); the
  driver must support CUDA 12.8 or later.

### 1. Export your Telegram chat

In **Telegram Desktop**: ☰ menu → **Settings → Advanced → Export Telegram
data** → choose the chat, enable **Photos**, **Video files**, **Voice
messages**, and set format to **Machine-readable JSON**. You'll get a folder
with `result.json` plus media sub-folders.

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
TELESEARCH_EXPORT_ROOT=/absolute/path/to/your/telegram_export
```

Optional: change `TELESEARCH_VLM_MODEL`, `TELESEARCH_CHAT_MODEL`, host ports,
or add `HUGGING_FACE_HUB_TOKEN` for gated models.

### 3. Build and start services

```bash
docker compose build

# VLM server (downloads the model on first run — can take a while)
docker compose up -d vllm

# Optional web UI at http://localhost:8501
docker compose up -d ui
```

The stack runs three services:

| Service | Role |
|---|---|
| **`vllm`** | OpenAI-compatible VLM/chat server (Qwen2.5-VL by default) |
| **`telesearch`** | CLI for indexing, search, and ask (invoked via `docker compose run`) |
| **`ui`** | Streamlit search UI |

All GPU workloads use **physical GPU 5 only** (`device_ids: ["5"]` in
`docker-compose.yml`). Inside the container that GPU is remapped to `cuda:0`, so
`TELESEARCH_DEVICE=cuda` is correct — do not set `CUDA_VISIBLE_DEVICES=5` in
the service environment.

Persistent data lives in Docker volumes: `telesearch-data` (LanceDB index) and
`huggingface-cache` (downloaded model weights).

#### vLLM GPU memory tuning

By default vLLM would try to reserve KV cache for the model's full 128k-token
context. For Qwen2.5-VL-32B that needs ~31 GiB on top of the ~62 GiB of weights,
which overflows the GPU and fails at startup with:

```
ValueError: To serve at least one request with the model's max seq len (128000),
(31.25 GiB KV cache is needed, which is larger than the available KV cache memory ...
```

telesearch only ever sends a few images plus a short prompt and caps output at
256 tokens, so the `vllm` service sets a smaller context window and disables
native-video profiling (video frames are sent as ordinary images). Tune these in
`.env` if needed:

```bash
TELESEARCH_VLLM_MAX_MODEL_LEN=32768            # raise if you have spare VRAM
TELESEARCH_VLLM_GPU_MEMORY_UTILIZATION=0.80    # fraction of GPU memory vLLM may use
TELESEARCH_VLLM_LIMIT_MM_PER_PROMPT={"image": 8, "video": 0}
```

#### Sharing one GPU between vLLM and telesearch

By default all three services are pinned to the **same** physical GPU. vLLM is
the memory hog, but the `telesearch`/`ui` containers also load their own GPU
models (bge-m3 embeddings, the reranker, Whisper). If vLLM is allowed to take
almost the whole card, those models can't allocate and indexing dies with:

```
OutOfMemoryError: CUDA out of memory ... GPU 0 has a total capacity of 94.97 GiB
of which 117.44 MiB is free ...
```

To avoid this, `TELESEARCH_VLLM_GPU_MEMORY_UTILIZATION` defaults to `0.80`
(~18 GiB left free on a 96 GiB card) and indexing embeds in small batches
(`TELESEARCH_EMBED_BATCH_SIZE=64`) with a capped input length
(`TELESEARCH_EMBED_MAX_SEQ_LENGTH=512`). The length cap matters most: bge-m3
otherwise allows 8192-token inputs, and a single long message in a batch can
spike a multi-GiB activation that OOMs the GPU. If you still hit OOM, lower
those values further; if vLLM runs on a *dedicated* GPU, raise utilization
toward `0.92`. For very large exports you can also give vLLM and telesearch
separate GPUs by editing the `device_ids` under each service in
`docker-compose.yml`.

---

## Usage

CLI commands run inside the `telesearch` container. Your export is mounted
read-only at `/export` (from `TELESEARCH_EXPORT_ROOT` in `.env`).

```bash
# Build the index (captions images, summarizes+transcribes videos, transcribes voice)
docker compose run --rm telesearch index /export

# Skip heavy media steps for a quick text-only index (documents need no GPU)
docker compose run --rm telesearch index /export --no-videos --no-audio --no-ocr

# Index ONLY typed text + documents (no VLM server needed)
docker compose run --rm --no-deps telesearch index /export --no-images --no-videos --no-audio --no-ocr

# Hybrid search (with cross-encoder rerank by default)
docker compose run --rm telesearch search "sunset photo from the beach trip"
docker compose run --rm telesearch search "invoice" --modality image      # only photo captions
docker compose run --rm telesearch search "total amount" --modality ocr    # only text read FROM images
docker compose run --rm telesearch search "quarterly budget" --modality document  # only file contents
docker compose run --rm telesearch search "address" --modality audio       # only voice transcripts
docker compose run --rm telesearch search "quick keyword lookup" --no-rerank   # skip reranking for speed

# Ask a question (RAG over the conversation, cites message ids)
docker compose run --rm telesearch ask "where did we say we'd meet on Saturday?"

# Show config + index status
docker compose run --rm telesearch info

# Add conversation-window + reply context to an EXISTING index, cheaply
# (refreshes only text/conversation chunks; no media re-processing, no vLLM)
docker compose run --rm --no-deps telesearch reindex-text /export
```

`--no-deps` skips starting the `vllm` service — use it for text-only indexing
when you don't need captioning or OCR.

### Indexing a large export (thousands of photos/videos)

A big chat export (e.g. ~3,500 photos + ~1,000 videos + ~500 files) means
thousands of VLM/Whisper calls. The build is designed for that:

- **Resumable.** Each block is persisted as it completes and already-indexed
  messages are skipped, so you can stop/restart freely. Just re-run the same
  command to continue; use `--rebuild` to start fresh.
- **Concurrent.** Remote VLM caption/OCR requests run across `--workers`
  threads (default 8) to keep the GPU server batching; local Whisper is
  serialized internally. Raise `--workers` if your vLLM server has spare
  capacity.
- **Robust.** Media that wasn't downloaded (Telegram's `(File not included…)`
  placeholder), stickers, and unreadable/binary files are skipped without
  failing the run.
- **MIME-aware.** Images/videos that were sent as plain *files* (so they live in
  the `files/` folder) are still captioned/transcribed by their MIME type, not
  treated as documents.

```bash
# Start (or resume) a full index; safe to Ctrl-C and re-run with the SAME flags
docker compose run --rm telesearch index /export --workers 12

# Force a clean rebuild
docker compose run --rm telesearch index /export --rebuild
```

Note: resume tracks progress per message, so re-run with the *same* flags to
continue. If you change which modalities to index, use `--rebuild` (changing
flags mid-way could otherwise skip messages whose media wasn't processed yet).

### Web UI

```bash
docker compose up -d ui
```

Open http://localhost:8501. Thumbnails require `TELESEARCH_EXPORT_ROOT` to be
set in `.env` (the export is mounted at `/export` inside the container).

### Development (without Docker)

For local hacking on the Python package:

```bash
pip install -e ".[all]" pytest   # also needs the ffmpeg system binary
cp .env.example .env
python3 -m telesearch.cli index /path/to/export
python3 -m pytest
```

Serve a VLM yourself (or point `TELESEARCH_LLM_BASE_URL` at an existing
OpenAI-compatible endpoint) when running outside Compose.

---

## Project layout

```
Dockerfile               # telesearch image (PyTorch CUDA + ffmpeg)
docker-compose.yml       # vllm + telesearch CLI + Streamlit UI (GPU 5)
.env.example             # configuration template

telesearch/
  config.py              # settings (env / .env)
  models.py              # Message + Chunk data classes
  ingest/telegram_parser.py   # result.json -> Messages
  media/captioner.py     # VLM image / video-frame captioning + OCR
  media/video.py         # frame extraction (PyAV/ffmpeg)
  media/asr.py           # Whisper transcription
  media/documents.py     # PDF/Office/text extraction + chunk splitter
  index/embeddings.py    # sentence-transformers text embeddings (bge-m3)
  index/store.py         # LanceDB vectors + BM25 + RRF hybrid search
  index/build.py         # end-to-end indexing pipeline
  search/retriever.py    # hybrid retrieval + rerank orchestration
  search/reranker.py     # bge-reranker-v2-m3 cross-encoder
  search/rag.py          # question answering with a local chat model
  cli.py                 # `telesearch` command
  ui/app.py              # optional Streamlit UI
```

---

## Notes & trade-offs

- **First indexing run is the expensive part** (captioning + transcription, and
  OCR adds a second VLM pass per photo). After that, search/ask are fast. Every
  media step (`--no-images/--no-videos/--no-audio/--no-ocr`) is toggleable.
- **Tuning conversation context.** Window size/stride
  (`TELESEARCH_CONVERSATION_WINDOW_SIZE` / `_STRIDE`) and the answer-time
  neighbour count (`TELESEARCH_CONTEXT_NEIGHBORS`) trade breadth of context for
  index size / prompt length. HyDE (`TELESEARCH_ENABLE_HYDE`) adds one extra LLM
  call per `ask` but markedly improves recall on conversational questions; turn
  it off (`--no-hyde`) for the fastest answers.
- **Adopting conversation windows on an existing index.** Window chunks and
  reply stitching are built at index time. To add them to an index you built
  before this feature existed, you have two options. A full `index --rebuild`
  re-does everything (including re-captioning every photo/video — slow, needs
  vLLM). If you only want the text-derived context, use **`reindex-text`**,
  which refreshes just the `text` + `conversation` chunks and leaves the media
  chunks alone — no VLM/Whisper, so it's fast and runs with `--no-deps`:

```bash
docker compose run --rm --no-deps telesearch reindex-text /export
```

  (A plain `index` re-run without `--rebuild` will *not* add windows to an
  already-indexed export: resume skips every seen message, so nothing new is
  produced. Use `reindex-text` or `--rebuild`.)
- The pipeline makes images searchable via **text captions** by default. For
  literal "find visually similar images" you can add CLIP/SigLIP embeddings
  (`TELESEARCH_IMAGE_EMBED_MODEL` is wired in config for that extension).
- Captioning is the quality bottleneck for visual search — a bigger VLM gives
  noticeably better, more specific captions, which your 96 GB GPU can afford.
- Privacy: all models are open-weight and run locally; the OpenAI client only
  talks to *your* local server.

## License
MIT
