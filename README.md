# telesearch

Local, private **AI search over a large Telegram conversation** — including the
photos and videos — powered entirely by **open-weight models** running on your
own GPU (built and tuned with an **NVIDIA RTX PRO 6000 / 96 GB** in mind).

You point it at a Telegram chat export. It reads the text, *describes the
photos*, *summarizes and transcribes the videos*, *transcribes voice messages*,
and puts everything into one searchable index. Then you can:

- **Search** semantically + by keyword: `telesearch search "the receipt from the sushi place"`
- **Ask** questions in natural language (RAG): `telesearch ask "what hotel did we book in Rome?"`

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
   search (RRF hybrid)  ──►  ask (RAG with local chat model)
```

Why this design:

- **Photos/videos become first-class search targets.** A vision-language model
  writes a detailed caption ("two people at a beach at sunset holding a
  cocktail menu") which is then embedded just like a typed message. So "that
  beach photo" finds the *image*, not just messages mentioning a beach.
- **Hybrid retrieval** (semantic embeddings + BM25 keyword) via Reciprocal Rank
  Fusion catches both fuzzy/conceptual queries *and* exact strings (names,
  numbers, links).
- **Embedded vector DB (LanceDB)** = no separate database server to run; the
  index is just files on disk, which scales fine to very large chats.
- **Open-weight, OpenAI-compatible serving.** The VLM/chat model is reached
  through a standard `/v1/chat/completions` endpoint, so you can serve any open
  model with vLLM/SGLang/Ollama and swap models freely.

---

## Recommended open-weight models for an RTX PRO 6000 (96 GB)

96 GB of VRAM is a lot — you can comfortably run a strong VLM *and* the
embedding/ASR models together, or a 70B-class chat model quantized.

| Job | Recommended | Notes |
|---|---|---|
| **Vision-language (image/video captioning)** | `Qwen/Qwen2.5-VL-32B-Instruct` | Excellent captioning + OCR; ~70 GB in bf16, or run AWQ/FP8 to leave room for other models. `Qwen2.5-VL-7B` is a great lighter option. |
| **Chat / RAG answers** | `Qwen/Qwen2.5-VL-32B-Instruct` (reuse) or `Qwen2.5-72B-Instruct` / `Llama-3.3-70B-Instruct` (AWQ 4-bit ≈ 40 GB) | The VLM can double as the chat model, so you only need one served endpoint. |
| **Text embeddings** | `BAAI/bge-m3` | Multilingual, strong retrieval, long context. Alternatives: `Qwen3-Embedding`, `intfloat/multilingual-e5-large`. ~2 GB. |
| **(Optional) image-text embeddings** | `jinaai/jina-clip-v2` or SigLIP | For pure visual similarity ("find similar-looking photos"). The default text pipeline already makes images searchable via captions. |
| **Speech-to-text** | Whisper `large-v3` via `faster-whisper` | Transcribes voice notes + video audio. ~3 GB in float16. |

Sizing tip: VLM-32B (FP8/AWQ) + bge-m3 + Whisper large-v3 all fit together in 96
GB with headroom. If you'd rather use a 72B/70B chat model *and* a separate VLM,
quantize both to 4-bit AWQ.

---

## Setup

### 1. Export your Telegram chat
In **Telegram Desktop**: ☰ menu → **Settings → Advanced → Export Telegram
data** → choose the chat, enable **Photos**, **Video files**, **Voice
messages**, and set format to **Machine-readable JSON**. You'll get a folder
with `result.json` plus media sub-folders.

### 2. Install
```bash
pip install -e ".[all]"      # core + asr + video + ui
# Also install the system ffmpeg binary for video frame extraction:
#   sudo apt-get install ffmpeg
```

### 3. Serve an open-weight VLM (OpenAI-compatible)
```bash
pip install vllm
vllm serve Qwen/Qwen2.5-VL-32B-Instruct --port 8000
```

### 4. Configure
```bash
cp .env.example .env
# edit .env: set TELESEARCH_LLM_BASE_URL / model names / device if needed
```

---

## Usage

```bash
# Build the index (captions images, summarizes+transcribes videos, transcribes voice)
telesearch index /path/to/telegram_export

# Skip heavy media steps for a quick text-only index
telesearch index /path/to/telegram_export --no-videos --no-audio

# Hybrid search
telesearch search "sunset photo from the beach trip"
telesearch search "invoice" --modality image      # only photos
telesearch search "address" --modality audio       # only voice transcripts

# Ask a question (RAG over the conversation, cites message ids)
telesearch ask "where did we say we'd meet on Saturday?"

# Show config + index status
telesearch info
```

### Web UI (optional)
```bash
export TELESEARCH_EXPORT_ROOT=/path/to/telegram_export   # to render thumbnails
streamlit run telesearch/ui/app.py
```

---

## Project layout

```
telesearch/
  config.py              # settings (env / .env)
  models.py              # Message + Chunk data classes
  ingest/telegram_parser.py   # result.json -> Messages
  media/captioner.py     # VLM image / video-frame captioning
  media/video.py         # frame extraction (PyAV/ffmpeg)
  media/asr.py           # Whisper transcription
  index/embeddings.py    # sentence-transformers text embeddings
  index/store.py         # LanceDB vectors + BM25 + RRF hybrid search
  index/build.py         # end-to-end indexing pipeline
  search/retriever.py    # hybrid retrieval
  search/rag.py          # question answering with a local chat model
  cli.py                 # `telesearch` command
  ui/app.py              # optional Streamlit UI
```

---

## Notes & trade-offs

- **First indexing run is the expensive part** (captioning + transcription).
  After that, search/ask are fast. Media steps are independently toggleable.
- The pipeline makes images searchable via **text captions** by default. For
  literal "find visually similar images" you can add CLIP/SigLIP embeddings
  (`TELESEARCH_IMAGE_EMBED_MODEL` is wired in config for that extension).
- Captioning is the quality bottleneck for visual search — a bigger VLM gives
  noticeably better, more specific captions, which your 96 GB GPU can afford.
- Privacy: all models are open-weight and run locally; the OpenAI client only
  talks to *your* local server.

## License
MIT
