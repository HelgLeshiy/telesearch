# AGENTS.md

## Cursor Cloud specific instructions

`telesearch` is a single Python package providing a CLI (`telesearch`) for local
multimodal AI search over a Telegram export. There is also an optional Streamlit
web UI (`telesearch/ui/app.py`). Standard install/usage commands live in
`README.md`; only the non-obvious, environment-specific notes are below.

### Environment / how to run
- Dependencies are installed into the **user site** (`~/.local`) by the startup
  update script (`pip install -e ".[all]" pytest ruff`). No virtualenv is used.
- `~/.local/bin` is **not on PATH**, so invoke entry points as modules instead of
  bare commands:
  - CLI: `python3 -m telesearch.cli <command>` (e.g. `... info`, `... index ...`).
  - Tests: `python3 -m pytest`.
  - Lint: `python3 -m ruff check .`.
- **The VM has no GPU.** The default `TELESEARCH_DEVICE` is `cuda`, which fails
  here. Set `TELESEARCH_DEVICE=cpu` before running anything that loads models
  (`index`, `search`, `ask`, and `info` with an existing index). The simplest
  way is a local `.env` (gitignored) containing `TELESEARCH_DEVICE=cpu`; also set
  `TELESEARCH_WHISPER_COMPUTE=int8` since `float16` is GPU-only.
- First model use downloads from Hugging Face: `bge-m3` (embeddings, ~2 GB) and
  `bge-reranker-v2-m3` (reranker, ~2 GB). Network access is required once; they
  are then cached under `~/.cache/huggingface`.

### What runs without extra services
- Text/document indexing and the retrieval+rerank model stack run fully on CPU.
  Index with media steps disabled (no GPU/VLM server needed):
  `python3 -m telesearch.cli index <export> --no-images --no-videos --no-audio --no-ocr`
- `ask` (RAG) and image/video captioning require an **OpenAI-compatible LLM/VLM
  server** at `TELESEARCH_LLM_BASE_URL` (default `http://localhost:8000/v1`,
  e.g. vLLM). That needs a GPU and is **not available** in this environment.

### Known issue: `search` / `ask` / re-index crash on an existing index
- `telesearch.index.store.VectorStore.__init__` checks
  `if TABLE_NAME in self.db.list_tables()`. With the installed LanceDB
  (>= 0.26, currently 0.33), `list_tables()` returns a `ListTablesResponse`
  object whose membership test never matches, so the code tries to re-create the
  existing table and raises `ValueError: Table 'chunks' already exists`.
- Effect: building a fresh index works, but any command that **re-opens** an
  existing index (`search`, `ask`, `info` with data present, re-running `index`)
  fails. No released LanceDB version makes the current `in` check work
  (versions <= 0.25.3 lack `list_tables()` entirely).
- This is a pre-existing application-code bug, not an environment problem. A
  one-line fix would be to use `self.db.table_names()` (which returns a plain
  `list[str]`) for the existence check. The full retrieval pipeline
  (bge-m3 hybrid + cross-encoder rerank) otherwise works on CPU — see
  `tests/test_store.py`, which exercises hybrid search end-to-end and passes.

### Lint
- The repo has **no configured linter**. `ruff check .` reports one pre-existing
  `F401` (unused `import json` in `telesearch/search/retriever.py`).
