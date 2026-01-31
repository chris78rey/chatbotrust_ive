# app/ (FastAPI backend)

## OVERVIEW
Single-file FastAPI service implementing document ingestion + embedding + Qdrant search + Ollama generation (sync + SSE streaming).

## ENTRY POINT
- `app/main.py`

## ENDPOINTS
- `POST /upload` (multipart): persists file to `UPLOAD_DIR`, enqueues `process_document`.
- `GET /status/{job_id}`: job status from SQLite (`jobs` table).
- `POST /query`: semantic search (Qdrant) + Ollama response (JSON).
- `POST /query-stream`: same as `/query`, but streams tokens via SSE (`text/event-stream`).

## DATA FLOW
- Startup hook:
  - Creates `UPLOAD_DIR`
  - Initializes SQLite schema (`jobs`, `queries`)
  - Creates Qdrant collection if missing
- Upload:
  - Saves `{job_id}_{filename}` into `UPLOAD_DIR`
  - Background task extracts text → chunks → embeddings → upserts into Qdrant
- Query:
  - Query embedding cached via `@lru_cache` (`get_query_embedding`)
  - Qdrant search → selects top contexts → builds prompt → calls Ollama `/api/generate`

## CONFIG (ENV VARS)
Defined in `app/main.py`:
- Qdrant: `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- Ollama: `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`
- Storage: `SQLITE_PATH`, `UPLOAD_DIR`
- Chunking: `CHUNK_SIZE`, `CHUNK_OVERLAP`
- Search/prompt: `SEARCH_TOP_K`, `SCORE_THRESHOLD`, `PROMPT_TOP_K`, `PROMPT_MAX_CHARS`
- Embeddings: `EMBED_MODEL_NAME`, `EMBED_DIM`

## PROMPTING
- The prompt enforces: Spanish output, detailed structured answer, and "context-only" responses.
- Defaults tuned for more complete answers:
  - `PROMPT_TOP_K=5`
  - `PROMPT_MAX_CHARS=6000`

## DOCKER
- Image built by `app/Dockerfile` (Python 3.11 slim)
- Container command overridden in `docker-compose.yml`:
  - `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
