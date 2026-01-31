# app/ (FastAPI backend)

## OVERVIEW
Single-file FastAPI service implementing document ingestion + embedding + Qdrant search + Ollama generation (sync + SSE streaming).

## ENTRY POINT
- `app/main.py`

## ENDPOINTS
- `POST /upload` (multipart): persists file to `UPLOAD_DIR`, enqueues `process_document`.
- `GET /status/{job_id}`: job status from SQLite (`jobs` table).
- `POST /query`: semantic search (Qdrant) + Ollama response.
- `POST /query-stream`: same as `/query`, but streams tokens via SSE (`text/event-stream`).

## DATA FLOW
- Startup hook:
  - Creates `UPLOAD_DIR`
  - Initializes SQLite schema (`jobs`, `queries`)
  - Creates Qdrant collection if missing
- Upload:
  - Saves `{job_id}_{filename}`
  - Writes job row with status `processing`
  - Background task extracts text → chunks → embeddings → upserts into Qdrant
- Query:
  - Embedding of query cached via `@lru_cache` (`get_query_embedding`)
  - Qdrant search → prompt assembly → `POST /api/generate` to Ollama

## CONFIG (ENV VARS)
Defined in `app/main.py`:
- Qdrant: `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- Ollama: `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL`
- Storage: `SQLITE_PATH`, `UPLOAD_DIR`
- Chunking/search: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`
- Embeddings: `EMBED_MODEL_NAME`, `EMBED_DIM`

## STORAGE
- SQLite file at `SQLITE_PATH` (defaults `/data/app.db`)
  - WAL mode enabled in `_db_connect()`
- Uploaded files in `UPLOAD_DIR` (defaults `/uploads`)

## DOCKER
- Image built by `app/Dockerfile` (Python 3.11 slim)
- Container command overridden in `docker-compose.yml`:
  - `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
