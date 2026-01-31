# frontend/ (static UI)

## OVERVIEW
Single `index.html` app for uploading docs and chatting against the FastAPI backend.

## ENTRY POINT
- `frontend/index.html`

## API INTEGRATION
- `API_BASE` is hardcoded to `http://localhost:8001`.
- Upload:
  - `POST ${API_BASE}/upload` (multipart)
  - Polls `GET ${API_BASE}/status/{job_id}` every 2s
- Query:
  - `POST ${API_BASE}/query` with JSON `{ "q": "..." }`

## DEPLOYMENT
- Served by `nginx:alpine` in `docker-compose.yml`.
- Access via `http://localhost:8080`.

## ANTI-PATTERNS
- Do not point the UI at `:8000` (container port); use `:8001` (host port).

## NOTES
- Backend CORS is permissive (`allow_origins=["*"]`) for local dev.
