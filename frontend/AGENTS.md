# frontend/ (static UI)

## OVERVIEW
Single `index.html` app for uploading docs and chatting against the FastAPI backend.

## ENTRY POINT
- `frontend/index.html`

## API INTEGRATION
- `API_BASE` hardcoded to `http://localhost:8001`.
- Upload:
  - `POST ${API_BASE}/upload` (multipart)
  - Polls `GET ${API_BASE}/status/{job_id}` every 2s
- Query:
  - `POST ${API_BASE}/query` with JSON `{ "q": "..." }`

## DEPLOYMENT
- Served by `nginx:alpine` in `docker-compose.yml`.
- Access via `http://localhost:8080`.

## NOTES
- CORS is permissive on the backend (`allow_origins=["*"]`), so this page can be served from anywhere during dev.
