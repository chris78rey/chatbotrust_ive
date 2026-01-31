# PROJECT GUIDE (AGENTS)

## OVERVIEW
Minimal RAG chatbot stack: FastAPI API + Qdrant vectors + Ollama LLM + static HTML frontend (nginx), orchestrated via Docker Compose.

## STRUCTURE
```
./
├── docker-compose.yml        # Local/dev stack (ports + env)
├── inst.md                   # Detailed implementation notes (Spanish)
├── app/                      # FastAPI backend
│   ├── main.py               # API endpoints + RAG pipeline
│   ├── requirements.txt      # Python deps
│   └── Dockerfile            # Backend image
├── frontend/                 # Static UI served by nginx
│   └── index.html
├── data/                     # Bind mount for SQLite file (runtime)
└── uploads/                  # Bind mount for uploaded docs (runtime)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Run full stack | `docker-compose.yml` | Defines ports + env vars for API/Qdrant/Ollama/frontend |
| Backend API | `app/main.py` | `/upload`, `/status/{job_id}`, `/query`, `/query-stream` |
| Frontend UI | `frontend/index.html` | Hits API at `http://localhost:8001` |
| Architecture notes | `inst.md` | Reference spec + rationale |

## RUNTIME PORTS (docker-compose)
- Frontend (nginx): `8080` (maps container `80`)
- API (FastAPI): `8001` (maps container `8000`)
- Qdrant: `6335` (maps container `6333`)
- Ollama: `11435` (maps container `11434`)

## COMMANDS
```bash
# Start everything
docker compose up --build

# Logs
docker compose logs -f api
docker compose logs -f qdrant
docker compose logs -f ollama

# Basic checks
curl http://localhost:8001/status/some-job-id
curl http://localhost:11435/api/tags
```

## NOTES / GOTCHAS
- `data/` and `uploads/` are runtime mounts; do not treat them as source.
- LSP diagnostics are not available by default in this environment:
  - Python: `basedpyright` not installed
  - Web/JS/HTML: `biome` not installed
