# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-31
**Commit:** 1a406ce
**Branch:** main

## OVERVIEW
Minimal RAG chatbot stack: FastAPI API + Qdrant vector search + Ollama LLM + static HTML frontend (nginx), orchestrated via Docker Compose.

## STRUCTURE
```
./
├── docker-compose.yml        # Local/dev stack (ports + env)
├── inst.md                   # Implementation notes (Spanish)
├── app/                      # FastAPI backend (RAG pipeline)
├── frontend/                 # Static UI (nginx)
├── data/                     # Bind mount for SQLite (runtime)
└── uploads/                  # Bind mount for uploaded docs (runtime)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Run full stack | `docker-compose.yml` | Ports + env vars for all services |
| Backend RAG logic | `app/main.py` | `/upload`, `/status/{job_id}`, `/query`, `/query-stream` |
| Frontend UI | `frontend/index.html` | Calls API at `http://localhost:8001` |
| Spec / rationale | `inst.md` | Kept in sync with current repo |

## RUNTIME PORTS (HOST)
- Frontend (nginx): `http://localhost:8080`
- API (FastAPI): `http://localhost:8001`
- Qdrant: `http://localhost:6335`
- Ollama: `http://localhost:11435`

## COMMANDS
```bash
# Start everything
docker compose up --build -d

# Logs
docker compose logs -f api

auth="Content-Type: application/json"
# Check Ollama model
curl http://localhost:11435/api/tags

# API docs
curl -I http://localhost:8001/docs
```

## ANTI-PATTERNS (THIS PROJECT)
- Do not commit `data/` or `uploads/` (runtime mounts).
- Do not treat `data/` or `uploads/` as source code.

## NOTES
- Compose `depends_on` does not guarantee readiness; if startup races happen, restart `api`.
- LSP diagnostics may be unavailable in this environment.
