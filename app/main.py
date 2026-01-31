import json
import os
import re
import sqlite3
import unicodedata
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import BackgroundTasks, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "BASE_CONOCIMIENTO")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))

SQLITE_PATH = os.getenv("SQLITE_PATH", "/data/app.db")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/uploads"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "10"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", str(max(20, TOP_K))))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.0"))
PROMPT_TOP_K = int(os.getenv("PROMPT_TOP_K", "5"))
PROMPT_MAX_CHARS = int(os.getenv("PROMPT_MAX_CHARS", "6000"))
REPLACE_FILENAME_ON_UPLOAD = os.getenv("REPLACE_FILENAME_ON_UPLOAD", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer(EMBED_MODEL_NAME)
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


class QueryRequest(BaseModel):
    q: str


def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = _db_connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            filename TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT,
            response_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def ensure_qdrant_collection() -> None:
    collections = qdrant.get_collections().collections
    if any(c.name == QDRANT_COLLECTION for c in collections):
        return

    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


@app.on_event("startup")
def _startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    ensure_qdrant_collection()


def _safe_filename(filename: str) -> str:
    return Path(filename).name


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_text_file(file_path: Path) -> str:
    return _normalize_text(_decode_bytes(file_path.read_bytes()))


def extract_text(file_path: Path, filename: str) -> str:
    from PyPDF2 import PdfReader
    import docx

    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(str(file_path))
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        return _normalize_text("\n".join(parts))

    if lower.endswith(".docx"):
        document = docx.Document(str(file_path))
        return _normalize_text("\n".join(p.text for p in document.paragraphs if p.text))

    return _read_text_file(file_path)


def chunk_text(text: str) -> list[str]:
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    overlap_paragraph = ""

    for paragraph in paragraphs:
        if len(paragraph) > CHUNK_SIZE:
            if current:
                chunks.append("\n\n".join(current))
                overlap_paragraph = current[-1] if CHUNK_OVERLAP > 0 else ""
                current = []
                current_len = 0

            step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
            for i in range(0, len(paragraph), step):
                part = paragraph[i : i + CHUNK_SIZE].strip()
                if part:
                    chunks.append(part)
            continue

        if not current and overlap_paragraph:
            current = [overlap_paragraph]
            current_len = len(overlap_paragraph)
            overlap_paragraph = ""

        separator_len = 2 if current else 0
        next_len = current_len + separator_len + len(paragraph)
        if next_len <= CHUNK_SIZE:
            current.append(paragraph)
            current_len = next_len
            continue

        if current:
            chunks.append("\n\n".join(current))
            overlap_paragraph = current[-1] if CHUNK_OVERLAP > 0 else ""

        current = [paragraph]
        current_len = len(paragraph)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def build_context_text(contexts: list[str], *, max_chars: int) -> str:
    if max_chars <= 0:
        return "\n\n".join(contexts)

    parts: list[str] = []
    total = 0
    for ctx in contexts:
        if not ctx:
            continue

        separator_len = 2 if parts else 0
        remaining = max_chars - total - separator_len
        if remaining <= 0:
            break

        if len(ctx) <= remaining:
            parts.append(ctx)
            total += separator_len + len(ctx)
            continue

        trimmed = ctx[:remaining].rstrip()
        if trimmed:
            parts.append(trimmed)
            total += separator_len + len(trimmed)
        break

    return "\n\n".join(parts)


def pick_prompt_contexts(query: str, results: list[Any], *, max_items: int) -> list[str]:
    max_items = max(1, max_items)
    query_lower = query.lower()
    query_tokens = {t for t in re.findall(r"\w+", query_lower) if len(t) >= 4}

    ordinal_tokens = {"primera", "segunda", "tercera", "cuarta", "quinta"}
    ordinals_in_query = query_tokens.intersection(ordinal_tokens)

    scored: list[tuple[float, str]] = []
    for hit in results:
        payload = getattr(hit, "payload", None) or {}
        text = payload.get("text", "")
        if not text:
            continue

        text_lower = text.lower()

        token_bonus = sum(1 for tok in query_tokens if tok in text_lower) * 0.02
        ordinal_bonus = 0.5 if any(tok in text_lower for tok in ordinals_in_query) else 0.0
        propuesta_bonus = 0.1 if "propuesta" in query_lower and "propuesta" in text_lower else 0.0

        score = float(getattr(hit, "score", 0.0)) + token_bonus + ordinal_bonus + propuesta_bonus
        scored.append((score, text))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in scored[:max_items]]


@lru_cache(maxsize=1000)
def get_query_embedding(query_text: str) -> list[float]:
    return model.encode(query_text).tolist()


def process_document(job_id: str, file_path: str, filename: str) -> None:
    conn = _db_connect()
    try:
        conn.execute("UPDATE jobs SET status=? WHERE id=?", ("processing", job_id))
        conn.commit()

        text = extract_text(Path(file_path), filename)
        chunks = chunk_text(text)

        if not chunks:
            conn.execute(
                "UPDATE jobs SET status=?, error=? WHERE id=?",
                ("error", "No se pudo extraer texto del documento", job_id),
            )
            conn.commit()
            return

        embeddings = model.encode(chunks)

        if REPLACE_FILENAME_ON_UPLOAD:
            qdrant.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="filename",
                                match=MatchValue(value=filename),
                            )
                        ]
                    )
                ),
            )

        points: list[PointStruct] = []
        job_uuid = uuid.UUID(job_id)
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = uuid.uuid5(job_uuid, str(idx))
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk,
                        "filename": filename,
                        "chunk_index": idx,
                        "job_id": job_id,
                    },
                )
            )

        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

        conn.execute("UPDATE jobs SET status=? WHERE id=?", ("completed", job_id))
        conn.commit()

    except Exception as exc:
        conn.execute(
            "UPDATE jobs SET status=?, error=? WHERE id=?",
            ("error", str(exc), job_id),
        )
        conn.commit()
    finally:
        conn.close()


@app.post("/upload")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks) -> dict[str, str]:
    job_id = str(uuid.uuid4())
    filename = _safe_filename(file.filename or "document")

    file_path = UPLOAD_DIR / f"{job_id}_{filename}"
    content = await file.read()
    file_path.write_bytes(content)

    conn = _db_connect()
    conn.execute(
        "INSERT INTO jobs (id, status, filename) VALUES (?, ?, ?)",
        (job_id, "processing", filename),
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(process_document, job_id, str(file_path), filename)

    return {"job_id": job_id, "status": "processing"}


@app.get("/status/{job_id}")
async def get_status(job_id: str) -> dict[str, str]:
    conn = _db_connect()
    cursor = conn.execute("SELECT status, error FROM jobs WHERE id=?", (job_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"status": "not_found"}

    status, error = row
    if status == "error" and error:
        return {"status": status, "error": error}

    return {"status": status}


@app.post("/query")
async def query(req: QueryRequest) -> dict[str, Any]:
    query_vector = get_query_embedding(req.q)

    results = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=SEARCH_TOP_K,
        score_threshold=SCORE_THRESHOLD,
    )

    contexts = pick_prompt_contexts(req.q, results, max_items=PROMPT_TOP_K)
    if not contexts:
        answer = "No encuentro información sobre eso en la base de conocimiento"
    else:
        context_text = build_context_text(contexts, max_chars=PROMPT_MAX_CHARS)

        prompt = (
            "Contexto de la base de conocimiento:\n"
            f"{context_text}\n\n"
            f"Pregunta del usuario: {req.q}\n\n"
            "Instrucciones: Responde en español y de forma completa, basándote ÚNICAMENTE en el contexto proporcionado. "
            "No inventes información ni completes con conocimiento externo. "
            "Si faltan datos para responder bien, di explícitamente qué parte no está en el contexto y luego responde solo con lo que sí está. "
            "Estructura la respuesta así:\n"
            "1) Resumen (2-4 líneas)\n"
            "2) Detalles (5-10 viñetas con hechos concretos)\n"
            "3) Evidencia (2-3 citas textuales cortas del contexto, entre comillas)\n\n"
            "Si no hay información suficiente, di \"No encuentro información sobre eso en la base de conocimiento\".\n\n"
            "Respuesta:"
        )

        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,
                        "temperature": 0.3,
                        "num_predict": 512,
                        "top_k": 10,
                        "top_p": 0.9,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()

        answer = result.get("response", "")

    conn = _db_connect()
    conn.execute(
        "INSERT INTO queries (query_text, response_text) VALUES (?, ?)",
        (req.q, answer),
    )
    conn.commit()
    conn.close()

    return {"query": req.q, "response": answer, "contexts": contexts}


@app.post("/query-stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    query_vector = get_query_embedding(req.q)

    results = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=SEARCH_TOP_K,
        score_threshold=SCORE_THRESHOLD,
    )

    contexts = pick_prompt_contexts(req.q, results, max_items=PROMPT_TOP_K)
    context_text = build_context_text(contexts, max_chars=PROMPT_MAX_CHARS)

    if not contexts:
        async def generate() -> AsyncIterator[str]:
            yield "data: No encuentro información sobre eso en la base de conocimiento\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    prompt = (
        "Contexto de la base de conocimiento:\n"
        f"{context_text}\n\n"
        f"Pregunta del usuario: {req.q}\n\n"
        "Instrucciones: Responde en español y de forma completa, basándote ÚNICAMENTE en el contexto proporcionado. "
        "No inventes información ni completes con conocimiento externo. "
        "Si faltan datos para responder bien, di explícitamente qué parte no está en el contexto y luego responde solo con lo que sí está. "
        "Estructura la respuesta así:\n"
        "1) Resumen (2-4 líneas)\n"
        "2) Detalles (5-10 viñetas con hechos concretos)\n"
        "3) Evidencia (2-3 citas textuales cortas del contexto, entre comillas)\n\n"
        "Si no hay información suficiente, di \"No encuentro información sobre eso en la base de conocimiento\".\n\n"
        "Respuesta:"
    )

    async def generate() -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_ctx": 2048,
                        "temperature": 0.3,
                        "num_predict": 512,
                        "top_k": 10,
                        "top_p": 0.9,
                    },
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("response")
                    if chunk:
                        yield f"data: {chunk}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
