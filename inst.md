# SISTEMA RAG EMPRESARIAL - VERSIÓN VIABLE SIMPLIFICADA

## OBJETIVO PRINCIPAL
Construir un chatbot empresarial que permita ingestión de documentos (TXT, PDF, DOCX) con búsqueda semántica mediante embeddings, optimizado para múltiples usuarios concurrentes, ejecución en CPU (sin GPU), y despliegue simple con Docker.

---

## CONFIGURACIÓN DE OLLAMA

### Instalación y Setup

**El Docker Compose ya incluye Ollama**, pero si quieres probarlo localmente primero:

```bash
# En Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# En Windows (descargar instalador)
# https://ollama.com/download/windows

# Iniciar servidor
ollama serve

# Descargar modelo (en otra terminal)
ollama pull llama3.2:1b
```

### Modelos Recomendados (por tamaño/performance)

| Modelo | Tamaño | RAM Necesaria | Velocidad CPU | Calidad |
|--------|--------|---------------|---------------|---------|
| `llama3.2:1b` | 1.3GB | 4GB | 20-40 tok/s | Básica ✅ |
| `llama3.2:3b` | 2GB | 6GB | 10-20 tok/s | Buena |
| `phi3:mini` | 2.3GB | 6GB | 15-25 tok/s | Muy buena |
| `mistral:7b` | 4.1GB | 10GB | 5-10 tok/s | Excelente |

**Para este proyecto: `llama3.2:1b` es perfecto** - balance ideal entre tamaño, velocidad y calidad.

### API de Ollama

Ollama expone una API compatible con OpenAI en el puerto del contenedor `11434`.

**En este proyecto (Docker Compose):**
- Desde tu máquina (host): `http://localhost:11435` (porque se mapea `11435:11434`)
- Entre contenedores: `http://ollama:11434`

**Endpoints principales:**

```bash
# Generar respuesta (sin streaming)
POST http://localhost:11435/api/generate
{
  "model": "llama3.2:1b",
  "prompt": "¿Qué es Python?",
  "stream": false
}

# Generar respuesta (con streaming)
POST http://localhost:11435/api/generate
{
  "model": "llama3.2:1b",
  "prompt": "¿Qué es Python?",
  "stream": true
}

# Chat (mantiene contexto)
POST http://localhost:11435/api/chat
{
  "model": "llama3.2:1b",
  "messages": [
    {"role": "user", "content": "Hola"}
  ]
}

# Listar modelos instalados
GET http://localhost:11435/api/tags
```

### Streaming en FastAPI (Avanzado)

Si quieres respuestas token por token (mejor UX):

```python
from fastapi.responses import StreamingResponse
import httpx
import json

@app.post("/query-stream")
async def query_stream(q: str):
    # ... (búsqueda en Qdrant igual que antes)
    
    async def generate():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                "http://ollama:11434/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if 'response' in data:
                            yield f"data: {data['response']}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Optimizaciones de Performance

**1. Configurar contexto y temperatura:**
```python
# En la llamada a Ollama
{
  "model": "llama3.2:1b",
  "prompt": prompt,
  "options": {
    "num_ctx": 2048,        # Contexto más corto = más rápido
    "temperature": 0.3,     # Menos creativo = más consistente
    "num_predict": 512,     # Máximo tokens en respuesta
    "top_k": 10,
    "top_p": 0.9
  }
}
```

**2. Caché de embeddings frecuentes:**
```python
# En main.py, agregar cache simple
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_query_embedding(query_text: str):
    return model.encode(query_text).tolist()
```

**3. Limitar chunks enviados a Ollama:**
```python
# Solo top-3 chunks en lugar de 5
results = qdrant.search(
    collection_name="BASE_CONOCIMIENTO",
    query_vector=query_vector,
    limit=3,  # Menos contexto = respuesta más rápida
    score_threshold=0.5  # Solo chunks relevantes
)
```

### Verificar que Ollama funciona

```bash
# Dentro del contenedor
docker-compose exec ollama ollama list

# Probar generación
docker-compose exec ollama ollama run llama3.2:1b "Hola, ¿cómo estás?"

# Ver logs
docker-compose logs -f ollama

# Probar API directamente
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:1b",
  "prompt": "Di hola en español",
  "stream": false
}'
```

---

## RECURSOS NECESARIOS CON OLLAMA

### Mínimos (con llama3.2:1b)
- **CPU**: 4 cores
- **RAM**: 10GB 
  - 2GB Qdrant
  - 3GB Ollama + modelo
  - 2GB API + sentence-transformers
  - 3GB OS
- **Disco**: 25GB 
  - 5GB modelos embeddings
  - 1.3GB llama3.2:1b
  - 10GB datos
  - 8GB sistema

### Recomendados para producción
- **CPU**: 8 cores (4 para Ollama)
- **RAM**: 16GB
- **Disco**: 50GB SSD

---

## STACK TECNOLÓGICO SIMPLIFICADO

### Backend
- **API única**: FastAPI (Python) - todo en un solo servicio
- **Procesamiento asíncrono**: asyncio + background tasks (sin Celery ni Redis)
- **Embeddings**: sentence-transformers con ONNX (CPU optimizado)

### Infraestructura
- **Base de datos vectorial**: Qdrant (colección única: BASE_CONOCIMIENTO)
- **Base de datos relacional**: SQLite (desarrollo) → PostgreSQL (producción)
- **Orquestación**: Docker Compose (2-3 contenedores)

### Frontend
- **Framework**: HTML + JavaScript vanilla con fetch API
- **Comunicación**: Server-Sent Events (SSE) para streaming de respuestas
- **UI**: CSS básico o framework CDN (Bootstrap/Tailwind via CDN)

### LLM Local
- **Ollama**: Servidor de modelos LLM local optimizado para CPU
- **Modelo**: llama3.2:1b (1B parámetros, ~1.3GB cuantizado)
- **API**: Compatible con OpenAI (fácil integración)

---

## ARQUITECTURA SIMPLIFICADA

### Flujo de Ingestión de Documentos
1. Usuario sube archivo → FastAPI endpoint `/upload`
2. Validación inmediata (tipo, tamaño) → respuesta HTTP
3. **Background task** procesa:
   - Extracción de texto (PyPDF2, python-docx, o texto plano)
   - Chunking simple (512 caracteres con overlap de 50)
   - Generación de embeddings (sentence-transformers)
   - Inserción en Qdrant
4. Estado consultable en endpoint `/status/{job_id}`

### Flujo de Consulta (RAG con Ollama)
1. Usuario envía pregunta → `/query` endpoint
2. Generar embedding de la query
3. Búsqueda vectorial en Qdrant (top-k=5)
4. Construir prompt con contexto: chunks + pregunta
5. Enviar a Ollama vía API local
6. Streaming respuesta vía SSE (token por token)
7. Log básico en SQLite

---

## ESPECIFICACIONES TÉCNICAS MÍNIMAS

### 1. API Backend (FastAPI - Python)

**Archivo: `app/main.py`**

```python
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import sqlite3
import uuid

app = FastAPI()

# Inicializar modelo de embeddings (se carga una vez al inicio)
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(host="qdrant", port=6333)

# Endpoints principales
@app.post("/upload")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    # Guardar archivo temporalmente
    file_path = f"/tmp/{job_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Procesar en background
    background_tasks.add_task(process_document, job_id, file_path, file.filename)
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    # Consultar estado en SQLite
    conn = sqlite3.connect('/data/app.db')
    cursor = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    return {"status": row[0] if row else "not_found"}

@app.post("/query")
async def query(q: str):
    import httpx
    
    # Generar embedding
    query_vector = model.encode(q).tolist()
    
    # Buscar en Qdrant
    results = qdrant.search(
        collection_name="BASE_CONOCIMIENTO",
        query_vector=query_vector,
        limit=5
    )
    
    # Extraer textos
    contexts = [hit.payload['text'] for hit in results]
    
    # Construir prompt para Ollama
    context_text = "\n\n".join(contexts)
    prompt = f"""Contexto de la base de conocimiento:
{context_text}

Pregunta del usuario: {q}

Instrucciones: Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado. Si no hay información suficiente, di "No encuentro información sobre eso en la base de conocimiento".

Respuesta:"""
    
    # Llamar a Ollama
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://ollama:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False
            }
        )
        result = response.json()
    
    # Guardar en BD
    conn = sqlite3.connect('/data/app.db')
    conn.execute(
        "INSERT INTO queries (query_text, response_text) VALUES (?, ?)",
        (q, result['response'])
    )
    conn.commit()
    conn.close()
    
    return {
        "query": q,
        "response": result['response'],
        "contexts": contexts
    }
```

**Dependencias (`requirements.txt`):**
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
sentence-transformers==2.3.1
qdrant-client==1.7.3
PyPDF2==3.0.1
python-docx==1.1.0
onnxruntime==1.16.3
httpx==0.26.0
```

**Función de procesamiento:**
```python
def process_document(job_id: str, file_path: str, filename: str):
    import sqlite3
    from PyPDF2 import PdfReader
    import docx
    
    # Actualizar estado
    conn = sqlite3.connect('/data/app.db')
    conn.execute("INSERT INTO jobs VALUES (?, 'processing')", (job_id,))
    conn.commit()
    
    try:
        # Extraer texto según tipo
        if filename.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = " ".join([page.extract_text() for page in reader.pages])
        elif filename.endswith('.docx'):
            doc = docx.Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])
        else:  # TXT
            with open(file_path) as f:
                text = f.read()
        
        # Chunking simple
        chunk_size = 512
        overlap = 50
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i+chunk_size])
        
        # Generar embeddings
        embeddings = model.encode(chunks)
        
        # Insertar en Qdrant
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append({
                "id": f"{job_id}_{idx}",
                "vector": embedding.tolist(),
                "payload": {
                    "text": chunk,
                    "filename": filename,
                    "chunk_index": idx
                }
            })
        
        qdrant.upsert(
            collection_name="BASE_CONOCIMIENTO",
            points=points
        )
        
        # Actualizar estado
        conn.execute("UPDATE jobs SET status='completed' WHERE id=?", (job_id,))
        conn.commit()
        
    except Exception as e:
        conn.execute("UPDATE jobs SET status='error' WHERE id=?", (job_id,))
        conn.commit()
    finally:
        conn.close()
```

### 2. Qdrant (Vector Database)

**Inicialización de colección:**
```python
# app/init_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="qdrant", port=6333)

client.create_collection(
    collection_name="BASE_CONOCIMIENTO",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
```

### 3. SQLite (Metadata Básica)

**Schema (`app/init_db.py`):**
```python
import sqlite3

conn = sqlite3.connect('/data/app.db')
conn.execute('''
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.execute('''
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_text TEXT,
        response_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
conn.close()
```

---

## DOCKER COMPOSE CON OLLAMA

**`docker-compose.yml` (estado actual del repo):**
```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11435:11434"
    volumes:
      - ollama_models:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    restart: unless-stopped
    entrypoint: ["/bin/sh", "-c"]
    command: >
      "ollama serve &
       sleep 5 &&
       ollama pull llama3.2:1b &&
       wait"

  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6335:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    depends_on:
      - ollama

  api:
    build: ./app
    ports:
      - "8001:8000"
    volumes:
      - ./data:/data
      - ./uploads:/uploads
    depends_on:
      - qdrant
      - ollama
    environment:
      QDRANT_HOST: qdrant
      OLLAMA_HOST: ollama
      SQLITE_PATH: /data/app.db
      UPLOAD_DIR: /uploads
      QDRANT_COLLECTION: BASE_CONOCIMIENTO
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./frontend:/usr/share/nginx/html

volumes:
  qdrant_data:
  ollama_models:
```

**`app/Dockerfile` (estado actual del repo):**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /data /uploads

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Nota:** la inicialización de SQLite y la creación de la colección en Qdrant se hacen en `app/main.py` durante el evento de startup (no hay `init_db.py` separado).

---

## FRONTEND MINIMALISTA

**`frontend/index.html`:**
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>RAG Chatbot</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; }
        #chat { border: 1px solid #ccc; height: 400px; overflow-y: auto; padding: 10px; }
        .message { margin: 10px 0; }
        .user { color: blue; }
        .assistant { color: green; }
        input, button { padding: 10px; margin: 5px; }
    </style>
</head>
<body>
    <h1>Chatbot RAG Empresarial</h1>

    <div>
        <h3>Subir Documento</h3>
        <input type="file" id="fileInput" accept=".txt,.pdf,.docx">
        <button onclick="uploadFile()">Subir</button>
        <div id="uploadStatus"></div>
    </div>

    <div>
        <h3>Chat</h3>
        <div id="chat"></div>
        <input type="text" id="queryInput" placeholder="Haz tu pregunta...">
        <button onclick="sendQuery()">Enviar</button>
    </div>

    <script>
        const API_BASE = 'http://localhost:8001';

        async function uploadFile() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            document.getElementById('uploadStatus').innerText = `Procesando... ID: ${data.job_id}`;

            const interval = setInterval(async () => {
                const statusRes = await fetch(`${API_BASE}/status/${data.job_id}`);
                const status = await statusRes.json();

                if (status.status === 'completed') {
                    document.getElementById('uploadStatus').innerText = 'Completado!';
                    clearInterval(interval);
                } else if (status.status === 'error') {
                    document.getElementById('uploadStatus').innerText = 'Error al procesar';
                    clearInterval(interval);
                }
            }, 2000);
        }

        async function sendQuery() {
            const query = document.getElementById('queryInput').value;
            if (!query) return;

            const chatDiv = document.getElementById('chat');
            chatDiv.innerHTML += `<div class="message user">Usuario: ${escapeHtml(query)}</div>`;

            const response = await fetch(`${API_BASE}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ q: query })
            });

            const data = await response.json();

            chatDiv.innerHTML += `<div class="message assistant">Asistente: ${escapeHtml(data.response ?? '')}</div>`;

            if (data.contexts && data.contexts.length > 0) {
                chatDiv.innerHTML += `<div class="message sources" style="color: gray; font-size: 0.9em;">Fuentes: ${data.contexts.length} chunks encontrados</div>`;
            }

            document.getElementById('queryInput').value = '';
            chatDiv.scrollTop = chatDiv.scrollHeight;
        }

        function escapeHtml(s) {
            return String(s)
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#039;');
        }
    </script>
</body>
</html>
```

---

## PLAN DE IMPLEMENTACIÓN ULTRA SIMPLIFICADO

###  1: Core Funcional
**Día 1-2: Infraestructura**
- Setup Docker Compose (Qdrant + API)
- Endpoint `/upload` que acepta TXT
- Guardar archivo y devolver job_id

**Día 3-4: Procesamiento**
- Background task que chunka texto
- Generar embeddings con sentence-transformers
- Insertar en Qdrant

**Día 5-7: Búsqueda**
- Endpoint `/query` 
- Búsqueda vectorial en Qdrant
- Devolver chunks relevantes (sin LLM)

###  2: Mejoras
**Día 8-10: Formatos adicionales**
- Soporte PDF (PyPDF2)
- Soporte DOCX (python-docx)

**Día 11-12: Frontend básico**
- HTML con formulario de carga
- Input de chat
- Mostrar resultados

**Día 13-14: Integración Ollama**
- Agregar servicio Ollama a docker-compose
- Endpoint `/query` con generación de respuestas
- Streaming SSE (Server-Sent Events)
- Optimizar prompts para RAG

###  3: Producción
**Día 15-16: Persistencia**
- Migrar de SQLite a PostgreSQL
- Scripts de backup de Qdrant

**Día 17-18: Testing**
- Probar con documentos reales
- Test de concurrencia (50-100 usuarios)

**Día 19-21: Deploy y documentación**
- README con instrucciones
- Variables de entorno
- Scripts de inicio/parada

---

## LO QUE SE QUITÓ (Y POR QUÉ)

### ❌ Eliminado
1. **Rust completamente**: Python es suficiente para este caso, más fácil de mantener
2. **Redis + Celery**: BackgroundTasks de FastAPI maneja carga moderada
3. **PostgreSQL inicial**: SQLite suficiente para desarrollo y MVP
4. **Múltiples workers**: Un solo servicio API con async
5. **WebSockets**: SSE o polling simple es más fácil
6. **Autenticación JWT**: Iniciar sin auth, agregar después
7. **Monitoreo complejo**: Logs básicos de FastAPI
8. **Versionado de docs**: Solo insert/delete

### ✅ Se mantiene
- Qdrant (es el corazón del sistema)
- Embeddings optimizados (ONNX + sentence-transformers)
- Docker (portabilidad)
- Arquitectura básica RAG
- Soporte multi-formato

---

## RECURSOS NECESARIOS

### Mínimos
- **CPU**: 4 cores
- **RAM**: 8GB (2GB Qdrant + 2GB modelo + 2GB API + 2GB OS)
- **Disco**: 20GB (5GB modelos + 10GB datos + 5GB sistema)

### Recomendados para producción
- **CPU**: 8 cores
- **RAM**: 16GB
- **Disco**: 50GB SSD

---

## COMANDOS DE INICIO RÁPIDO

```bash
# 1. Crear estructura
mkdir rag-simple && cd rag-simple
mkdir -p app frontend data uploads

# 2. Crear archivos (copiar código de arriba)
# - docker-compose.yml
# - app/main.py
# - app/requirements.txt
# - app/Dockerfile
# - frontend/index.html

# 3. Iniciar todo (descargará modelo automáticamente)
docker compose up --build

# Esperar a que Ollama descargue llama3.2:1b (1-2 min)
# Verás: "pulling manifest... pulling model..."

# 4. Verificar que todo esté arriba
docker compose ps
# Debe mostrar: ollama, qdrant, api (todos "running")

# 5. Verificar Ollama
curl http://localhost:11435/api/tags
# Debe listar: llama3.2:1b

# 6. Probar API docs
# (en Windows: start; en Linux: xdg-open; en Mac: open)
open http://localhost:8001/docs

# 7. Probar upload
curl -X POST "http://localhost:8001/upload" \
  -F "file=@test.txt"

# 8. Probar query (con Ollama)
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"q": "¿Qué dice el documento?"}'

# Deberías recibir una respuesta generada por el LLM
```

---

## EXTENSIONES FUTURAS (Cuando funcione básico)

### Fase 2: Mejoras de UX
- Barra de progreso real en frontend
- Preview de chunks antes de responder
- Historial de conversaciones

### Fase 3: Escalabilidad
- Migrar a PostgreSQL
- Agregar Redis para cache de embeddings
- Múltiples workers con Celery

### Fase 4: Features avanzadas
- Autenticación con JWT
- Roles (admin/usuario)
- Metrics y logs estructurados
- Backups automáticos

---

## CRITERIOS DE ÉXITO SIMPLIFICADOS

- [ ] Subir documento TXT/PDF/DOCX y procesarlo en <60s
- [ ] Hacer query y obtener chunks relevantes en <2s
- [ ] 20 usuarios concurrentes sin errores
- [ ] Sistema corre en laptop con 8GB RAM
- [ ] Startup completo en <10 segundos
- [ ] Documentación clara para otro dev usarlo

---

## ¿POR QUÉ ESTA VERSIÓN ES VIABLE?

1. **Un solo lenguaje**: Python para todo (ecosistema maduro)
2. **Sin distributed systems**: No Redis, no Celery, no orquestación compleja
3. **Background tasks**: FastAPI lo incluye gratis
4. **SQLite**: Cero configuración para empezar
5. **Frontend vanilla**: Sin framework JS complejo
6. **2-3 contenedores**: vs 7-8 de la versión original
7. **Modelo pequeño**: sentence-transformers es ligero y rápido
