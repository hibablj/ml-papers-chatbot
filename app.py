import logging
import os
import tempfile
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from chunk_embed import chunk_text, embed_chunks
from llm import answer_question
from pdf_extract import extract_text_from_pdf
from retrieval import build_index, search

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="ML Paper Chatbot")

BASE_DIR = Path(__file__).parent
MAX_UPLOAD_MB = 15
MAX_SESSIONS = 50                # oldest paper is dropped when this is exceeded
MAX_QUESTION_CHARS = 500
UPLOADS_PER_HOUR = 10            # per IP
QUESTIONS_PER_MINUTE = 10        # per IP

# session_id -> {"chunks", "index", "filename", "created"}
sessions: dict[str, dict] = {}

# ip -> timestamps of recent requests
upload_log: dict[str, deque] = defaultdict(deque)
ask_log: dict[str, deque] = defaultdict(deque)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(log: dict, ip: str, limit: int, window_seconds: int, message: str):
    """Raises 429 if this IP made `limit` requests in the last window."""
    now = time.time()
    recent = log[ip]
    while recent and now - recent[0] > window_seconds:
        recent.popleft()
    if len(recent) >= limit:
        raise HTTPException(status_code=429, detail=message)
    recent.append(now)


def store_session(chunks, index, filename) -> str:
    if len(sessions) >= MAX_SESSIONS:
        oldest = min(sessions, key=lambda s: sessions[s]["created"])
        del sessions[oldest]
    session_id = uuid.uuid4().hex
    sessions[session_id] = {
        "chunks": chunks,
        "index": index,
        "filename": filename,
        "created": time.time(),
    }
    return session_id


class Question(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    k: int = Field(default=4, ge=1, le=8)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    """Upload a PDF: extract text, chunk it, embed it, and index it."""
    check_rate_limit(
        upload_log, client_ip(request), UPLOADS_PER_HOUR, 3600,
        "Too many uploads. Please try again later.",
    )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File is too large. The limit is {MAX_UPLOAD_MB} MB."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        pages = extract_text_from_pdf(tmp_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that file as a PDF.")
    finally:
        os.remove(tmp_path)

    chunks = chunk_text(pages)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text found in this PDF. It may be a scanned image.",
        )

    chunks, embeddings = embed_chunks(chunks)
    index = build_index(embeddings)
    session_id = store_session(chunks, index, file.filename)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
    }


@app.post("/ask")
def ask(request: Request, body: Question):
    """Ask a question about an uploaded paper."""
    check_rate_limit(
        ask_log, client_ip(request), QUESTIONS_PER_MINUTE, 60,
        "You are asking too fast. Please wait a moment.",
    )

    session = sessions.get(body.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Paper not found. It may have expired, so please upload it again.",
        )

    results = search(session["index"], session["chunks"], body.question, k=body.k)

    try:
        answer = answer_question(body.question, results)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        logger.exception("Groq request failed")
        raise HTTPException(
            status_code=502, detail="The language model request failed. Please try again."
        )

    return {
        "answer": answer,
        "sources": [
            {"page": r["page"], "score": round(r["score"], 3), "preview": r["text"][:200]}
            for r in results
        ],
    }
