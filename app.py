import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from chunk_embed import chunk_text, embed_chunks
from llm import answer_question
from pdf_extract import extract_text_from_pdf
from retrieval import build_index, search

app = FastAPI(title="ML Paper Chatbot")

# Holds the currently uploaded paper (one paper at a time, kept in memory).
state = {"chunks": None, "index": None, "filename": None}


class Question(BaseModel):
    question: str
    k: int = 4


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a PDF: extract text, chunk it, embed it, and index it."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pages = extract_text_from_pdf(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")
    finally:
        os.remove(tmp_path)

    chunks = chunk_text(pages)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text found in this PDF (it may be a scanned image).",
        )

    chunks, embeddings = embed_chunks(chunks)
    state["chunks"] = chunks
    state["index"] = build_index(embeddings)
    state["filename"] = file.filename

    return {"filename": file.filename, "pages": len(pages), "chunks": len(chunks)}


@app.post("/ask")
def ask(body: Question):
    """Ask a question about the uploaded paper."""
    if state["index"] is None:
        raise HTTPException(status_code=400, detail="Upload a paper first via /upload.")

    results = search(state["index"], state["chunks"], body.question, k=body.k)

    try:
        answer = answer_question(body.question, results)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "answer": answer,
        "sources": [
            {"page": r["page"], "score": round(r["score"], 3), "preview": r["text"][:200]}
            for r in results
        ],
    }
