# ml-papers-chatbot

# ML Paper Chatbot

Upload a machine learning research paper (PDF) and ask questions about it.
Uses retrieval-augmented generation (RAG): the paper is split into chunks,
embedded, and the most relevant chunks are retrieved and passed to an LLM
to answer your question.

## Status

Pipeline:
- [ Done ] PDF text extraction
- [ Done ] Chunking + embedding
- [ Done ] Vector search (retrieval)
- [ Done ] LLM answer generation (Groq)
- [ Done ] FastAPI endpoints (upload, ask)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env   # free key at console.groq.com
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000/docs to try it:

1. `POST /upload` with a PDF
2. `POST /ask` with `{"question": "What optimizer did they use?"}`

## Files

- `pdf_extract.py` - PDF to text, page by page
- `chunk_embed.py` - overlapping chunks + sentence-transformers embeddings
- `retrieval.py` - FAISS index and similarity search
- `llm.py` - builds the prompt and calls Groq
- `app.py` - FastAPI endpoints
