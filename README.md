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
# optional: echo "GROQ_MODEL=llama-3.1-8b-instant" >> .env
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000 in your browser, upload a PDF, and ask a question.
The raw API docs are at http://127.0.0.1:8000/docs.

## Deploying

The repo includes a `Dockerfile` that listens on port 7860, which is what Hugging Face
Spaces expects. On Spaces, create a new Docker Space, push this repo to it, and add
`GROQ_API_KEY` under the Space's secrets. Do not commit the key. Any other host that
runs Docker (Render, Railway, Fly.io) works too; set the port and secret the same way.

Limits built into the app: 15 MB per PDF, 10 uploads per hour and 10 questions per
minute per IP, and at most 50 papers held in memory at once.

## Files

- `pdf_extract.py` - PDF to text, page by page
- `chunk_embed.py` - overlapping chunks + sentence-transformers embeddings
- `retrieval.py` - FAISS index and similarity search
- `llm.py` - builds the prompt and calls Groq
- `app.py` - FastAPI endpoints, per-user sessions, rate limits
- `static/index.html` - the web page
- `Dockerfile` - container for deployment
