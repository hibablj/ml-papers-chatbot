import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Override with GROQ_MODEL in .env if your account cannot use the default.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Puts the retrieved excerpts and the question into one prompt."""
    excerpts = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks)
    return (
        "You are answering questions about a machine learning research paper.\n"
        "Use ONLY the excerpts below. If the answer is not in them, say you "
        "couldn't find it in the paper. Mention page numbers when helpful.\n\n"
        f"Excerpts:\n{excerpts}\n\n"
        f"Question: {question}"
    )


def answer_question(question: str, chunks: list[dict]) -> str:
    """Asks Groq to answer the question using the retrieved chunks."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to a .env file.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        temperature=0.2,
    )
    return response.choices[0].message.content
