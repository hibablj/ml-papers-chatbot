import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    """Loads the embedding model once and reuses it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Builds a FAISS index from chunk embeddings.

    Vectors are normalized so inner product equals cosine similarity.
    """
    vectors = np.ascontiguousarray(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search(index: faiss.IndexFlatIP, chunks: list[dict], question: str, k: int = 4) -> list[dict]:
    """Returns the k chunks most similar to the question, best first.

    Each result is {"text", "page", "score"}.
    """
    query = get_model().encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query)

    k = min(k, len(chunks))
    scores, ids = index.search(query, k)

    results = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        results.append({**chunks[i], "score": float(score)})
    return results
