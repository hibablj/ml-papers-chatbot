from sentence_transformers import SentenceTransformer
from pdf_extract import extract_text_from_pdf

def chunk_text(pages_text: list[str], chunk_size: int=500, overlap: int = 50):

    chunks =[]

    for page_num, page_text in enumerate(pages_text, start=1):
        words = page_text.split()
        if not words: 
            continue

        start = 0
        while start < len(words): 
            end = start + chunk_size 
            chunk_words = words[start:end]
            chunk_str = " ". join(chunk_words)

            chunks.append({"text" : chunk_str, "page": page_num})

            if end >= len(words):
                break
            start = end - overlap

    return chunks 

def embed_chunks(chunks: list[dict], model_name: str = "all-MiniLM-L6-v2") : 

    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return chunks, embeddings 



if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python chunk_embed.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages.")

    chunks = chunk_text(pages)
    print(f"Created {len(chunks)} chunks.")

    chunks, embeddings = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks -> shape {embeddings.shape}")
    print("First chunk (page {}):".format(chunks[0]["page"]))
    print(chunks[0]["text"][:300])