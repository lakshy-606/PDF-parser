import os
import sys
import re
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import faiss
import pickle

# Load environment variables
load_dotenv()

def chunk_by_paragraphs(text_content, pdf_name):
    """Split text into paragraphs and extract metadata"""
    # Split content into sections by metadata lines
    sections = re.split(r'(\[PAGE:.*?\]\s*\[TYPE:.*?\])', text_content)

    chunks = []
    current_page = None
    current_type = None

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Check if this is a metadata line
        if section.startswith('[PAGE:'):
            page_match = re.search(r'\[PAGE:\s*([^\]]+?)\]', section)
            type_match = re.search(r'\[TYPE:\s*([^\]]+?)\]', section)

            current_page = page_match.group(1).strip() if page_match else None
            current_type = type_match.group(1).strip() if type_match else None
            continue

        # This is actual content - split into paragraphs
        paragraphs = section.split('\n\n')

        for para in paragraphs:
            para = para.strip()
            # Skip empty paragraphs and short content
            if para and len(para) > 10:
                chunk_data = {
                    'text': para,
                    'page': current_page,
                    'type': current_type,
                    'pdf_name': pdf_name
                }
                chunks.append(chunk_data)

    print(f"Created {len(chunks)} paragraph chunks with metadata")
    return chunks

def create_embeddings_and_store(chunks):
    """Create embeddings for chunks and store in FAISS with metadata"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Extract just the text for embedding
    texts = [chunk['text'] for chunk in chunks]

    # Generate embeddings
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )

    # Convert to numpy array
    embeddings = np.array([v.embedding for v in response.data], dtype=np.float32)

    # Create FAISS index
    index = faiss.IndexFlatIP(1536)  # 1536 is embedding dimension
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings)

    # Prepare metadata (without text for storage efficiency)
    metadata = []
    for chunk in chunks:
        metadata.append({
            'text': chunk['text'],
            'page': chunk['page'],
            'type': chunk['type'], 
            'pdf_name': chunk['pdf_name']
        })

    # Save index and metadata
    faiss.write_index(index, "faiss_index.bin")
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Stored {len(chunks)} chunks in FAISS index with metadata")

def main():
    if len(sys.argv) < 2:
        print("Usage: python embedder.py <text_file> [pdf_name]")
        return

    text_file = sys.argv[1]
    pdf_name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(text_file)

    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = chunk_by_paragraphs(content, pdf_name)
    create_embeddings_and_store(chunks)
    print("Embedding and indexing complete. Files created: faiss_index.bin, metadata.pkl")

if __name__ == "__main__":
    main()
