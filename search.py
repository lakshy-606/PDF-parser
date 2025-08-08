import os
import sys
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def search_similar(query, top_k=5):
    """Search for similar chunks using FAISS"""
    try:
        # Load FAISS index
        index = faiss.read_index("faiss_index.bin")
        # Load metadata
        with open("metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Generate query embedding
        response = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )

        query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Search FAISS index
        scores, indices = index.search(query_embedding, top_k)

        # Prepare results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(metadata):
                result = metadata[idx].copy()
                result['similarity_score'] = float(score)
                # Truncate long text for display
                if len(result['text']) > 200:
                    result['preview'] = result['text'][:200] + "..."
                else:
                    result['preview'] = result['text']
                results.append(result)
        return results

    except Exception as e:
        print(f"Search failed: {e}")
        return []

def display_results(results, query):
    print(f"\nSearch Results for: '{query}'")
    print("="*60)
    if not results:
        print("No matching results found.")
        return
    for idx, result in enumerate(results, 1):
        print(f"\n{idx}. [Page {result.get('page', '?')}] [{result.get('type', '?')}] (Score: {result['similarity_score']:.4f})")
        print(f"PDF: {result.get('pdf_name', 'Unknown')}")
        print(f"Content Preview: {result['preview']}")
        print("-"*60)

def main():
    # Usage: python faiss_search.py "Your query here" [top_k]
    if len(sys.argv) > 1:
        # If query provided on command line
        query = sys.argv[1]
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    else:
        # Interactive prompt
        query = input("Enter your search query: ")
        top_k = 5

    results = search_similar(query, top_k=top_k)
    display_results(results, query)

if __name__ == "__main__":
    main()
