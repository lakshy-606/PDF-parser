import streamlit as st
import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_search_index():
    """Load FAISS index and metadata (cached for performance)"""
    try:
        index = faiss.read_index("faiss_index.bin")
        with open("metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        st.error(f"Error loading search index: {e}")
        return None, None

@st.cache_data
def search_similar(query, top_k=5, _index=None, _metadata=None):
    """Search for similar chunks using FAISS"""
    if _index is None or _metadata is None:
        return []

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Generate query embedding
        response = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )

        query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Search FAISS index
        scores, indices = _index.search(query_embedding, top_k)

        # Prepare results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(_metadata):
                result = _metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)

        return results

    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def main():
    # App header
    st.title("🔍 PDF Semantic Search")
    st.markdown("Search through your PDF documents using AI-powered semantic similarity")

    # Sidebar configuration
    st.sidebar.header("⚙️ Search Settings")

    # Load search index
    with st.spinner("Loading search index..."):
        index, metadata = load_search_index()

    if index is None or metadata is None:
        st.error("❌ Could not load search index. Make sure faiss_index.bin and metadata.pkl are in the current directory.")
        st.info("💡 Run your embedder script first to create the search index files.")
        return

    # Display index stats
    with st.sidebar:
        st.success(f"✅ Index loaded successfully")
        st.metric("Total Chunks", len(metadata))

        # Get unique PDFs and pages
        unique_pdfs = set(m.get('pdf_name', 'Unknown') for m in metadata)
        unique_pages = set(m.get('page', 'Unknown') for m in metadata if m.get('page'))

        st.metric("PDF Documents", len(unique_pdfs))
        st.metric("Pages", len(unique_pages))

        # Filter options
        st.subheader("🎛️ Filters")
        selected_pdf = st.selectbox("Filter by PDF:", ["All"] + list(unique_pdfs))

        # Number of results
        top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)

    # Main search interface
    st.subheader("🔎 Search Query")
    query = st.text_input("Enter your search query:", placeholder="e.g., financial performance, tax calculations, revenue growth...")

    # Search button and results
    if st.button("🚀 Search", type="primary"):
        if query.strip():
            with st.spinner("Searching..."):
                # Perform search
                results = search_similar(query, top_k, index, metadata)

                # Filter results by PDF if selected
                if selected_pdf != "All":
                    results = [r for r in results if r.get('pdf_name') == selected_pdf]

                # Display results
                if results:
                    st.success(f"Found {len(results)} results for: **{query}**")

                    # Results container
                    for i, result in enumerate(results, 1):
                        with st.expander(
                            f"📄 Result {i}: Page {result.get('page', '?')} | Score: {result['similarity_score']:.4f}",
                            expanded=i <= 3  # Auto-expand top 3 results
                        ):
                            # Result metadata
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("📖 PDF Name", result.get('pdf_name', 'Unknown'))

                            with col2:
                                st.metric("📃 Page", result.get('page', 'Unknown'))

                            with col3:
                                st.metric("🏷️ Element Type", result.get('type', 'Unknown'))

                            # Similarity score bar
                            st.progress(result['similarity_score'])
                            st.caption(f"Similarity Score: {result['similarity_score']:.4f}")

                            # Content
                            st.markdown("**Content:**")
                            st.text_area(
                                "Content",
                                value=result['text'],
                                height=150,
                                key=f"content_{i}",
                                label_visibility="collapsed"
                            )

                            # 🔧 FIXED: Replace nested expander with st.code directly
                            st.markdown("**📋 Copy Text:**")
                            st.code(result['text'], language="text")

                else:
                    st.warning("No results found. Try a different query or check your search terms.")
        else:
            st.warning("Please enter a search query.")


if __name__ == "__main__":
    main()
