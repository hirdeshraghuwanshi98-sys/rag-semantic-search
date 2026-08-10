"""
Streamlit app for semantic search over your documents.

Type a query -> it's embedded -> compared against the FAISS index ->
top-k most similar chunks are shown, with an optional LLM-generated
answer built from those chunks.

Run with: streamlit run app.py
"""

import os
import time
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    try:
        return FAISS.load_local(
            "vectorstore", embeddings, allow_dangerous_deserialization=True
        )
    except Exception:
        return None


def generate_answer(question, context):
    """Optional: ask Groq's LLM to answer using only the retrieved chunks."""
    try:
        from groq import Groq

        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None

        client = Groq(api_key=api_key)
        prompt = (
            "Answer the question using only the context below. "
            "If the answer isn't in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def main():
    st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")
    st.title("🔍 Semantic Document Search")

    vectorstore = load_vectorstore()
    if vectorstore is None:
        st.error("No vector index found. Run `python ingest.py` first.")
        return

    k = st.sidebar.slider("Number of results (k)", 1, 5, 3)
    question = st.text_input("Enter your search query:")

    if not question:
        return

    # Step 1: embed the query and search the FAISS index for similar chunks
    start = time.time()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(question)
    latency_ms = (time.time() - start) * 1000

    st.caption(f"Retrieved {len(results)} chunks in {latency_ms:.0f}ms")

    # Step 2 (optional): generate a natural-language answer from the chunks
    context = "\n\n".join(doc.page_content for doc in results)
    answer = generate_answer(question, context)
    if answer:
        st.subheader("Answer")
        st.success(answer)

    # Step 3: show the raw retrieved chunks with their source
    st.subheader("Matching Chunks")
    for i, doc in enumerate(results, start=1):
        with st.expander(f"Source {i}"):
            st.write(doc.page_content)
            st.caption(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    main()
    
