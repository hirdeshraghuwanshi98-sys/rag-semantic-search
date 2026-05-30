import streamlit as st
import os
import time
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/rag_system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    ADMIN_USER = st.secrets["ADMIN_USER"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
except Exception:
    ADMIN_USER = "admin"
    ADMIN_PASSWORD = "admin123"

@st.cache_resource
def load_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        logging.error(f"Failed to load vector store: {str(e)}")
        st.error("⚠️ Vectorstore not found. Please run 'python ingest.py' first.")
        return None

def main():
    st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")

    st.sidebar.title("🔐 Production Gateway")
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔍 Semantic Search System")
        st.markdown("---")
        st.info("👈 Please log in from the sidebar to continue.")
        user_input = st.sidebar.text_input("Username")
        password_input = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if user_input == ADMIN_USER and password_input == ADMIN_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid Credentials")
        st.stop()

    st.title("🔍 Semantic Search System")
    st.markdown("---")

    st.sidebar.subheader("🎛️ Search Configuration")
    k_value = st.sidebar.slider("Number of Results (k)", min_value=1, max_value=5, value=3)

    question = st.text_input("🔍 Enter your search query:")

    if question:
        vectorstore = load_vectorstore()
        if vectorstore:
            start_time = time.time()
            retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
            retrieved_docs = retriever.invoke(question)
            latency = (time.time() - start_time) * 1000
            logging.info(f"Query: '{question}' processed in {latency:.2f}ms with k={k_value}")

            st.subheader("🎯 Search Results")
            st.caption(f"Retrieved {len(retrieved_docs)} chunks in {latency:.2f}ms")

            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"📍 Result {i+1}"):
                    st.write(doc.page_content)
                    st.markdown(f"**Source:** `{doc.metadata}`")

if __name__ == "__main__":
    main()
