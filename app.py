import streamlit as st
import os
import time
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

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
    st.set_page_config(page_title="Enterprise RAG Ecosystem", page_icon="📊", layout="wide")

    st.sidebar.title("🔐 Production Gateway")
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("📊 Enterprise Knowledge Base RAG Ecosystem")
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

    st.title("📊 Enterprise Knowledge Base RAG Ecosystem")
    st.markdown("---")

    st.sidebar.subheader("🎛️ Retrieval Configuration")
    k_value = st.sidebar.slider("Retrieval Chunk Count (k)", min_value=1, max_value=5, value=3)
    temperature_value = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2)

    try:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
    except Exception:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token:
        st.warning("⚠️ HUGGINGFACEHUB_API_TOKEN not found. Running in Retrieval-Only mode.")

    question = st.text_input("🔍 Input query to explore document semantic spaces:")

    if question:
        vectorstore = load_vectorstore()
        if vectorstore:
            start_time = time.time()
            retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
            retrieved_docs = retriever.invoke(question)
            latency = (time.time() - start_time) * 1000
            logging.info(f"Query: '{question}' processed in {latency:.2f}ms with k={k_value}")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🤖 Generative LLM Response")
                if hf_token:
                    with st.spinner("Synthesizing response..."):
                        try:
                            client = InferenceClient(
                                model="HuggingFaceH4/zephyr-7b-beta",
                                token=hf_token
                            )
                            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                            full_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
                            response = client.text_generation(
                                full_prompt,
                                max_new_tokens=512,
                                temperature=temperature_value
                            )
                            st.write(response)
                        except Exception as e:
                            st.error(f"Generative engine error: {str(e)}")
                else:
                    st.info("💡 Provide a HuggingFace API Token to enable generative responses.")

            with col2:
                st.subheader("🎯 Semantic Search Results")
                st.caption(f"Retrieved {len(retrieved_docs)} chunks in {latency:.2f}ms")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"📍 Chunk {i+1}"):
                        st.write(doc.page_content)
                        st.markdown(f"**Source:** `{doc.metadata}`")

if __name__ == "__main__":
    main()
