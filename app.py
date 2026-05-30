import streamlit as st
import os
import time
import logging
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
                            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
                            llm = HuggingFaceEndpoint(
                                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                                temperature=temperature_value,
                                max_new_tokens=512
                            )
                            system_prompt = (
                                "You are an advanced enterprise document assistant. Use the following pieces of retrieved context "
                                "to answer the question completely and accurately. If you do not know the answer, state honestly "
                                "that the document base does not contain sufficient context. Do not make up answers.\n\n"
                                "Context:\n{context}"
                            )
                            prompt = ChatPromptTemplate.from_messages([
                                ("system", system_prompt),
                                ("human", "{input}"),
                            ])

                            def format_docs(docs):
                                return "\n\n".join(doc.page_content for doc in docs)

                            rag_chain = (
                                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                                | prompt
                                | llm
                                | StrOutputParser()
                            )
                            response = rag_chain.invoke(question)
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
