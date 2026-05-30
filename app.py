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
    ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

@st.cache_resource
def load_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Vectorstore is self-generated locally — deserialization is safe
# as the pickle files are produced by our own ingest.py pipeline
        vectorstore = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True    # Safe: index built locally by ingest.py
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

        # Build context from retrieved chunks
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate answer using Groq LLM
        try:
            from groq import Groq
            groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            
            prompt = f"""You are a helpful assistant. 
Using ONLY the context below, answer the question clearly and concisely.
If the answer is not in the context, say "I could not find this in the documents."

Context:
{context}

Question: {question}
Answer:"""

            llm_response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            answer = llm_response.choices[0].message.content

            st.subheader("🤖 Generated Answer")
            st.success(answer)

        except Exception as e:
            logging.error(f"LLM generation failed: {str(e)}")
            st.warning("⚠️ LLM answer generation unavailable. Showing raw results only.")

        # Show source chunks below the answer
        st.subheader("📚 Source Chunks")
        st.caption(f"Retrieved {len(retrieved_docs)} chunks in {latency:.2f}ms")
        for i, doc in enumerate(retrieved_docs):
            with st.expander(f"📍 Source {i+1}"):
                st.write(doc.page_content)
                st.markdown(f"**Source:** `{doc.metadata}`")
            logging.info(f"Query: '{question}' processed in {latency:.2f}ms with k={k_value}")

            st.subheader("🎯 Search Results")
            st.caption(f"Retrieved {len(retrieved_docs)} chunks in {latency:.2f}ms")

            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"📍 Result {i+1}"):
                    st.write(doc.page_content)
                    st.markdown(f"**Source:** `{doc.metadata}`")

if __name__ == "__main__":
    main()
