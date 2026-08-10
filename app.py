import streamlit as st
import os
import time
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================================================
# LOGGING
# =========================================================

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/rag_system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================================
# ADMIN CREDENTIALS
# =========================================================

try:
    ADMIN_USER = st.secrets["ADMIN_USER"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]

except Exception:
    ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")


# =========================================================
# LOAD VECTORSTORE
# =========================================================

@st.cache_resource
def load_vectorstore():

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vectorstore is self-generated locally.
        # Deserialization is safe because the index was
        # created by our own ingest.py pipeline.

        vectorstore = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    except Exception as e:

        logging.error(
            f"Failed to load vector store: {str(e)}"
        )

        st.error(
            "⚠️ Vectorstore not found. "
            "Please run 'python ingest.py' first."
        )

        return None


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    st.set_page_config(
        page_title="Semantic Search",
        page_icon="🔍",
        layout="wide"
    )

    # =====================================================
    # AUTHENTICATION
    # =====================================================

    st.sidebar.title("🔐 Production Gateway")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:

        st.title("🔍 Semantic Search System")
        st.markdown("---")

        st.info(
            "👈 Please log in from the sidebar to continue."
        )

        user_input = st.sidebar.text_input("Username")

        password_input = st.sidebar.text_input(
            "Password",
            type="password"
        )

        if st.sidebar.button("Login"):

            if (
                user_input == ADMIN_USER
                and password_input == ADMIN_PASSWORD
            ):

                st.session_state["authenticated"] = True
                st.rerun()

            else:
                st.sidebar.error("❌ Invalid Credentials")

        st.stop()

    # =====================================================
    # MAIN UI
    # =====================================================

    st.title("🔍 Semantic Search System")
    st.markdown("---")

    st.sidebar.subheader("🎛️ Search Configuration")

    k_value = st.sidebar.slider(
        "Number of Results (k)",
        min_value=1,
        max_value=5,
        value=3
    )

    question = st.text_input(
        "🔍 Enter your search query:"
    )

    # IMPORTANT:
    # Initialize before using it.
    vectorstore = None

    # =====================================================
    # SEARCH
    # =====================================================

    if question:

        vectorstore = load_vectorstore()

        if vectorstore:

            start_time = time.time()

            retriever = vectorstore.as_retriever(
                search_kwargs={"k": k_value}
            )

            retrieved_docs = retriever.invoke(question)

            latency = (time.time() - start_time) * 1000

            logging.info(
                f"Query: '{question}' processed in "
                f"{latency:.2f}ms with k={k_value}"
            )

            # =================================================
            # BUILD CONTEXT
            # =================================================

            context = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            # =================================================
            # GROQ LLM
            # =================================================

            try:

                from groq import Groq

                # Get API key from Streamlit Secrets
                # or environment variable.

                try:
                    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

                except Exception:
                    GROQ_API_KEY = os.environ.get(
                        "GROQ_API_KEY"
                    )

                if not GROQ_API_KEY:

                    raise ValueError(
                        "GROQ_API_KEY is not configured."
                    )

                groq_client = Groq(
                    api_key=GROQ_API_KEY
                )

                prompt = f"""
You are a helpful assistant.

Using ONLY the context below, answer the question clearly
and concisely.

If the answer is not in the context, say:
"I could not find this in the documents."

Context:
{context}

Question:
{question}

Answer:
"""

                llm_response = (
                    groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        max_tokens=300
                    )
                )

                answer = (
                    llm_response
                    .choices[0]
                    .message
                    .content
                )

                # =================================================
                # GENERATED ANSWER
                # =================================================

                st.subheader("🤖 Generated Answer")

                st.success(answer)

            except Exception as e:

                logging.error(
                    f"LLM generation failed: {str(e)}"
                )

                st.warning(
                    "⚠️ LLM answer generation unavailable. "
                    "Showing raw results only."
                )

            # =================================================
            # SOURCE CHUNKS
            # =================================================

            st.subheader("📚 Source Chunks")

            st.caption(
                f"Retrieved {len(retrieved_docs)} chunks "
                f"in {latency:.2f}ms"
            )

            for i, doc in enumerate(retrieved_docs):

                with st.expander(
                    f"📍 Source {i + 1}"
                ):

                    st.write(
                        doc.page_content
                    )

                    st.markdown(
                        f"**Source:** `{doc.metadata}`"
                    )

        else:

            st.warning(
                "⚠️ Vectorstore could not be loaded."
            )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    main()
