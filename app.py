import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def main():

    st.title("Semantic Search System")

    question = st.text_input("Ask a question from your documents:")

    if question:

        vectorstore = load_vectorstore()

        docs = vectorstore.similarity_search(question, k=3)

        st.subheader("Retrieved Results")

        for i, doc in enumerate(docs):

            st.write(f"Result {i+1}:")
            st.write(doc.page_content)
            st.write("------")


if __name__ == "__main__":
    main()