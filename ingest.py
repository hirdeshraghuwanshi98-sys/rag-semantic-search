import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


def load_documents():

    docs = []

    folder_path = "documents"

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        if file.endswith(".txt"):

            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())

        elif file.endswith(".pdf"):

            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

    return docs


def main():

    print("Loading documents...")

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("vectorstore")

    print("Vector store created successfully")


if __name__ == "__main__":
    main()