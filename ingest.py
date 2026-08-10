"""
Ingest documents from /documents, split into chunks, embed them,
and save a FAISS vector index to /vectorstore.

Run this once before starting the app: python ingest.py
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOCS_FOLDER = "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """Load every .txt and .pdf file from the documents folder."""
    docs = []
    os.makedirs(DOCS_FOLDER, exist_ok=True)

    for filename in os.listdir(DOCS_FOLDER):
        path = os.path.join(DOCS_FOLDER, filename)

        if filename.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif filename.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())

    return docs


def main():
    print("Loading documents...")
    documents = load_documents()

    if not documents:
        print("No .txt or .pdf files found in /documents. Add some and rerun.")
        return

    # Step 1: split documents into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=75)
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")

    # Step 2: embed each chunk into a vector
    print(f"Generating embeddings with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Step 3: build and save the FAISS vector index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print("Vector index saved to /vectorstore.")


if __name__ == "__main__":
    main()
    
