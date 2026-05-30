import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure file-based logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/rag_system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_documents():
    docs = []
    folder_path = "documents"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.warning(f"Created target directory root '{folder_path}' because it did not exist.")
        return docs

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                docs.extend(loader.load())
                logging.info(f"Successfully processed plain-text node asset: {file}")
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
                logging.info(f"Successfully processed document PDF layout: {file}")
        except Exception as e:
            logging.error(f"Failed to ingest file tracking row {file}: {str(e)}")
            
    return docs

def main():
    print("🚀 Initializing Enterprise Ingestion Sequence Engine...")
    logging.info("Starting knowledge base pipeline synchronization run.")
    
    documents = load_documents()
    
    if not documents:
        print("❌ Ingestion halted: No viable .pdf or .txt source documents found in /documents/ target folder.")
        logging.warning("Ingestion stopped: Empty source document array context.")
        return

    # Split documents into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=75,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"📦 Document decomposition verified: Generated {len(chunks)} high-density context chunks.")
    logging.info(f"Decomposed documentation landscape into {len(chunks)} isolated chunk vectors.")

    # Generate sentence embeddings
    print("🧠 Computing multi-dimensional embedding matrices via all-MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("💾 Indexing metadata vectors inside local Meta FAISS data structures...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save FAISS index to disk
    vectorstore.save_local("vectorstore")
    print("🏆 Production Vector Database constructed and written to /vectorstore/ successfully!")
    logging.info("Vector database compilation task execution run closed successfully.")

if __name__ == "__main__":
    main()
