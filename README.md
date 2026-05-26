# Semantic Search System using LangChain and FAISS

An AI-powered semantic search application that retrieves relevant information from PDF and text documents using vector embeddings and similarity search.

---

## Features

- Semantic search over documents
- PDF and TXT support
- Vector similarity retrieval
- FAISS vector database
- Streamlit web interface
- Hugging Face embeddings
- Completely FREE (No OpenAI billing)

---

## Tech Stack

- Python
- LangChain
- Hugging Face
- FAISS
- Streamlit
- Sentence Transformers

---

## Project Architecture

User Query
↓
Embedding Generation
↓
FAISS Similarity Search
↓
Retrieve Relevant Chunks
↓
Display Results

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/rag-semantic-search.git
cd rag-semantic-search
pip install -r requirements.txt
```

## Run Locally

```bash
python ingest.py
streamlit run app.py
```

---

## Live Demo

https://rag-semantic-search-sfnle2sbzjtx62wfnqucqq.streamlit.app/

---


## Screenshots

### Home Page

![Home](screenshots/home.png)

### Results

![Results](screenshots/results.png)
---

## Resume Description

Developed a semantic search system using LangChain, Hugging Face embeddings, and FAISS to retrieve relevant information from PDF and text documents through vector similarity search.

---

## Future Improvements

- Chat memory
- Multi-document upload
- Conversational RAG
- LLM answer generation
- Cloud deployment optimization
