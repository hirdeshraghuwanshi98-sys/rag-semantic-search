# 📂 Enterprise Semantic Search & Conversational RAG Ecosystem

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.1-green?style=for-the-badge&logo=chainlink)](https://www.langchain.com/)
[![VectorDB](https://img.shields.io/badge/FAISS-Enabled-0172B2?style=for-the-badge&logo=meta)](https://github.com/facebookresearch/faiss)
[![Dashboard](https://img.shields.io/badge/Streamlit-v1.45-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)

An end-to-end, enterprise-grade Retrieval-Augmented Generation (RAG) platform designed to ingest unstructured corporate knowledge assets (PDFs, TXT), compute localized dense vector embeddings, and serve high-context conversational query resolutions using open-source Large Language Models (LLMs) with zero infrastructure operational costs.

---

## 🚀 Live Production Links & Access
* **Interactive Frontend Dashboard:** [Streamlit Service UI](https://rag-semantic-search-sfnle2sbzjtx62wfnqucqq.streamlit.app/)

### 🔑 Demo Evaluation Credentials
To bypass the secure access administrative boundary on the live production interface, please utilize the following credentials:
* **Username:** `admin`
* **Password:** `admin123`

---

## 📊 RAG Architecture Performance Metrics

The architecture uses an isolated pipeline executing dynamic semantic ranking across local data matrices.

| Retrieval Vector Store Engine | Embeddings Model Layer | Synthesis LLM Engine | Average Search Latency | Cost Mapping | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Meta FAISS (CPU-Optimized)** | **all-MiniLM-L6-v2** | **Llama-3-8B-Instruct** | **~140ms** | 💰 **$0.00 (Free)** | 🏆 **Active Champion** |

---

## 🛠️ Step-by-Step System Walkthrough

### Step 1: Knowledge Base Ingestion (`ingest.py`)
Parses, normalizes, chunks text payload components using a recursive text splitter matrix, calculates mathematical multi-dimensional dense embeddings, and registers index targets safely.

<img src="images/prediction.png" alt="Knowledge Base Vector Ingestion Pipeline Visualization" width="100%">

### Step 2: Conversational UI & Synthesis Dashboard (`app.py`)
Loads local indexing parameters, handles administrative layer validations, scales query context inputs, metrics execution logs, and presents fully contextualized LLM answers.

<img src="images/dashboard.png" alt="Streamlit RAG Conversational Evaluation Interface" width="100%">

---

## 📂 Repository Blueprint

```text
rag-semantic-search/
│
├── ingest.py                 # Automated document processing & vector storage generator
├── app.py                    # Multi-tab operational Streamlit RAG interface
│
├── documents/                # Corporate raw knowledge source directory
│   └── sample.txt            # Local context payload targets
│
├── vectorstore/              # Serialized vector database metrics matrices
│   ├── index.faiss           # Meta FAISS high-dimensional vector array index
│   └── index.pkl             # Persisted metadata catalog matrix
│
├── logs/
│   └── rag_system.log        # Self-contained runtime validation execution logs
│
├── requirements.txt          # Explicitly pinned application package distributions
└── README.md                 # Interactive architectural summary documentation
