# 🤖 Production RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with Gradio interface and LangGraph workflow.

## 🌟 Features

- 📁 **Multi-format Support**: PDF, TXT, DOCX documents
- 🔍 **Semantic Search**: ChromaDB vector database
- 🤖 **AI-Powered**: Groq Llama 3.3 70B model
- 🎨 **Beautiful UI**: Interactive Gradio interface
- 🔄 **LangGraph Workflow**: Structured RAG pipeline
- 🚀 **Production-Ready**: Error handling, logging, modular design

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
- Get your key from: https://console.groq.com/keys

### 3. Run the Application

```bash
python scripts/run_app.py
```

Open your browser at: http://localhost:7860

## 📖 Usage

### Upload Documents
1. Go to **"Document Management"** tab
2. Click **"Upload Documents"**
3. Select PDF, TXT, or DOCX files
4. Click **"Upload & Process"**

### Ask Questions
1. Go to **"Ask Questions"** tab
2. Type your question
3. Click **"Ask"**
4. Get AI-generated answers with sources

## 🏗️ Project Structure

```
rag-system/
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── rag_graph.py
│   └── app.py
├── data/
│   ├── documents/
│   └── vectordb/
├── scripts/
│   ├── ingest_documents.py
│   └── run_app.py
└── tests/
    └── test_rag.py
```

## 🛠️ Technologies

- **LangChain**: LLM framework
- **LangGraph**: Workflow orchestration
- **Groq**: Fast LLM inference
- **ChromaDB**: Vector database
- **Gradio**: Web interface
- **Sentence Transformers**: Embeddings

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.
