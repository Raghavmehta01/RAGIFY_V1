RAGIFY — Local RAG Chatbot with Ollama
RAGIFY is a simple Retrieval-Augmented Generation chatbot that runs locally using Ollama and an LLM (default: Mistral 7B). It fetches content from the web, chunks and embeds it, stores vectors in ChromaDB, and answers questions through a tiny HTTP server with a minimal HTML/JS UI.

Features
Local inference via Ollama; default model is mistral:7b.​

WebBaseLoader to crawl a seed URL (configurable).​

Smarter chunking options with semantic/markdown/recursive strategies.​

Persistent vector store using Chroma in chroma_db/ (ignored by Git).​

Lightweight server in Python; simple frontend (index.html + app.js + styles.css).​

Project Structure
main.py — Python server and RAG pipeline.​

index.html, app.js, styles.css — Minimal chat UI.​

main_initial.py — Earlier baseline kept for reference.​

requirements.txt — Python dependencies.​

.gitignore — Excludes .venv, chroma_db, caches, secrets.​

Prerequisites
Python 3.10+ with venv.​

Ollama installed and running locally.​

Model pulled in Ollama: mistral:7b or mistral:latest.​

Quick Start
Create and activate a virtual environment

python3 -m venv .venv

source .venv/bin/activate # macOS/Linux

.venv\Scripts\activate # Windows​

Install Python dependencies

pip install -r requirements.txt​

Start Ollama and pull a model

ollama serve

ollama pull mistral:7b # or mistral:latest

curl http://127.0.0.1:11434/api/tags # verify model appears​

Configure environment (optional)

export OLLAMA_BASE=http://127.0.0.1:11434

export OLLAMA_MODEL=mistral:7b

export USER_AGENT="RAGBot/1.0 (+contact@example.com)"

export CHUNKING_STRATEGY=semantic # semantic|markdown|recursive ​

Run the app

python3 main.py

Open index.html in a browser (or serve statically) and chat.​

How It Works
Load data

WebBaseLoader scrapes the seed URL; set web_paths in main.py to your source.​

Chunk documents

Choose one:

semantic (requires langchain-experimental)

markdown (header-aware)

recursive (default, robust)​

Embed and store

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector store: Chroma with persistence at chroma_db/​

Retrieve and generate

Similarity search retrieves top-k chunks; prompt + Ollama LLM produce final answer.​

Switching Models
Pull the model in Ollama:

ollama pull mistral:7b

Set env var:

export OLLAMA_MODEL=mistral:7b

Restart main.py.​

Knowledge Graph (Optional: LangGraph)
Add a graph agent step for entity extraction and KG lookups:

pip install langgraph

Create a graph with nodes: entity_extractor → retriever → generator.

Use NetworkX or Neo4j for a persistent KG, and wire a retriever node to query it. See LangGraph tutorials for RAG/agentic flows.​

Environment Variables
OLLAMA_BASE: Ollama server URL, default http://127.0.0.1:11434.​

OLLAMA_MODEL: e.g., mistral:7b.​

USER_AGENT: passed to the web loader.​

CHROMA_DIR: path of the Chroma persistence directory (default chroma_db).​

CHUNKING_STRATEGY: semantic|markdown|recursive. ​

Development Tips
If Ollama port is in use, kill the other process or change the port and OLLAMA_BASE.​

Avoid committing heavy/generated files:

.venv/, chroma_db/, pycache/, *.pyc, .DS_Store, .env — already in .gitignore.​

If Git rejects pushes due to >100 MB:

git rm -r --cached .venv chroma_db

git commit -m "Remove local artifacts"

git push​

Roadmap
Add LangGraph-based entity graph for better disambiguation.​

Multi-source loaders and scheduled recrawls.​

Citations with snippet highlighting.​

License
Choose a license (e.g., MIT) and add LICENSE.​

Acknowledgments
LangChain + LangGraph for orchestration.​

ChromaDB for vector storage.​

Ollama for local LLM serving.
