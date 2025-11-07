RAGIFY — Cloud-Ready RAG Chatbot with Multiple LLM Providers
RAGIFY is a Retrieval-Augmented Generation chatbot designed for Azure Web Apps deployment. It supports multiple LLM providers (OpenAI GPT, Google Gemini) via APIs, eliminating the need for local LLM hosting. It fetches content from the web, chunks and embeds it, stores vectors in ChromaDB, and answers questions through a lightweight HTTP server with a minimal HTML/JS UI.

Features
Multiple LLM provider support: OpenAI GPT and Google Gemini via APIs.​

WebBaseLoader to crawl seed URLs (configurable).​

Smarter chunking options with semantic/markdown/recursive strategies.​

Persistent vector store using Chroma in chroma_db/ (persisted in Azure).​

Lightweight server in Python; simple frontend (index.html + app.js + styles.css).​

Azure Web App ready with Docker support.​

Project Structure
main.py — Python server and RAG pipeline.​

index.html, app.js, styles.css — Minimal chat UI.​

main_initial.py — Earlier baseline kept for reference.​

requirements.txt — Python dependencies.​

.gitignore — Excludes .venv, chroma_db, caches, secrets.​

Prerequisites
Python 3.10+ with venv.​

API key for your chosen LLM provider:
- OpenAI API key (for GPT models)
- Google API key (for Gemini models)​

Quick Start

Local Development
Create and activate a virtual environment

python3 -m venv .venv

source .venv/bin/activate # macOS/Linux

.venv\Scripts\activate # Windows​

Install Python dependencies

pip install -r requirements.txt​

Configure environment variables

For OpenAI:
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-openai-api-key
export OPENAI_MODEL=gpt-3.5-turbo  # optional, default is gpt-3.5-turbo
```

For Google Gemini:
```bash
export LLM_PROVIDER=gemini
export GOOGLE_API_KEY=your-google-api-key
export GEMINI_MODEL=gemini-pro  # optional, default is gemini-pro
```

Optional settings:
```bash
export TEMPERATURE=0.1  # default 0.1
export CHUNKING_STRATEGY=semantic  # semantic|markdown|recursive
export CHROMA_DIR=chroma_db  # default chroma_db
export PORT=8000  # default 8000
export HOST=0.0.0.0  # default 0.0.0.0
```

Run the app

python3 main.py

Open http://localhost:8000 in your browser.​

Azure Web App Deployment

Option 1: Using Docker
Build and push to Azure Container Registry:
```bash
docker build -t ragify:latest .
# Tag and push to your Azure Container Registry
az acr login --name <your-acr-name>
docker tag ragify:latest <your-acr-name>.azurecr.io/ragify:latest
docker push <your-acr-name>.azurecr.io/ragify:latest
```

Create Azure Web App:
```bash
az webapp create --resource-group <your-rg> --plan <your-plan> --name <your-app-name> --deployment-container-image-name <your-acr-name>.azurecr.io/ragify:latest
```

Configure environment variables in Azure Portal or CLI:
```bash
az webapp config appsettings set --resource-group <your-rg> --name <your-app-name> --settings \
  LLM_PROVIDER=openai \
  OPENAI_API_KEY=<your-key> \
  OPENAI_MODEL=gpt-3.5-turbo \
  CHUNKING_STRATEGY=semantic \
  PORT=8000 \
  HOST=0.0.0.0
```

Option 2: Direct Deployment
Use Azure Web App's built-in Python support:
1. Push code to GitHub
2. Connect Azure Web App to your GitHub repository
3. Set environment variables in Azure Portal
4. Deploy

Important: Set environment variables in Azure Portal:
- `LLM_PROVIDER`: `openai` or `gemini`
- `OPENAI_API_KEY` or `GOOGLE_API_KEY` (depending on provider)
- `OPENAI_MODEL` or `GEMINI_MODEL` (optional)
- `TEMPERATURE` (optional, default 0.1)
- `CHUNKING_STRATEGY` (optional, default semantic)
- `CHROMA_DIR` (optional, default chroma_db)
- `PORT` (optional, default 8000)
- `HOST` (optional, default 0.0.0.0)

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

Switching LLM Providers
To switch between providers, simply change the `LLM_PROVIDER` environment variable:

For OpenAI:
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key
export OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo, gpt-4-turbo-preview, etc.
```

For Gemini:
```bash
export LLM_PROVIDER=gemini
export GOOGLE_API_KEY=your-key
export GEMINI_MODEL=gemini-pro  # or gemini-1.5-pro, etc.
```

Restart the application after changing providers.​

Knowledge Graph (Optional: LangGraph)
Add a graph agent step for entity extraction and KG lookups:

pip install langgraph

Create a graph with nodes: entity_extractor → retriever → generator.

Use NetworkX or Neo4j for a persistent KG, and wire a retriever node to query it. See LangGraph tutorials for RAG/agentic flows.​

Environment Variables

Required (based on provider):
- `LLM_PROVIDER`: `openai` or `gemini` (default: `openai`)
- `OPENAI_API_KEY`: Required if using OpenAI
- `GOOGLE_API_KEY`: Required if using Gemini

Optional:
- `OPENAI_MODEL`: OpenAI model name (default: `gpt-3.5-turbo`)
- `GEMINI_MODEL`: Gemini model name (default: `gemini-pro`)
- `TEMPERATURE`: LLM temperature (default: `0.1`)
- `USER_AGENT`: User agent string for web loader (default: RAGBot/1.0)
- `CHROMA_DIR`: Chroma persistence directory (default: `chroma_db`)
- `CHUNKING_STRATEGY`: `semantic|markdown|recursive` (default: `semantic`)
- `PORT`: Server port (default: `8000`)
- `HOST`: Server host (default: `0.0.0.0`)

Development Tips
Always keep API keys secure. Never commit them to version control. Use environment variables or Azure Key Vault in production.​

Avoid committing heavy/generated files:

.venv/, chroma_db/, pycache/, *.pyc, .DS_Store, .env — already in .gitignore.​

For Azure deployment, consider using Azure Key Vault for storing API keys securely.​

The ChromaDB persistence directory (`chroma_db/`) will be persisted in Azure Web App's storage. For production, consider using Azure Blob Storage or a dedicated database.​

Cost optimization: Use GPT-3.5-turbo for lower costs, or Gemini Pro for free tier usage. Monitor API usage in Azure Portal.​

Roadmap
Add LangGraph-based entity graph for better disambiguation.​

Multi-source loaders and scheduled recrawls.​

Citations with snippet highlighting.​

License
This project is licensed under the GNU Affero General Public License v3 (AGPL v3).

Copyright (C) 2025 Raghav Mehta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

**Important:** If you use this software in a network service (SaaS/web application),
you must make the source code available to all users under the same AGPL v3 license.

For commercial use or licensing inquiries, please contact the copyright holder.

Acknowledgments
LangChain for orchestration.​

ChromaDB for vector storage.​

OpenAI and Google for LLM APIs.​

Azure for cloud hosting platform.
