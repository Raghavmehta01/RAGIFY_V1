# Ragify Setup Instructions

## Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
```

### 3. Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# LLM Provider (choose one: gemini, openai, or groq)
LLM_PROVIDER=gemini

# API Keys (at least one required based on LLM_PROVIDER)
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional Settings
TEMPERATURE=0.1
CHUNKING_STRATEGY=semantic
PORT=8000
HOST=0.0.0.0
```

### 6. Run the Application
```bash
python3 app/main.py
```

### 7. Access the Application
Open your browser and go to:
- Home: http://localhost:8000
- Chatbot: http://localhost:8000/chatbot.html
- HR Agent: http://localhost:8000/hr_agent.html

## Project Structure
```
ragify_v1/
├── app/
│   └── main.py          # Main backend server
├── frontend/
│   ├── index.html       # Home page
│   ├── chatbot.html     # Chatbot UI
│   ├── hr_agent.html    # HR Agent UI
│   ├── app.js           # Chatbot frontend logic
│   ├── hr_agent.js      # HR Agent frontend logic
│   └── styles.css       # Styling
├── docs/                # Documents folder (auto-loaded)
├── chroma_db/           # Vector database storage
├── config/              # Configuration files
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (create this)
```

## Features
- Multi-LLM support (OpenAI, Gemini, Groq)
- Document upload (PDF, Word, TXT)
- URL indexing
- HR Agent with LangGraph
- Persistent vector database

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### API Key Errors
- Make sure your `.env` file has the correct API keys
- Verify the API keys are valid and have sufficient credits

## Notes
- The `chroma_db/` folder stores the vector database
- Documents in `docs/` folder are automatically loaded on startup
- First run will take longer as it builds the vector database

