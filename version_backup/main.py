# Azure-ready RAG application with multiple LLM provider support
# Set LLM_PROVIDER=openai or LLM_PROVIDER=gemini
# For OpenAI: set OPENAI_API_KEY
# For Gemini: set GOOGLE_API_KEY

import os, json
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars only

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# ---- LLM Provider Settings ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # openai | gemini | groq
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
UA = os.getenv("USER_AGENT", "RAGBot/1.0 (+https://example.com/contact) LangChain-WebBaseLoader")

# Initialize LLM based on provider
def get_llm(provider=None, model=None):
    # Use provided provider or default to environment variable
    selected_provider = (provider or LLM_PROVIDER).lower()
    
    if selected_provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI provider")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or OPENAI_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
    elif selected_provider == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required when using Gemini provider")
        # Use google-generativeai directly due to version compatibility issues
        import google.generativeai as genai
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        genai.configure(api_key=GOOGLE_API_KEY)
        
        selected_model = model or GEMINI_MODEL
        
        class GeminiLLM(BaseChatModel):
            """Custom LangChain wrapper for Google Gemini"""
            model_name: str = selected_model
            temperature: float = TEMPERATURE
            
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                # Create model instance for this request
                model = genai.GenerativeModel(self.model_name)
                
                # Convert messages to prompt string
                prompt_parts = []
                for msg in messages:
                    if hasattr(msg, 'content'):
                        prompt_parts.append(str(msg.content))
                    else:
                        prompt_parts.append(str(msg))
                prompt = "\n".join(prompt_parts)
                
                # Generate response using the model instance
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(temperature=self.temperature)
                )
                
                # Extract text from response
                if hasattr(response, 'text'):
                    text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    text = response.candidates[0].content.parts[0].text
                else:
                    text = str(response)
                
                message = AIMessage(content=text)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            
            @property
            def _llm_type(self):
                return "gemini"
        
        return GeminiLLM(model_name=selected_model, temperature=TEMPERATURE)
    elif selected_provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required when using Groq provider")
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=model or GROQ_MODEL,
            temperature=TEMPERATURE,
            groq_api_key=GROQ_API_KEY
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {selected_provider}. Use 'openai', 'gemini', or 'groq'")

print(f"🤖 Default LLM Provider: {LLM_PROVIDER.upper()}")
if LLM_PROVIDER == "openai":
    print(f"   Default Model: {OPENAI_MODEL}")
elif LLM_PROVIDER == "gemini":
    print(f"   Default Model: {GEMINI_MODEL}")
elif LLM_PROVIDER == "groq":
    print(f"   Default Model: {GROQ_MODEL}")
print(f"✅ Multi-model support enabled - users can select provider in UI")

# ---- Build RAG at startup ----

loader = WebBaseLoader(
    web_paths=( "https://orangedatatech.com/team/",
                "https://www.w3schools.com/python/default.asp",
                "https://www.python.org/doc/",
                
               
               
               ),
    raise_for_status=False, continue_on_failure=True, trust_env=True,
)
loader.requests_kwargs = {"headers": {"User-Agent": UA}}
docs = loader.load()

# -------- Advanced chunking + persistent vector store --------
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# Try semantic chunker if available
try:
    from langchain_experimental.text_splitter import SemanticChunker
    HAS_SEMANTIC = True
except Exception:
    HAS_SEMANTIC = False

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = os.getenv("CHROMA_DIR", "chroma_db")
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "semantic")  # semantic | markdown | recursive

# 1) Choose a smarter splitter
if CHUNKING_STRATEGY == "semantic" and HAS_SEMANTIC:
    embedder_for_split = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    text_splitter = SemanticChunker(
        embedder_for_split,
        breakpoint_threshold_type="percentile",     # or "standard_deviation"
        breakpoint_threshold_amount=0.3,            # higher => fewer chunks
        min_chunk_size=300,
        buffer_size=64,
    )
    splits = text_splitter.split_documents(docs)
    

elif CHUNKING_STRATEGY == "markdown":
    headers_to_split_on = [("#","h1"),("##","h2"),("###","h3"),("####","h4")]
    md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_docs = []
    for d in docs:
        for md in md_header_splitter.split_text(d.page_content):
            md.metadata.update(d.metadata)  # preserve original metadata
            md_docs.append(md)
    # Light recursive pass inside sections
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    splits = text_splitter.split_documents(md_docs or docs)

else:
    # Default improved recursive settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,       # smaller for snappier local inference
        chunk_overlap=120,
        add_start_index=True, # keeps offsets in metadata
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    splits = text_splitter.split_documents(docs)

print(f"✂️ Created {len(splits)} chunks using '{CHUNKING_STRATEGY}' strategy")

# 2) Build or load vector store with persistence
emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Track loaded URLs in a file
URLS_FILE = os.path.join(PERSIST_DIR, ".loaded_urls.txt")

def load_urls():
    """Load list of previously indexed URLs"""
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_url(url):
    """Save URL to the list"""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(URLS_FILE, 'a') as f:
        f.write(url + '\n')

loaded_urls = load_urls()
new_urls = set(loader.web_paths) - loaded_urls

if new_urls:
    print(f"📥 Found {len(new_urls)} new URL(s) to index")
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        # Load existing vectorstore
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
        # Add new documents
        new_docs = [d for d in docs if d.metadata.get('source') in new_urls]
        if new_docs:
            # Re-chunk new documents
            if CHUNKING_STRATEGY == "semantic" and HAS_SEMANTIC:
                embedder_for_split = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                text_splitter_new = SemanticChunker(
                    embedder_for_split,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=0.3,
                    min_chunk_size=300,
                    buffer_size=64,
                )
                new_splits = text_splitter_new.split_documents(new_docs)
            elif CHUNKING_STRATEGY == "markdown":
                headers_to_split_on = [("#","h1"),("##","h2"),("###","h3"),("####","h4")]
                md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                md_docs = []
                for d in new_docs:
                    for md in md_header_splitter.split_text(d.page_content):
                        md.metadata.update(d.metadata)
                        md_docs.append(md)
                text_splitter_new = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
                new_splits = text_splitter_new.split_documents(md_docs or new_docs)
            else:
                text_splitter_new = RecursiveCharacterTextSplitter(
                    chunk_size=700,
                    chunk_overlap=120,
                    add_start_index=True,
                    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
                )
                new_splits = text_splitter_new.split_documents(new_docs)
            
            if new_splits:
                vectorstore.add_documents(new_splits)
                print(f"✅ Added {len(new_splits)} new chunks to vector store")
                for url in new_urls:
                    save_url(url)
    else:
        # Create new vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=emb,
            persist_directory=PERSIST_DIR,
        )
        for url in loader.web_paths:
            save_url(url)
        print(f"✅ Created new vector store with {len(splits)} chunks")
else:
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
        print("✅ Using existing vector store (no new URLs)")
    else:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=emb,
            persist_directory=PERSIST_DIR,
        )
        for url in loader.web_paths:
            save_url(url)
        print(f"✅ Created vector store with {len(splits)} chunks")

# 3) Tuned retriever
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,      # increased to get more context for better answers
    }
)
# -------- end advanced chunking block --------

# Function to add URLs dynamically to vectorstore
def add_url_to_vectorstore(url):
    """Add a new URL to the vector store"""
    try:
        # Check if URL already loaded
        loaded_urls_set = load_urls()
        will_reindex = url in loaded_urls_set
        if will_reindex:
            print(f"⚠️  URL {url} is tracked. Re-indexing to ensure it's in vector store...")
        
        # Load the URL
        new_loader = WebBaseLoader(
            web_paths=[url],
            raise_for_status=False,
            continue_on_failure=True,
            trust_env=True,
        )
        new_loader.requests_kwargs = {"headers": {"User-Agent": UA}}
        new_docs = new_loader.load()
        
        if not new_docs:
            return False, "Failed to load content from URL. The page might be JavaScript-heavy or require authentication."
        
        # Check if documents have meaningful content
        total_content = sum(len(d.page_content) for d in new_docs)
        if total_content < 100:
            return False, f"URL loaded but has very little content ({total_content} chars). Try a more specific page with actual documentation. For Python docs, try: https://docs.python.org/3/tutorial/ or https://docs.python.org/3/faq/"
        
        # Chunk the documents
        if CHUNKING_STRATEGY == "semantic" and HAS_SEMANTIC:
            embedder_for_split = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            text_splitter_new = SemanticChunker(
                embedder_for_split,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.3,
                min_chunk_size=300,
                buffer_size=64,
            )
            new_splits = text_splitter_new.split_documents(new_docs)
        elif CHUNKING_STRATEGY == "markdown":
            headers_to_split_on = [("#","h1"),("##","h2"),("###","h3"),("####","h4")]
            md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_docs = []
            for d in new_docs:
                for md in md_header_splitter.split_text(d.page_content):
                    md.metadata.update(d.metadata)
                    md_docs.append(md)
            text_splitter_new = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
            new_splits = text_splitter_new.split_documents(md_docs or new_docs)
        else:
            text_splitter_new = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=120,
                add_start_index=True,
                separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
            )
            new_splits = text_splitter_new.split_documents(new_docs)
        
        # Add to vectorstore (need to access global vectorstore)
        global vectorstore
        vectorstore.add_documents(new_splits)
        
        # Only save URL if it wasn't already tracked (to avoid duplicates)
        if url not in loaded_urls_set:
            save_url(url)
        
        # Update retriever to include new documents
        global retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5,  # increased to get more context
            }
        )
        
        return True, f"Successfully added {len(new_splits)} chunks from {url}. You can now ask questions about this content."
    except Exception as e:
        return False, f"Error: {str(e)}"

# Initialize LLM
# LLM will be created dynamically per request based on user selection

prompt = ChatPromptTemplate.from_template(
    """Answer the question strictly using the Context. Perform all reasoning internally and do not reveal steps.
If the context is insufficient, reply exactly: I don't know based on the provided information.

Constraints:
- Be concise and specific.
- Use bullet points only when listing.
- Do not invent sources or facts.

Context:
{context}

Question:
{question}

Final answer:"""
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# RAG chain will be created dynamically per request based on selected provider/model

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        """Serve static files (HTML, CSS, JS)"""
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
        try:
            # Map file extensions to MIME types
            mime_types = {
                '.html': 'text/html',
                '.css': 'text/css',
                '.js': 'application/javascript',
                '.json': 'application/json',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.svg': 'image/svg+xml',
            }
            
            file_path = self.path.lstrip('/')
            if not file_path:
                file_path = 'index.html'
            
            # Security: prevent directory traversal
            if '..' in file_path:
                self.send_error(403, "Forbidden")
                return
            
            ext = os.path.splitext(file_path)[1]
            content_type = mime_types.get(ext, 'application/octet-stream')
            
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_error(404, "File not found")
        except Exception as e:
            self.send_error(500, str(e))
    
    def do_POST(self):
        if self.path == "/add-url":
            # New endpoint to add URLs
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body) if body else {}
                url = (data.get("url") or "").strip()
                
                if not url:
                    self.send_error(400, "URL is required")
                    return
                
                # Validate URL format
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                
                success, message = add_url_to_vectorstore(url)
                
                payload = {"success": success, "message": message}
                blob = json.dumps(payload).encode("utf-8")
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(blob)
            except Exception as e:
                msg = json.dumps({"success": False, "message": f"Error: {str(e)}"}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            return
        
        if self.path != "/ask":
            self.send_error(404, "Not Found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body) if body else {}
            q = (data.get("question") or "").strip()
            if not q:
                self.send_error(400, "question is required")
                return
            
            # Get provider and model from request (default to environment settings)
            provider = data.get("provider") or LLM_PROVIDER
            model = data.get("model")
            
            # Create dynamic RAG chain with selected LLM
            selected_llm = get_llm(provider=provider, model=model)
            rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | selected_llm)
            
            answer_response = rag_chain.invoke(q)
            # Extract content from ChatMessage if needed
            if hasattr(answer_response, 'content'):
                answer = answer_response.content
            else:
                answer = str(answer_response)
            
            docs = retriever.get_relevant_documents(q)
            sources = []
            if docs:
                for d in docs[:3]:
                    src = d.metadata.get("source") or "Unknown"
                    title = d.metadata.get("title") or ""
                    label = f"{title} - {src}".strip(" -")
                    sources.append(label)
            else:
                sources.append("No sources found for this query.")
            
            # Check if answer indicates insufficient information
            answer_lower = answer.lower()
            needs_url = (
                "i don't know" in answer_lower or
                "insufficient" in answer_lower or
                "not found" in answer_lower or
                "no information" in answer_lower or
                (len(docs) == 0) or
                (len(answer) < 50 and "don't know" in answer_lower)
            )
            
            md_lines = ["### Answer:", answer, "", "### Sources:"]
            for i, s in enumerate(sources, 1):
                md_lines.append(f"{i}. {s}")
            
            payload = {
                "answer": answer,
                "sources": sources,
                "markdown": "\n".join(md_lines),
                "needs_url": needs_url
            }
            blob = json.dumps(payload).encode("utf-8")
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(blob)
        except Exception as e:
            msg = json.dumps({"detail": f"{type(e).__name__}: {e}"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_OPTIONS(self):
        if self.path in ["/ask", "/add-url"]:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
        else:
            super().do_OPTIONS()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Serving on http://{host}:{port}")
    HTTPServer((host, port), Handler).serve_forever()


# ORIGINAL CODE STARTS HERE


# # export OLLAMA_MODEL=mistral:7b && python3 main.py


# import os, time, json, requests
# from http.server import HTTPServer, SimpleHTTPRequestHandler
# from langchain_chroma import Chroma

# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.runnables import RunnablePassthrough
# from langchain.prompts import ChatPromptTemplate
# from langchain_ollama import OllamaLLM  # new import

# # ---- Settings ----
# OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
# MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")  # default to Mistral 7B
# UA = os.getenv("USER_AGENT", "RAGBot/1.0 (+https://example.com/contact) LangChain-WebBaseLoader")

# def wait_for_ollama(base=OLLAMA_BASE, timeout=60):
#     start = time.time()
#     while time.time() - start < timeout:
#         try:
#             r = requests.get(f"{base}/api/tags", timeout=2)
#             if r.status_code == 200:
#                 return
#         except Exception:
#             pass
#         time.sleep(1)
#     raise RuntimeError("Ollama not responding at 11434")

# # ---- Build RAG at startup ----
# wait_for_ollama()

# loader = WebBaseLoader(
#     web_paths=( "https://orangedatatech.com/team/",
#                 "https://www.w3schools.com/python/default.asp",
#                 "https://www.python.org/doc/",
                
               
               
#                ),
#     raise_for_status=False, continue_on_failure=True, trust_env=True,
# )
# loader.requests_kwargs = {"headers": {"User-Agent": UA}}
# docs = loader.load()

# # -------- Advanced chunking + persistent vector store --------
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# # Try semantic chunker if available
# try:
#     from langchain_experimental.text_splitter import SemanticChunker
#     HAS_SEMANTIC = True
# except Exception:
#     HAS_SEMANTIC = False

# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# PERSIST_DIR = os.getenv("CHROMA_DIR", "chroma_db")
# CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "semantic")  # semantic | markdown | recursive

# # 1) Choose a smarter splitter
# if CHUNKING_STRATEGY == "semantic" and HAS_SEMANTIC:
#     embedder_for_split = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     text_splitter = SemanticChunker(
#         embedder_for_split,
#         breakpoint_threshold_type="percentile",     # or "standard_deviation"
#         breakpoint_threshold_amount=0.3,            # higher => fewer chunks
#         min_chunk_size=300,
#         buffer_size=64,
#     )
#     splits = text_splitter.split_documents(docs)
    

# elif CHUNKING_STRATEGY == "markdown":
#     headers_to_split_on = [("#","h1"),("##","h2"),("###","h3"),("####","h4")]
#     md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
#     md_docs = []
#     for d in docs:
#         for md in md_header_splitter.split_text(d.page_content):
#             md.metadata.update(d.metadata)  # preserve original metadata
#             md_docs.append(md)
#     # Light recursive pass inside sections
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
#     splits = text_splitter.split_documents(md_docs or docs)

# else:
#     # Default improved recursive settings
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=700,       # smaller for snappier local inference
#         chunk_overlap=120,
#         add_start_index=True, # keeps offsets in metadata
#         separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
#     )
#     splits = text_splitter.split_documents(docs)

# print(f"✂️ Created {len(splits)} chunks using '{CHUNKING_STRATEGY}' strategy")

# # 2) Build or load vector store with persistence
# emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
#     vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
# else:
#     vectorstore = Chroma.from_documents(
#         documents=splits,
#         embedding=emb,
#         persist_directory=PERSIST_DIR,
#     )
#     # vectorstore.persist()  # optional on recent Chroma

# # 3) Tuned retriever
# retriever = vectorstore.as_retriever(
#     search_kwargs={
#         "k": 3,      # tighten to reduce context; raise if answers feel thin
#     }
# )
# # -------- end advanced chunking block --------

# # Use new langchain-ollama LLM
# llm = OllamaLLM(model=MODEL, base_url=OLLAMA_BASE, temperature=0.1)

# prompt = ChatPromptTemplate.from_template(
#     """Answer the question strictly using the Context. Perform all reasoning internally and do not reveal steps.
# If the context is insufficient, reply exactly: I don't know based on the provided information.

# Constraints:
# - Be concise and specific.
# - Use bullet points only when listing.
# - Do not invent sources or facts.

# Context:
# {context}

# Question:
# {question}

# Final answer:"""
# )

# def format_docs(docs):
#     return "\n\n".join(d.page_content for d in docs)

# rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm)

# class Handler(SimpleHTTPRequestHandler):
#     def do_POST(self):
#         if self.path != "/ask":
#             self.send_error(404, "Not Found")
#             return
#         try:
#             length = int(self.headers.get("Content-Length", "0"))
#             body = self.rfile.read(length).decode("utf-8")
#             data = json.loads(body) if body else {}
#             q = (data.get("question") or "").strip()
#             if not q:
#                 self.send_error(400, "question is required")
#                 return

#             answer = rag_chain.invoke(q)
#             docs = retriever.get_relevant_documents(q)
#             sources = []
#             if docs:
#                 for d in docs[:3]:
#                     src = d.metadata.get("source") or "Unknown"
#                     title = d.metadata.get("title") or ""
#                     label = f"{title} - {src}".strip(" -")
#                     sources.append(label)
#             else:
#                 sources.append("No sources found for this query.")

#             md_lines = ["### Answer:", str(answer), "", "### Sources:"]
#             for i, s in enumerate(sources, 1):
#                 md_lines.append(f"{i}. {s}")
#             payload = {"answer": str(answer), "sources": sources, "markdown": "\n".join(md_lines)}
#             blob = json.dumps(payload).encode("utf-8")

#             self.send_response(200)
#             self.send_header("Content-Type", "application/json")
#             self.send_header("Content-Length", str(len(blob)))
#             self.send_header("Access-Control-Allow-Origin", "*")
#             self.end_headers()
#             self.wfile.write(blob)
#         except Exception as e:
#             msg = json.dumps({"detail": f"{type(e).__name__}: {e}"}).encode("utf-8")
#             self.send_response(500)
#             self.send_header("Content-Type", "application/json")
#             self.send_header("Content-Length", str(len(msg)))
#             self.end_headers()
#             self.wfile.write(msg)

#     def do_OPTIONS(self):
#         if self.path == "/ask":
#             self.send_response(204)
#             self.send_header("Access-Control-Allow-Origin", "*")
#             self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
#             self.send_header("Access-Control-Allow-Headers", "Content-Type")
#             self.end_headers()
#         else:
#             super().do_OPTIONS()

# if __name__ == "__main__":
#     port = int(os.getenv("PORT", "8000"))
#     host = os.getenv("HOST", "0.0.0.0")
#     print(f"Serving on http://{host}:{port}")
#     HTTPServer((host, port), Handler).serve_forever()






