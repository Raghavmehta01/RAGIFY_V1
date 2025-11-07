# Azure-ready RAG application with multiple LLM provider support
# Set LLM_PROVIDER=openai or LLM_PROVIDER=gemini
# For OpenAI: set OPENAI_API_KEY
# For Gemini: set GOOGLE_API_KEY

import os, json
from typing import TypedDict, Annotated, Sequence, TYPE_CHECKING
from http.server import HTTPServer, SimpleHTTPRequestHandler

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages

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
# Get base directory (parent of app/) for proper path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.getenv("CHROMA_DIR", os.path.join(BASE_DIR, "chroma_db"))
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
# Set environment variables to avoid TensorFlow import issues
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("USE_TF", "0")  # Disable TensorFlow
os.environ.setdefault("USE_TORCH", "1")  # Use PyTorch instead

try:
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
except (ImportError, AttributeError, Exception) as e:
    print(f"⚠️  Warning: Could not load HuggingFaceEmbeddings: {e}")
    print("   This may be due to dependency conflicts. The server will continue but embeddings may not work.")
    print("   To fix: pip install 'protobuf>=4.21.6,<5.0.0' --force-reinstall")
    # Re-raise to prevent silent failures
    raise

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

# Track loaded files from docs/ folder
DOCS_FILE = os.path.join(PERSIST_DIR, ".loaded_docs.txt")

def load_docs_files():
    """Load list of previously indexed files from docs/ folder"""
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_doc_file(filepath):
    """Save file path to the list"""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(DOCS_FILE, 'a') as f:
        f.write(filepath + '\n')

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

# 3) Function to add uploaded files (PDF/Word/Text) to vectorstore - MUST be defined before load_docs_folder()
def add_file_to_vectorstore(file_content, filename, file_type):
    """Add a PDF, Word, or Text document to the vector store"""
    try:
        from langchain_core.documents import Document as LangChainDoc
        import io
        import tempfile
        
        new_docs = []
        
        if file_type == "application/pdf":
            # Handle PDF
            from langchain_community.document_loaders import PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                new_docs = loader.load()
                # Add filename to metadata
                for doc in new_docs:
                    doc.metadata['source'] = filename
                    doc.metadata['type'] = 'pdf'
            finally:
                os.unlink(tmp_path)
        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            # Handle Word documents (.docx and .doc)
            from docx import Document
            try:
                doc = Document(io.BytesIO(file_content))
                paragraphs = []
                for para in doc.paragraphs:
                    if para.text.strip():
                        paragraphs.append(para.text)
                
                if not paragraphs:
                    return False, "Word document appears to be empty or could not be read."
                
                # Create a single document from all paragraphs
                full_text = "\n\n".join(paragraphs)
                new_docs = [LangChainDoc(
                    page_content=full_text,
                    metadata={"source": filename, "type": "word"}
                )]
            except Exception as e:
                return False, f"Failed to read Word document: {str(e)}"
        elif file_type == "text/plain" or filename.lower().endswith('.txt'):
            # Handle text files
            try:
                # Decode the file content
                if isinstance(file_content, bytes):
                    text_content = file_content.decode('utf-8')
                else:
                    text_content = file_content
                
                if not text_content.strip():
                    return False, "Text file appears to be empty."
                
                new_docs = [LangChainDoc(
                    page_content=text_content,
                    metadata={"source": filename, "type": "text"}
                )]
            except UnicodeDecodeError:
                return False, "Failed to decode text file. Please ensure it's UTF-8 encoded."
            except Exception as e:
                return False, f"Failed to read text file: {str(e)}"
        else:
            return False, f"Unsupported file type: {file_type}. Please upload PDF, Word, or Text documents."
        
        if not new_docs:
            return False, "Failed to extract content from the file. The file might be empty or corrupted."
        
        # Check if documents have meaningful content
        total_content = sum(len(d.page_content) for d in new_docs)
        if total_content < 50:
            return False, f"File loaded but has very little content ({total_content} chars). Please ensure the file contains readable text."
        
        # Chunk the documents using the same strategy as URLs
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
        
        # Add to vectorstore
        global vectorstore
        vectorstore.add_documents(new_splits)
        
        # Track uploaded file (for UI uploads, not docs/ folder files)
        FILES_FILE = os.path.join(PERSIST_DIR, ".loaded_files.txt")
        os.makedirs(PERSIST_DIR, exist_ok=True)
        with open(FILES_FILE, 'a') as f:
            f.write(f"{filename}|{file_type}\n")
        
        # Update retriever to include new documents
        global retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 20,  # More context for detailed answers
            }
        )
        
        return True, f"Successfully added {len(new_splits)} chunks from {filename}. You can now ask questions about this document."
    except Exception as e:
        return False, f"Error processing file: {str(e)}"

# 3) Load documents from docs/ folder automatically
DOCS_DIR = os.path.join(BASE_DIR, "docs")
loaded_doc_files = load_docs_files()

def load_docs_folder():
    """Load all PDF, Word, and TXT files from docs/ folder"""
    if not os.path.exists(DOCS_DIR) or not os.path.isdir(DOCS_DIR):
        print("📁 docs/ folder not found. Skipping document loading.")
        return
    
    # Find all supported files
    supported_extensions = ['.pdf', '.doc', '.docx', '.txt']
    doc_files = []
    
    for root, dirs, files in os.walk(DOCS_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in supported_extensions:
                # Use relative path for tracking
                rel_path = os.path.relpath(file_path, BASE_DIR)
                doc_files.append((file_path, rel_path, file_ext))
    
    if not doc_files:
        print("📁 No documents found in docs/ folder.")
        return
    
    # Filter out already loaded files
    new_doc_files = [(fp, rp, ext) for fp, rp, ext in doc_files if rp not in loaded_doc_files]
    
    if not new_doc_files:
        print(f"✅ All {len(doc_files)} document(s) in docs/ already indexed.")
        return
    
    print(f"📥 Found {len(new_doc_files)} new document(s) in docs/ to index")
    
    # Ensure vectorstore exists
    global vectorstore
    if not 'vectorstore' in globals() or vectorstore is None:
        if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
        else:
            print("⚠️  Vector store not initialized. Creating new one...")
            vectorstore = Chroma.from_documents(documents=[], embedding=emb, persist_directory=PERSIST_DIR)
    
    # Process each file
    total_chunks = 0
    for file_path, rel_path, file_ext in new_doc_files:
        try:
            # Determine file type
            if file_ext == '.pdf':
                file_type = 'application/pdf'
            elif file_ext == '.docx':
                file_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            elif file_ext == '.doc':
                file_type = 'application/msword'
            elif file_ext == '.txt':
                file_type = 'text/plain'
            else:
                continue
            
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Use existing add_file_to_vectorstore function
            filename = os.path.basename(file_path)
            success, message = add_file_to_vectorstore(file_content, filename, file_type)
            
            if success:
                # Extract chunk count from message
                try:
                    chunks_str = message.split('added ')[1].split(' chunks')[0]
                    total_chunks += int(chunks_str)
                except:
                    pass
                
                # Track this file
                save_doc_file(rel_path)
                print(f"  ✅ {filename}: {message}")
            else:
                print(f"  ❌ {filename}: {message}")
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(file_path)}: {str(e)}")
    
    if total_chunks > 0:
        print(f"✅ Added {total_chunks} total chunks from docs/ folder")
    
    # Update retriever after adding new documents
    global retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})  # More context for detailed answers

# Load documents from docs/ folder
load_docs_folder()

# Changes made by Raghav Mehta with current timestamp: 2025-11-07 12:12:09
# Reason: Implemented MMR (Maximal Marginal Relevance) retriever for better diversity
# - MMR balances similarity with diversity to avoid redundant chunks
# - fetch_k=30 fetches more candidates before diversifying
# - lambda_mult=0.5 balances similarity (1.0) vs diversity (0.0)
# - Falls back to similarity search if MMR not supported
# 4) Tuned retriever with MMR (Maximal Marginal Relevance) for better diversity
# MMR balances similarity with diversity to avoid redundant chunks
if 'retriever' not in globals() or retriever is None:
    try:
        # Try MMR first (better for diverse, comprehensive context)
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance - diversifies results
            search_kwargs={
                "k": 20,           # Number of documents to return
                "fetch_k": 30,     # Fetch more candidates, then diversify
                "lambda_mult": 0.5  # Balance: 0 = more diverse, 1 = more similar
            }
        )
        print("✅ Using MMR retriever for better diversity")
    except Exception as e:
        # Fallback to similarity search if MMR not supported
        print(f"⚠️  MMR not available, using similarity search: {e}")
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 20,     # increased to 20 for maximum comprehensive context
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
        
        # Changes made by Raghav Mehta with current timestamp: 2025-11-07 12:12:09
        # Reason: Updated retriever to use MMR when adding new documents
        # - Ensures consistent retrieval strategy (MMR) after document additions
        # - Maintains diversity in retrieved chunks for better context
        # Update retriever to include new documents (with MMR if available)
        global retriever
        try:
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 20,
                    "fetch_k": 30,
                    "lambda_mult": 0.5
                }
            )
        except Exception:
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 20,  # increased for more comprehensive context for detailed answers
                }
            )
        
        return True, f"Successfully added {len(new_splits)} chunks from {url}. You can now ask questions about this content."
    except Exception as e:
        return False, f"Error: {str(e)}"

# ===== DYNAMIC PROMPT BUILDER FUNCTION =====
# Changes made by Raghav Mehta with current timestamp: 2025-11-07 12:12:09
# Reason: Enhanced prompt builder with specialized extraction guidance for meetings and projects
# - Added detailed meeting extraction requirements (metadata, discussions, decisions, action items)
# - Added comprehensive project information extraction (tech stack, team, clients, rules)
# - Improved output formatting with structured sections and tables
# - Added multi-project handling capabilities
def build_dynamic_prompt(format_type, content_type_hints, wants_explicit_detail, wants_brief):
    """Build a prompt template based on detected format preferences and content type"""
    
    base_instruction = """You are a helpful AI assistant that provides accurate and comprehensive answers based on the provided context.

Answer the question using ONLY the information from the Context below. If the context is insufficient, reply exactly: "I don't know based on the provided information."

"""
    
    # Format-specific instructions
    format_instructions = {
        "table": """FORMAT REQUIREMENTS:
- Present the information in a well-structured table format using markdown
- Use clear column headers that describe the data
- Organize data in rows and columns logically
- Include all relevant information from the context
- Make tables readable and well-formatted
- Use proper markdown table syntax with alignment

Example structure:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

""",
        "list": """FORMAT REQUIREMENTS:
- Use bullet points or numbered lists
- Organize information clearly with proper hierarchy
- Use sub-bullets when needed for nested information
- Make each point concise but informative
- Include all relevant details from the context
- Use markdown list syntax (- or * for bullets, 1. for numbered)

""",
        "structured": """FORMAT REQUIREMENTS:
- Organize information into clear sections with headers
- Use emoji headers (🏢, 🔗, 📦, ⚙️, 💬, 📅, etc.) for visual organization
- Create subsections for related topics
- Use tables for structured data when appropriate (issues, action items, comparisons)
- Use bullet points for lists within sections
- Maintain professional formatting with proper markdown
- Use ## for main sections, ### for subsections

Example structure:
## 🏢 Section 1
Content here with details...

### Subsection
More details...

## 🔗 Section 2
Content here...

""",
        "summary": """FORMAT REQUIREMENTS:
- Create a comprehensive summary with clear sections
- Organize by major themes or topics
- Use section headers for different topics (## for main sections)
- Include key points, decisions, and action items
- Use tables for structured data (issues, action items, comparisons) when appropriate
- Use bullet points for lists
- Maintain professional, organized structure
- Use emoji headers for visual organization if helpful

""",
        "comparison": """FORMAT REQUIREMENTS:
- Create a comparison table or structured comparison
- Highlight similarities and differences clearly
- Use side-by-side format or comparison table
- Include all relevant comparison points from context
- Make differences easy to identify
- Use markdown table format for side-by-side comparisons

""",
        "step_by_step": """FORMAT REQUIREMENTS:
- Present information as a step-by-step guide or process
- Number each step clearly (1., 2., 3., etc.)
- Include detailed instructions for each step
- Use clear transitions between steps
- Make it easy to follow sequentially
- Use markdown numbered list format

""",
    }
    
    # Get format instruction
    format_instruction = format_instructions.get(format_type, "")
    
    # Changes made by Raghav Mehta with current timestamp: 2025-11-07 12:12:09
    # Reason: Enhanced content type guidance for specialized extraction
    # - Meeting extraction: metadata, discussions, decisions, action items with table format
    # - Project extraction: tech stack, team structure, clients, rules with structured output
    # - Multi-project handling: clear separation and comparison capabilities
    # Content type specific guidance - Enhanced for meetings and projects
    content_guidance = ""
    if content_type_hints.get("meeting"):
        content_guidance = """CONTENT TYPE: Meeting/Transcript

EXTRACTION REQUIREMENTS:
1. **Meeting Metadata**: Extract date, time, participants, meeting type (standup, sprint, client call, etc.)
2. **Discussions**: Identify all topics discussed with key points and important quotes
3. **Decisions**: List all decisions made with context, decision-makers, and rationale
4. **Action Items**: Extract all action items with:
   - Task description
   - Owner/assignee
   - Deadline (if mentioned)
   - Dependencies
5. **Participants**: List all speakers with their roles and key contributions
6. **Next Steps**: Identify follow-up actions and scheduled items

OUTPUT FORMAT:
- Use structured sections: Overview, Discussions, Decisions, Action Items, Next Steps
- Use tables for action items (Task | Owner | Deadline | Status)
- Use bullet points for lists
- Include quotes when speakers make important statements
- Organize chronologically or by topic
- Use emoji headers (📅, 👥, 💬, ✅, 📋) for visual organization

"""
    elif content_type_hints.get("project"):
        content_guidance = """CONTENT TYPE: Project Information

EXTRACTION REQUIREMENTS:
1. **Project Identification**: Extract project name, type, status, and phase
2. **Technology Stack**: List all technologies:
   - Programming languages
   - Frameworks and libraries
   - Tools and platforms
   - Infrastructure/cloud services
   - Databases
3. **Team Structure**: Extract team members with:
   - Names and roles
   - Responsibilities
   - Reporting structure
4. **Clients/Stakeholders**: List clients/stakeholders with:
   - Names/companies
   - Relationship status
   - Key requirements
5. **Rules & Guidelines**: Extract project-specific:
   - Development standards
   - Process guidelines
   - Compliance requirements
   - Best practices
6. **Project Details**: Timeline, milestones, deliverables, challenges, solutions

OUTPUT FORMAT:
- Use clear project identifiers (project name or descriptive label)
- Use tables for team members (Name | Role | Responsibilities)
- Categorize tech stack clearly (Languages, Frameworks, Tools, Infrastructure)
- Use structured sections for different aspects
- Use emoji headers (🏢, 💻, 👥, 🤝, 📋, 📅) for visual organization

MULTI-PROJECT HANDLING:
- If multiple projects are mentioned, clearly separate information by project
- Use project names/identifiers in headers (## Project: [Name])
- Compare projects when relevant
- Specify which project each piece of information belongs to

"""
    elif content_type_hints.get("document"):
        content_guidance = """CONTENT TYPE: Document/Report
- Extract key information and main points
- Identify sections and subsections if present
- Note important data, statistics, or facts
- Maintain document structure if relevant

"""
    elif content_type_hints.get("data"):
        content_guidance = """CONTENT TYPE: Data/Information
- Present data clearly and accurately
- Use tables for numerical or structured data
- Include relevant statistics and facts
- Organize data logically

"""
    
    # Length guidance
    if wants_explicit_detail and not wants_brief:
        length_guidance = """LENGTH REQUIREMENTS:
- Provide a comprehensive answer (500+ words minimum)
- Expand on every point with full context
- Include all relevant details, examples, and explanations
- Be thorough and complete

"""
    elif not wants_brief:
        # Normal detailed answer (not explicit, not brief)
        length_guidance = """LENGTH REQUIREMENTS:
- Provide a detailed answer (300-500 words)
- Include sufficient context and explanations
- Balance comprehensiveness with readability

"""
    elif wants_brief:
        length_guidance = """LENGTH REQUIREMENTS:
- Keep the answer concise (2-3 sentences or brief format)
- Focus on essential information only

"""
    else:
        length_guidance = """LENGTH REQUIREMENTS:
- Provide a detailed answer (300-500 words)
- Include sufficient context and explanations
- Balance comprehensiveness with readability

"""
    
    # Combine all parts
    if format_instruction:
        full_prompt = f"""{base_instruction}{format_instruction}{content_guidance}{length_guidance}
Context:
{{context}}

Question:
{{question}}

Provide your answer in the requested format:"""
    else:
        # Default format instruction
        if wants_explicit_detail and not wants_brief:
            format_instruction = """FORMAT REQUIREMENTS:
- Write in full paragraphs (4-6 sentences each)
- Expand extensively on each point
- Use transition words to connect ideas
- Include context, examples, and explanations
- Provide comprehensive coverage (500+ words)

"""
        elif not wants_brief:
            format_instruction = """FORMAT REQUIREMENTS:
- Write in clear, well-structured paragraphs
- Include relevant details and context
- Use proper transitions between ideas
- Provide comprehensive coverage (300-500 words)

"""
        else:
            format_instruction = """FORMAT REQUIREMENTS:
- Write in clear, well-structured paragraphs
- Include relevant details and context
- Use proper transitions between ideas

"""
        
        full_prompt = f"""{base_instruction}{format_instruction}{content_guidance}{length_guidance}
Context:
{{context}}

Question:
{{question}}

Provide a comprehensive answer:"""
    
    return ChatPromptTemplate.from_template(full_prompt)

# Initialize LLM
# LLM will be created dynamically per request based on user selection

prompt = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant that provides accurate and comprehensive answers based on the provided context.

Answer the question using ONLY the information from the Context below. If the context is insufficient, reply exactly: "I don't know based on the provided information."

Your goal is to provide a complete, well-structured answer that fully addresses the question using all relevant information from the context. Structure your response with clear paragraphs that explain concepts thoroughly and include all relevant details, examples, and nuances from the provided context.

Context:
{context}

Question:
{question}

Provide a comprehensive answer:"""
)

# Changes made by Raghav Mehta with current timestamp: 2025-11-07 12:12:09
# Reason: Enhanced document formatting with rich metadata for better context understanding
# - Added project identifiers, meeting dates, document types, and categories to headers
# - Improved LLM's ability to understand document context and relationships
# - Better source attribution and metadata visibility for structured extraction
def format_docs(docs):
    """Format documents with enhanced structure and metadata for richer context, optimized for meetings and projects"""
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        title = doc.metadata.get("title", "")
        
        # Extract enhanced metadata for better context
        project = doc.metadata.get("project", "")
        meeting_date = doc.metadata.get("meeting_date", "")
        doc_type = doc.metadata.get("type", "")  # "meeting", "project_doc", etc.
        category = doc.metadata.get("category", "")  # "tech_stack", "team", "client", "rules", etc.
        
        # Build structured header with metadata
        header_parts = [f"Source {i}"]
        if project:
            header_parts.append(f"Project: {project}")
        if meeting_date:
            header_parts.append(f"Date: {meeting_date}")
        if doc_type:
            header_parts.append(f"Type: {doc_type}")
        if category:
            header_parts.append(f"Category: {category}")
        if title:
            header_parts.append(f"Title: {title}")
        if not title and source != "Unknown source":
            header_parts.append(f"Source: {source}")
        
        header = " | ".join(header_parts)
        section = f"[{header}]\n{doc.page_content}"
        formatted_parts.append(section)
    
    # Join with clear separators to help LLM understand structure
    return "\n\n---\n\n".join(formatted_parts)

# Changes made by Raghav Mehta with current timestamp: 2025-11-07 12:12:09
# Reason: Added query enhancement to improve retrieval relevance
# - Expands meeting queries with relevant terms (participants, action items, decisions)
# - Expands project queries with relevant terms (tech stack, team, clients, rules)
# - Improves semantic search accuracy by adding context-specific keywords
def enhance_query_for_retrieval(query, content_type_hints):
    """Enhance query with context-specific terms for better retrieval"""
    enhanced = query
    
    if content_type_hints.get("meeting"):
        # Add meeting-related terms to improve retrieval
        enhanced = f"{query} meeting transcript discussion participants action items decisions next steps"
    
    elif content_type_hints.get("project"):
        # Add project-related terms
        enhanced = f"{query} project documentation technology stack team members clients stakeholders rules guidelines"
    
    return enhanced

# RAG chain will be created dynamically per request based on selected provider/model

# ===== HR Agent with LangGraph =====
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from typing import TypedDict, Annotated, Sequence
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("⚠️  LangGraph not available. HR Agent will use simple LLM responses.")
    # Define dummy types for when LangGraph is not available
    BaseMessage = None
    add_messages = None

# HR Agent State (only used when LangGraph is available)
if HAS_LANGGRAPH:
    class HRState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        service: str
        context: dict
else:
    # Dummy class for when LangGraph is not available
    class HRState:
        pass

def create_hr_agent_graph(service_type):
    """Create LangGraph workflow for HR agent based on service type"""
    if not HAS_LANGGRAPH:
        return None
    
    # Define workflow nodes
    def route_request(state: HRState):
        """Route to appropriate handler based on service - returns state unchanged"""
        return state
    
    def handle_leave(state: HRState):
        """Handle leave management requests"""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Create specialized prompt for leave management
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an HR assistant specialized in Leave Management.
            You can help with:
            - Checking leave balance
            - Requesting leaves (sick, vacation, personal)
            - Viewing leave history
            - Explaining leave policies
            
            Be helpful, professional, and provide specific information when available.
            If you need employee ID or dates, ask for them."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        llm = get_llm(provider=LLM_PROVIDER)
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        
        return {
            "messages": [AIMessage(content=response.content if hasattr(response, 'content') else str(response))]
        }
    
    def handle_payroll(state: HRState):
        """Handle payroll management requests"""
        messages = state["messages"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an HR assistant specialized in Payroll Management.
            You can help with:
            - Viewing payslips
            - Understanding salary breakdown
            - Tax information
            - Deductions and benefits
            - Pay schedule
            
            Be professional and provide clear explanations.
            If you need specific employee information, ask for employee ID."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        llm = get_llm(provider=LLM_PROVIDER)
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        
        return {
            "messages": [AIMessage(content=response.content if hasattr(response, 'content') else str(response))]
        }
    
    def handle_recruitment(state: HRState):
        """Handle recruitment management requests"""
        messages = state["messages"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an HR assistant specialized in Recruitment Management.
            You can help with:
            - Posting job openings
            - Reviewing applications
            - Scheduling interviews
            - Candidate screening
            - Job descriptions
            
            Be professional and guide through the recruitment process.
            If you need job details or candidate information, ask for specifics."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        llm = get_llm(provider=LLM_PROVIDER)
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        
        return {
            "messages": [AIMessage(content=response.content if hasattr(response, 'content') else str(response))]
        }
    
    def general_response(state: HRState):
        """General HR response"""
        messages = state["messages"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful HR assistant. Provide friendly and professional assistance."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        llm = get_llm(provider=LLM_PROVIDER)
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        
        return {
            "messages": [AIMessage(content=response.content if hasattr(response, 'content') else str(response))]
        }
    
    # Router function for conditional edges
    def route_decision(state: HRState):
        """Route to appropriate handler based on service"""
        service = state.get("service", "")
        if service == "leave":
            return "handle_leave"
        elif service == "payroll":
            return "handle_payroll"
        elif service == "recruitment":
            return "handle_recruitment"
        return "general_response"
    
    # Build graph
    workflow = StateGraph(HRState)
    
    # Add nodes
    workflow.add_node("route", route_request)
    workflow.add_node("handle_leave", handle_leave)
    workflow.add_node("handle_payroll", handle_payroll)
    workflow.add_node("handle_recruitment", handle_recruitment)
    workflow.add_node("general_response", general_response)
    
    # Set entry point
    workflow.set_entry_point("route")
    
    # Set conditional edges from route
    workflow.add_conditional_edges(
        "route",
        route_decision,
        {
            "handle_leave": "handle_leave",
            "handle_payroll": "handle_payroll",
            "handle_recruitment": "handle_recruitment",
            "general_response": "general_response"
        }
    )
    
    # All handlers end
    workflow.add_edge("handle_leave", END)
    workflow.add_edge("handle_payroll", END)
    workflow.add_edge("handle_recruitment", END)
    workflow.add_edge("general_response", END)
    
    return workflow.compile()

# Simple HR agent (fallback if LangGraph not available)
def simple_hr_agent(service: str, message: str):
    """Simple HR agent without LangGraph"""
    service_prompts = {
        "leave": """You are an HR assistant specialized in Leave Management.
        You can help with:
        - Checking leave balance
        - Requesting leaves (sick, vacation, personal)
        - Viewing leave history
        - Explaining leave policies
        
        Be helpful, professional, and provide specific information when available.""",
        "payroll": """You are an HR assistant specialized in Payroll Management.
        You can help with:
        - Viewing payslips
        - Understanding salary breakdown
        - Tax information
        - Deductions and benefits
        - Pay schedule
        
        Be professional and provide clear explanations.""",
        "recruitment": """You are an HR assistant specialized in Recruitment Management.
        You can help with:
        - Posting job openings
        - Reviewing applications
        - Scheduling interviews
        - Candidate screening
        - Job descriptions
        
        Be professional and guide through the recruitment process."""
    }
    
    prompt_text = service_prompts.get(service, "You are a helpful HR assistant.")
    prompt = ChatPromptTemplate.from_template(f"{prompt_text}\n\nUser: {{message}}\n\nAssistant:")
    
    llm = get_llm(provider=LLM_PROVIDER)
    response = llm.invoke(prompt.format(message=message))
    
    return response.content if hasattr(response, 'content') else str(response)

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        """Serve static files (HTML, CSS, JS)"""
        # Get base directory (parent of app/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        frontend_dir = os.path.join(base_dir, 'frontend')
        
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
            
            # Look for file in frontend directory
            full_path = os.path.join(frontend_dir, file_path)
            
            # If file doesn't exist in frontend, try root (for backward compatibility)
            if not os.path.exists(full_path):
                full_path = os.path.join(base_dir, file_path)
            
            ext = os.path.splitext(file_path)[1]
            content_type = mime_types.get(ext, 'application/octet-stream')
            
            try:
                with open(full_path, 'rb') as f:
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
        # Normalize path (remove query string and trailing slashes)
        path = self.path.split('?')[0].rstrip('/')
        if not path:
            path = '/'
        
        if path == "/hr-agent":
            # HR Agent endpoint
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body) if body else {}
                service = (data.get("service") or "").strip()
                message = (data.get("message") or "").strip()
                
                if not service or not message:
                    self.send_error(400, "service and message are required")
                    return
                
                # Process HR agent request
                if HAS_LANGGRAPH:
                    try:
                        # Create or get graph for this service
                        graph = create_hr_agent_graph(service)
                        if graph:
                            # Use LangGraph workflow
                            state = {
                                "messages": [HumanMessage(content=message)],
                                "service": service,
                                "context": {}
                            }
                            result = graph.invoke(state)
                            response_text = result["messages"][-1].content if result.get("messages") else "No response generated"
                        else:
                            response_text = simple_hr_agent(service, message)
                    except Exception as e:
                        print(f"LangGraph error: {e}")
                        response_text = simple_hr_agent(service, message)
                else:
                    response_text = simple_hr_agent(service, message)
                
                payload = {"response": response_text}
                blob = json.dumps(payload).encode("utf-8")
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(blob)
            except Exception as e:
                msg = json.dumps({"response": f"Error: {str(e)}"}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(msg)
            return
        
        if path == "/add-url":
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
        
        if path == "/upload-file":
            # Handle file uploads (PDF/Word)
            try:
                import cgi
                import tempfile
                
                content_type = self.headers.get("Content-Type", "")
                if not content_type.startswith("multipart/form-data"):
                    self.send_error(400, "Content-Type must be multipart/form-data")
                    return
                
                # Parse multipart form data using cgi
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': content_type,
                        'CONTENT_LENGTH': self.headers.get("Content-Length", "0")
                    }
                )
                
                if 'file' not in form:
                    self.send_error(400, "No file provided")
                    return
                
                file_item = form['file']
                if not hasattr(file_item, 'filename') or not file_item.filename:
                    self.send_error(400, "No filename provided")
                    return
                
                filename = file_item.filename
                file_content = file_item.file.read()
                file_type = file_item.type or "application/octet-stream"
                
                # Validate file type
                allowed_types = [
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/msword",
                    "text/plain"
                ]
                
                if file_type not in allowed_types:
                    # Check by extension as fallback
                    ext = os.path.splitext(filename)[1].lower()
                    if ext == ".pdf":
                        file_type = "application/pdf"
                    elif ext in [".docx", ".doc"]:
                        file_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if ext == ".docx" else "application/msword"
                    elif ext == ".txt":
                        file_type = "text/plain"
                    else:
                        self.send_error(400, f"Unsupported file type. Please upload PDF (.pdf), Word (.docx, .doc), or Text (.txt) files.")
                        return
                
                # Check file size (limit to 10MB)
                max_size = 10 * 1024 * 1024  # 10MB
                if len(file_content) > max_size:
                    self.send_error(400, f"File too large. Maximum size is 10MB. Your file is {len(file_content) / 1024 / 1024:.2f}MB.")
                    return
                
                success, message = add_file_to_vectorstore(file_content, filename, file_type)
                
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
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(msg)
            return
        
        if path != "/ask":
            self.send_error(404, f"Not Found: {path}")
            return
        
        # Handle /ask endpoint
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
            
            # Changes made by Raghav Mehta with current timestamp: 2025-11-07 12:12:09
            # Reason: Updated retriever initialization to use MMR for better diversity
            # - Uses MMR retriever by default for better chunk diversity
            # - Falls back to similarity search if MMR not supported
            # Ensure we have the latest retriever (in case it was updated)
            global retriever
            if retriever is None:
                global vectorstore
                try:
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 20, "fetch_k": 30, "lambda_mult": 0.5}
                    )
                except Exception:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
            
            # Changes made by Raghav Mehta with current timestamp: 2025-11-07 12:12:09
            # Reason: Added query enhancement before retrieval to improve document matching
            # - Detects content type (meeting/project) early for query expansion
            # - Enhances query with relevant terms before semantic search
            # - Improves retrieval accuracy for specialized queries
            # ===== QUERY ENHANCEMENT (before retrieval) =====
            # Detect content type hints early for query enhancement
            question_lower_preview = q.lower()
            content_type_hints_preview = {
                "meeting": any(word in question_lower_preview for word in [
                    "meeting", "transcript", "discussion", "conversation", "call", "recording",
                    "standup", "sprint", "retrospective", "planning"
                ]),
                "project": any(word in question_lower_preview for word in [
                    "project", "tech stack", "technology", "team", "team members",
                    "client", "clients", "rules", "guidelines", "process", "workflow",
                    "framework", "stack", "tools", "infrastructure"
                ]),
            }
            
            # Enhance query for better retrieval
            enhanced_query = enhance_query_for_retrieval(q, content_type_hints_preview)
            
            # Retrieve documents with enhanced query
            docs = retriever.get_relevant_documents(enhanced_query)
            
            if not docs:
                # No context found - return early
                payload = {
                    "answer": "I don't know based on the provided information. No relevant documents were found in the knowledge base.",
                    "sources": ["No sources found for this query."],
                    "markdown": "### Answer:\nI don't know based on the provided information. No relevant documents were found in the knowledge base.\n\n### Sources:\n1. No sources found for this query.",
                    "needs_url": True
                }
                blob = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(blob)
                return
            
            # ===== DYNAMIC FORMAT & CONTENT DETECTION =====
            question_lower = q.lower()
            
            # Check for brief keywords first (explicit request for short answers)
            wants_brief = any(word in question_lower for word in [
                "brief", "short", "concise", "quick", "one sentence", "in short", "precise", "be precise"
            ])
            
            # Check for explicit detail phrases (strong indicators of detailed request)
            explicit_detail_phrases = ["in detail", "in depth", "more detail", "full detail", "detailed explanation", "comprehensive explanation"]
            has_explicit_detail_phrase = any(phrase in question_lower for phrase in explicit_detail_phrases)
            
            # Check for explicit detail keywords (strong indicators)
            explicit_detail_keywords = [
                "detailed", "comprehensive", "elaborate", "expand", "expanded", 
                "full explanation", "complete explanation", "thorough", "extensive"
            ]
            has_explicit_detail_keyword = any(word in question_lower for word in explicit_detail_keywords)
            
            # Determine if user explicitly wants detailed/expanded answer
            # Only use aggressive 500+ words prompt when EXPLICITLY requested
            wants_explicit_detail = has_explicit_detail_phrase or has_explicit_detail_keyword
            
            # For normal questions, provide balanced detailed answers (not forced 500+ words)
            # Only set wants_detail to True if explicitly requested OR if it's a general question (not brief)
            if wants_brief:
                wants_detail = False
                wants_explicit_detail = False
            elif wants_explicit_detail:
                wants_detail = True  # User explicitly asked for detailed answer
            else:
                # Normal question - provide balanced detailed answer (300-500 words), not forced 500+
                wants_detail = True  # Still provide detailed answer, but not forced 500+ words
                wants_explicit_detail = False  # Not explicitly requested
            
            # ===== FORMAT PREFERENCE DETECTION =====
            # Detect what format/structure the user wants based on their query
            format_keywords = {
                "summary": ["summary", "summarize", "summarise", "overview", "brief overview"],
                "table": ["table", "tabular", "in a table", "as a table", "create a table", "tabular format"],
                "list": ["list", "bullet", "bullets", "bullet points", "numbered list", "itemize", "items", "points"],
                "structured": ["structured", "organized", "organize", "format", "formatted", "sections", "categorized", "categorize"],
                "comparison": ["compare", "comparison", "difference", "differences", "versus", "vs", "contrast"],
                "step_by_step": ["step", "steps", "process", "procedure", "how to", "guide", "tutorial", "walkthrough"],
            }
            
            # Detect primary format preference
            detected_format = None
            for fmt, keywords in format_keywords.items():
                if any(kw in question_lower for kw in keywords):
                    detected_format = fmt
                    break
            
            # Changes made by Raghav Mehta with current timestamp: 2025-11-07 12:12:09
            # Reason: Enhanced content type detection with expanded keywords for better classification
            # - Added more meeting-related keywords (standup, sprint, retrospective, planning, action items)
            # - Added comprehensive project-related keywords (tech stack, team members, clients, stakeholders, rules, infrastructure)
            # - Improves automatic detection of query intent for specialized prompt selection
            # Content type hints (what type of content is being discussed?) - Enhanced with project detection
            content_type_hints = {
                "meeting": any(word in question_lower for word in [
                    "meeting", "transcript", "discussion", "conversation", "call", "recording",
                    "standup", "sprint", "retrospective", "planning", "action items", "decisions"
                ]),
                "project": any(word in question_lower for word in [
                    "project", "tech stack", "technology", "team", "team members", "teammate",
                    "client", "clients", "stakeholder", "rules", "guidelines", "process", 
                    "workflow", "framework", "stack", "tools", "infrastructure", "database",
                    "programming language", "libraries", "api", "deployment"
                ]),
                "document": any(word in question_lower for word in [
                    "document", "report", "file", "text", "content", "specification"
                ]),
                "data": any(word in question_lower for word in [
                    "data", "information", "facts", "statistics", "numbers", "metrics"
                ]),
            }
            
            # Debug: Log detection results
            print(f"🔍 Question: {q[:100]}...")
            print(f"   Explicit detail phrase detected: {has_explicit_detail_phrase}")
            print(f"   Explicit detail keyword detected: {has_explicit_detail_keyword}")
            print(f"   Wants explicit detail (500+ words): {wants_explicit_detail}")
            print(f"   Wants detail (normal): {wants_detail}")
            print(f"   Wants brief: {wants_brief}")
            print(f"   📋 Detected format: {detected_format or 'default'}")
            print(f"   📄 Content type hints: {[k for k, v in content_type_hints.items() if v] or 'none'}")
            
            # Adjust temperature based on user intent (higher for explicit detailed, lower for brief)
            original_temp = TEMPERATURE
            if wants_explicit_detail and not wants_brief:
                # Much higher temperature for explicit detailed requests (0.6-0.7 range)
                adjusted_temp = min(0.7, TEMPERATURE + 0.6)
                print(f"   ⚡ Explicit detail mode: Temperature set to {adjusted_temp:.2f} (was {TEMPERATURE:.2f})")
            elif wants_detail and not wants_brief:
                # Moderate temperature for normal detailed answers
                adjusted_temp = min(0.5, TEMPERATURE + 0.2)
                print(f"   ⚡ Normal detail mode: Temperature set to {adjusted_temp:.2f} (was {TEMPERATURE:.2f})")
            elif wants_brief:
                # Lower temperature for more focused/concise answers
                adjusted_temp = max(0.0, TEMPERATURE - 0.05)
            else:
                adjusted_temp = TEMPERATURE
            
            # ===== DYNAMIC PROMPT SELECTION =====
            # Use dynamic prompt builder if format is detected, otherwise use default behavior
            if detected_format:
                # Use dynamic prompt builder for detected format
                enhanced_prompt = build_dynamic_prompt(detected_format, content_type_hints, wants_explicit_detail, wants_brief)
                print(f"   📝 Using dynamic prompt for format: {detected_format}")
            elif wants_explicit_detail and not wants_brief:
                # Use the aggressive 500+ words prompt ONLY when explicitly requested
                enhanced_prompt = ChatPromptTemplate.from_template(
                    """You are a helpful AI assistant that provides accurate and comprehensive answers based on the provided context.

Answer the question using ONLY the information from the Context below. If the context is insufficient, reply exactly: "I don't know based on the provided information."

🚨🚨🚨 CRITICAL: The user explicitly requested DETAILED/EXPANDED information. Your response MUST be at least 500 words. This is NOT optional - it is a MANDATORY REQUIREMENT.

MANDATORY REQUIREMENTS:
1. Your answer MUST be 500+ words - count your words and ensure you meet this minimum
2. Write ONLY in full paragraphs (4-6 sentences each) - NO bullet points, NO lists, NO short phrases
3. Expand extensively on EVERY single point, topic, and detail from the context
4. Include ALL relevant details, context, background, nuances, implications, and examples

WRITING STYLE:
- Use well-structured paragraphs with proper transitions
- Each paragraph should be 4-6 sentences explaining one topic in depth
- Connect ideas with transition words (Furthermore, Additionally, Moreover, etc.)
- Include specific examples, quotes, and detailed explanations from the context
- Do NOT summarize - EXPAND and ELABORATE on everything

Your goal is to provide a complete, extensive, well-structured answer that fully addresses the question using ALL relevant information from the context. Every topic mentioned should be explained in detail with full context.

Context:
{context}

Question:
{question}

🚨🚨🚨 FINAL REMINDER: Write at least 500 words. Use full paragraphs only. Expand extensively on every point. NO bullet points. NO brief summaries. Provide comprehensive, detailed explanations."""
                )
                print(f"   📝 Using explicit detail prompt with 500+ words requirement")
            elif wants_brief:
                # Use a brief prompt for explicit brief requests
                enhanced_prompt = ChatPromptTemplate.from_template(
                    """You are a helpful AI assistant that provides concise and precise answers.

Answer the question using ONLY the information from the Context below. If the context is insufficient, reply exactly: "I don't know based on the provided information."

Provide a brief, concise answer (2-3 sentences maximum).

Context:
{context}

Question:
{question}"""
                )
                print(f"   📝 Using brief prompt")
            elif wants_detail and not wants_brief:
                # Normal detailed answer (balanced, not forced 500+ words)
                enhanced_prompt = ChatPromptTemplate.from_template(
                    """You are a helpful AI assistant that provides accurate and comprehensive answers based on the provided context.

Answer the question using ONLY the information from the Context below. If the context is insufficient, reply exactly: "I don't know based on the provided information."

Provide a detailed, well-structured answer with full paragraphs (4-6 sentences each). Expand on each point with context, examples, and explanations. Use transition words to connect ideas. Aim for 300-500 words - be comprehensive but natural, not forced.

WRITING STYLE:
- Use well-structured paragraphs with proper transitions
- Each paragraph should be 4-6 sentences explaining one topic
- Connect ideas with transition words (Furthermore, Additionally, Moreover, etc.)
- Include relevant examples and explanations from the context
- Be thorough but natural - don't pad unnecessarily

Your goal is to provide a complete, well-structured answer that fully addresses the question using relevant information from the context.

Context:
{context}

Question:
{question}

Provide a comprehensive answer:"""
                )
                print(f"   📝 Using balanced detailed prompt (300-500 words)")
            else:
                # Default: Use balanced prompt
                enhanced_prompt = ChatPromptTemplate.from_template(
                    """You are a helpful AI assistant that provides accurate and comprehensive answers based on the provided context.

Answer the question using ONLY the information from the Context below. If the context is insufficient, reply exactly: "I don't know based on the provided information."

Provide a clear, well-structured answer with proper paragraphs. Include relevant details and context. Be comprehensive but concise.

Context:
{context}

Question:
{question}

Provide a clear and comprehensive answer:"""
                )
                print(f"   📝 Using default balanced prompt")
            
            # Create dynamic RAG chain with selected LLM
            try:
                # Get LLM with potentially adjusted temperature
                if provider.lower() == "openai":
                    from langchain_openai import ChatOpenAI
                    # Force longer responses for explicit detail mode (500+ words)
                    max_tokens = 2000 if (wants_explicit_detail and not wants_brief) else (1500 if (wants_detail and not wants_brief) else 1000)
                    selected_llm = ChatOpenAI(
                        model=model or OPENAI_MODEL,
                        temperature=adjusted_temp,
                        api_key=OPENAI_API_KEY,
                        max_tokens=max_tokens
                    )
                    print(f"   📊 OpenAI max_tokens: {max_tokens}")
                elif provider.lower() == "gemini":
                    # For Gemini, we need to recreate with adjusted temp
                    import google.generativeai as genai
                    from langchain_core.language_models.chat_models import BaseChatModel
                    from langchain_core.messages import AIMessage
                    from langchain_core.outputs import ChatGeneration, ChatResult
                    
                    genai.configure(api_key=GOOGLE_API_KEY)
                    selected_model = model or GEMINI_MODEL
                    
                    class GeminiLLM(BaseChatModel):
                        model_name: str = selected_model
                        temperature: float = adjusted_temp
                        wants_detail: bool = False  # Store wants_detail as class attribute
                        
                        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                            model = genai.GenerativeModel(self.model_name)
                            prompt_parts = []
                            for msg in messages:
                                if hasattr(msg, 'content'):
                                    prompt_parts.append(str(msg.content))
                                else:
                                    prompt_parts.append(str(msg))
                            prompt = "\n".join(prompt_parts)
                            
                            # Configure generation with max_output_tokens for longer responses
                            if self.wants_detail:
                                # Force longer output for detail mode
                                gen_config = genai.types.GenerationConfig(
                                    temperature=self.temperature,
                                    max_output_tokens=2048  # Maximum tokens for Gemini
                                )
                                print(f"   📊 Gemini max_output_tokens: 2048 (detail mode)")
                            else:
                                gen_config = genai.types.GenerationConfig(
                                    temperature=self.temperature,
                                    max_output_tokens=1024  # Default for non-detail
                                )
                                print(f"   📊 Gemini max_output_tokens: 1024 (normal mode)")
                            
                            response = model.generate_content(
                                prompt,
                                generation_config=gen_config
                            )
                            
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
                    
                    selected_llm = GeminiLLM(
                        model_name=selected_model, 
                        temperature=adjusted_temp,
                        wants_detail=wants_explicit_detail and not wants_brief  # Pass wants_explicit_detail for 500+ words mode
                    )
                    print(f"   📊 Gemini detail mode: {wants_explicit_detail and not wants_brief} (explicit: {wants_explicit_detail})")
                elif provider.lower() == "groq":
                    from langchain_groq import ChatGroq
                    selected_llm = ChatGroq(
                        model_name=model or GROQ_MODEL,
                        temperature=adjusted_temp,
                        groq_api_key=GROQ_API_KEY
                    )
                else:
                    selected_llm = get_llm(provider=provider, model=model)
                
                print(f"🔄 Using {provider.upper()} model: {model or 'default'} (temp: {adjusted_temp:.2f})")
            except Exception as e:
                error_msg = f"Error initializing {provider} LLM: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
                payload = {
                    "answer": f"Error: {error_msg}. Please check your API keys and model configuration.",
                    "sources": [],
                    "markdown": f"### Error:\n{error_msg}",
                    "needs_url": False
                }
                blob = json.dumps(payload).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(blob)
                return
            
            # Create RAG chain with enhanced context formatting
            # Use enhanced_prompt if detail is requested, otherwise use default prompt
            # The format_docs function now provides better structured context
            rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | enhanced_prompt | selected_llm)
            
            # Invoke the chain with better error handling
            try:
                answer_response = rag_chain.invoke(q)
            except Exception as e:
                error_msg = f"Error generating response with {provider}: {str(e)}"
                print(f"❌ {error_msg}")
                print(f"   Full error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                payload = {
                    "answer": f"Error generating response: {str(e)}. Please try again or switch to a different model.",
                    "sources": [],
                    "markdown": f"### Error:\n{error_msg}",
                    "needs_url": False
                }
                blob = json.dumps(payload).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(blob)
                return
            
            # Extract content from ChatMessage if needed
            if hasattr(answer_response, 'content'):
                answer = answer_response.content
            else:
                answer = str(answer_response)
            
            # POST-PROCESSING: Expand short answers only in explicit detail mode (500+ words)
            answer_word_count = len(answer.split())
            if wants_explicit_detail and not wants_brief:
                print(f"   📏 Initial answer length: {answer_word_count} words")
                
                if answer_word_count < 400:
                    print(f"   ⚠️  Answer too short ({answer_word_count} words), attempting expansion...")
                    try:
                        # Get context for expansion
                        docs_for_expansion = retriever.get_relevant_documents(q)
                        expanded_context = format_docs(docs_for_expansion[:15])  # Use top 15 chunks
                        
                        # Create expansion prompt
                        expansion_prompt_text = f"""The following answer is too brief ({answer_word_count} words). Expand it to at least 500 words by:
1. Adding more details, examples, and explanations from the context
2. Expanding each point into full paragraphs (4-6 sentences each)
3. Including background information, context, and implications
4. Using transition words to connect ideas
5. NO bullet points, NO lists - ONLY full paragraphs

Original Answer:
{answer}

Context:
{expanded_context}

Question:
{q}

Write an expanded version (500+ words minimum, full paragraphs only):"""
                        
                        # Use the same LLM to expand (invoke with string directly)
                        from langchain_core.messages import HumanMessage
                        expansion_message = HumanMessage(content=expansion_prompt_text)
                        expanded_response = selected_llm.invoke([expansion_message])
                        
                        if hasattr(expanded_response, 'content'):
                            expanded_answer = expanded_response.content
                        else:
                            expanded_answer = str(expanded_response)
                        
                        expanded_word_count = len(expanded_answer.split())
                        if expanded_word_count > answer_word_count:
                            answer = expanded_answer
                            print(f"   ✅ Expanded answer length: {expanded_word_count} words")
                        else:
                            print(f"   ⚠️  Expansion didn't help, keeping original answer")
                    except Exception as e:
                        print(f"   ⚠️  Expansion failed: {str(e)}, using original answer")
                else:
                    print(f"   ✅ Answer length is good ({answer_word_count} words)")
            
            sources = []
            if docs:
                for d in docs[:5]:  # Show up to 5 sources for better context
                    src = d.metadata.get("source") or "Unknown"
                    title = d.metadata.get("title") or ""
                    label = f"{title} - {src}".strip(" -")
                    if label not in sources:  # Avoid duplicates
                        sources.append(label)
            
            if not sources:
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
        if self.path in ["/ask", "/add-url", "/hr-agent", "/upload-file"]:
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

