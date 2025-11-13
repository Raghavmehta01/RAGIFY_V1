# Azure-ready RAG application with multiple LLM provider support
# Set LLM_PROVIDER=gemini or LLM_PROVIDER=groq
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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # gemini | groq
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
UA = os.getenv("USER_AGENT", "RAGBot/1.0 (+https://example.com/contact) LangChain-WebBaseLoader")

# Initialize LLM based on provider
def get_llm(provider=None, model=None):
    # Use provided provider or default to environment variable
    selected_provider = (provider or LLM_PROVIDER).lower()
    
    if selected_provider == "gemini":
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
        raise ValueError(f"Unsupported LLM provider: {selected_provider}. Use 'gemini' or 'groq'")

print(f"🤖 Default LLM Provider: {LLM_PROVIDER.upper()}")
if LLM_PROVIDER == "gemini":
    print(f"   Default Model: {GEMINI_MODEL}")
elif LLM_PROVIDER == "groq":
    print(f"   Default Model: {GROQ_MODEL}")
print(f"✅ Multi-model support enabled - users can select provider in UI")

# ---- Build RAG at startup ----

# Changes made by Raghav Mehta with current timestamp: 2025-11-07 13:52:27
# Reason: Added PayPlus 360 URLs for web scraping to build knowledge base
# - Added main PayPlus 360 page, legal notice, privacy policy, and terms of use
# - These URLs provide context about PayPlus 360 application for ODT team restructuring project
loader = WebBaseLoader(
    web_paths=(
        
        "https://www.payplus360.com/index.html",
        "https://www.payplus.com/PayPlus360",
        "https://www.payplus360.com/legal-notice.html",
        "https://www.payplus360.com/privacy-policy.html",
        "https://www.payplus360.com/terms-of-use.html",
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

# Try reranker if available
# Reranking improves document relevance by using cross-encoder models
# Set RERANKER_ENABLED=false to disable reranking
# Set RERANKER_MODEL to use a different model (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
# Set RERANK_TOP_K to control final number of documents after reranking (default: 20)
# Set RETRIEVE_K to control how many documents to fetch before reranking (default: 40)
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
try:
    if RERANKER_ENABLED:
        from sentence_transformers import CrossEncoder
        HAS_RERANKER = True
        # Initialize reranker model (using a lightweight cross-encoder)
        RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        try:
            reranker = CrossEncoder(RERANKER_MODEL)
            print(f"✅ Reranker initialized: {RERANKER_MODEL}")
        except Exception as e:
            print(f"⚠️  Could not load reranker model {RERANKER_MODEL}: {e}")
            print("   Reranking will be disabled. Install: pip install sentence-transformers")
            HAS_RERANKER = False
            reranker = None
    else:
        HAS_RERANKER = False
        reranker = None
        print("ℹ️  Reranking disabled via RERANKER_ENABLED=false")
except ImportError:
    HAS_RERANKER = False
    reranker = None
    if RERANKER_ENABLED:
        print("⚠️  Reranker not available. Install: pip install sentence-transformers")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Get base directory (parent of app/) for proper path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.getenv("CHROMA_DIR", os.path.join(BASE_DIR, "chroma_db"))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "hybrid")  # hybrid (semantic + recursive)

# Hybrid chunking function: combines semantic and recursive chunking
def hybrid_chunk_documents(documents):
    """Hybrid chunking: first semantic, then recursive for size control"""
    all_splits = []
    
    # Step 1: Semantic chunking (if available) for semantic boundaries
    if HAS_SEMANTIC:
        try:
            embedder_for_split = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            semantic_splitter = SemanticChunker(
                embedder_for_split,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.3,
                min_chunk_size=300,
                buffer_size=64,
            )
            semantic_splits = semantic_splitter.split_documents(documents)
            print(f"   📊 Semantic chunking created {len(semantic_splits)} initial chunks")
        except Exception as e:
            print(f"   ⚠️  Semantic chunking failed: {e}, using recursive only")
            semantic_splits = documents
    else:
        semantic_splits = documents
    
    # Step 2: Recursive character splitting for size control and overlap
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    all_splits = recursive_splitter.split_documents(semantic_splits)
    
    return all_splits

# 1) Use hybrid chunking
splits = hybrid_chunk_documents(docs)

print(f"✂️ Created {len(splits)} chunks using 'hybrid' strategy (semantic + recursive)")

# COMMENTED OUT: Other chunking strategies
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
#     
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
# 
# else:
#     # Default improved recursive settings
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=700,       # smaller for snappier local inference
#         chunk_overlap=120,
#         add_start_index=True, # keeps offsets in metadata
#         separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
#     )
#     splits = text_splitter.split_documents(docs)

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
            # Re-chunk new documents using hybrid chunking
            new_splits = hybrid_chunk_documents(new_docs)
            
            # COMMENTED OUT: Other chunking strategies
            # if CHUNKING_STRATEGY == "semantic" and HAS_SEMANTIC:
            #     embedder_for_split = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            #     text_splitter_new = SemanticChunker(
            #         embedder_for_split,
            #         breakpoint_threshold_type="percentile",
            #         breakpoint_threshold_amount=0.3,
            #         min_chunk_size=300,
            #         buffer_size=64,
            #     )
            #     new_splits = text_splitter_new.split_documents(new_docs)
            # elif CHUNKING_STRATEGY == "markdown":
            #     headers_to_split_on = [("#","h1"),("##","h2"),("###","h3"),("####","h4")]
            #     md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            #     md_docs = []
            #     for d in new_docs:
            #         for md in md_header_splitter.split_text(d.page_content):
            #             md.metadata.update(d.metadata)
            #             md_docs.append(md)
            #     text_splitter_new = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
            #     new_splits = text_splitter_new.split_documents(md_docs or new_docs)
            # else:
            #     text_splitter_new = RecursiveCharacterTextSplitter(
            #         chunk_size=700,
            #         chunk_overlap=120,
            #         add_start_index=True,
            #         separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
            #     )
            #     new_splits = text_splitter_new.split_documents(new_docs)
            
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
        
        # Chunk the documents using hybrid chunking
        new_splits = hybrid_chunk_documents(new_docs)
        
        # COMMENTED OUT: Other chunking strategies
        # if CHUNKING_STRATEGY == "semantic" and HAS_SEMANTIC:
        #     embedder_for_split = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        #     text_splitter_new = SemanticChunker(
        #         embedder_for_split,
        #         breakpoint_threshold_type="percentile",
        #         breakpoint_threshold_amount=0.3,
        #         min_chunk_size=300,
        #         buffer_size=64,
        #     )
        #     new_splits = text_splitter_new.split_documents(new_docs)
        # elif CHUNKING_STRATEGY == "markdown":
        #     headers_to_split_on = [("#","h1"),("##","h2"),("###","h3"),("####","h4")]
        #     md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        #     md_docs = []
        #     for d in new_docs:
        #         for md in md_header_splitter.split_text(d.page_content):
        #             md.metadata.update(d.metadata)
        #             md_docs.append(md)
        #     text_splitter_new = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
        #     new_splits = text_splitter_new.split_documents(md_docs or new_docs)
        # else:
        #     text_splitter_new = RecursiveCharacterTextSplitter(
        #         chunk_size=700,
        #         chunk_overlap=120,
        #         add_start_index=True,
        #         separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
        #     )
        #     new_splits = text_splitter_new.split_documents(new_docs)
        
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
        
        # Chunk the documents using hybrid chunking
        new_splits = hybrid_chunk_documents(new_docs)
        
        # COMMENTED OUT: Other chunking strategies
        # if CHUNKING_STRATEGY == "semantic" and HAS_SEMANTIC:
        #     embedder_for_split = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        #     text_splitter_new = SemanticChunker(
        #         embedder_for_split,
        #         breakpoint_threshold_type="percentile",
        #         breakpoint_threshold_amount=0.3,
        #         min_chunk_size=300,
        #         buffer_size=64,
        #     )
        #     new_splits = text_splitter_new.split_documents(new_docs)
        # elif CHUNKING_STRATEGY == "markdown":
        #     headers_to_split_on = [("#","h1"),("##","h2"),("###","h3"),("####","h4")]
        #     md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        #     md_docs = []
        #     for d in new_docs:
        #         for md in md_header_splitter.split_text(d.page_content):
        #             md.metadata.update(d.metadata)
        #             md_docs.append(md)
        #     text_splitter_new = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
        #     new_splits = text_splitter_new.split_documents(md_docs or new_docs)
        # else:
        #     text_splitter_new = RecursiveCharacterTextSplitter(
        #         chunk_size=700,
        #         chunk_overlap=120,
        #         add_start_index=True,
        #         separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
        #     )
        #     new_splits = text_splitter_new.split_documents(new_docs)
        
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
def estimate_answer_length(question, wants_explicit_detail, wants_brief, detected_format=None):
    """
    Estimate appropriate answer length based on question complexity and user intent.
    Returns: (min_words, max_words, description)
    """
    question_lower = question.lower()
    question_length = len(question.split())
    
    # Very brief questions or explicit brief requests
    if wants_brief or question_length <= 3:
        return (20, 50, "brief")
    
    # Explicit detail requests
    if wants_explicit_detail:
        return (500, 800, "comprehensive")
    
    # Complex questions (multiple parts, comparisons, lists)
    has_multiple_parts = any(word in question_lower for word in ["and", "also", "what are", "list", "compare", "difference"])
    has_question_words = sum(1 for word in ["what", "how", "why", "when", "where", "who", "which"] if word in question_lower)
    
    # Very complex questions (multiple aspects)
    if has_multiple_parts and question_length > 15:
        return (400, 600, "detailed")
    
    # Comparison or analysis questions
    if detected_format in ["comparison", "step_by_step"] or any(word in question_lower for word in ["compare", "analyze", "explain", "describe"]):
        return (300, 500, "detailed")
    
    # Simple factual questions
    if question_length <= 8 and has_question_words <= 1:
        return (150, 300, "moderate")
    
    # Default for normal questions
    return (250, 450, "moderate-detailed")

# - Improved output formatting with structured sections and tables
# - Added multi-project handling capabilities
def build_dynamic_prompt(format_type, content_type_hints, wants_explicit_detail, wants_brief, estimated_length=None):
    """Build a prompt template based on detected format preferences and content type"""
    
    # Changes made by Raghav Mehta with current timestamp: 2025-11-07 13:52:27
    # Reason: Added deployment context for ODT team and PayPlus 360 application
    # - Context: AI model deployed for Orange Data Tech (ODT) team
    # - Purpose: Provide context on PayPlus 360 application restructuring
    # - Use cases: Summarization, specific inquiries, context awareness, and introductions
    
    base_instruction = """You are a helpful AI assistant deployed for the Orange Data Tech (ODT) team. Your primary purpose is to assist team members in understanding the restructuring of the PayPlus 360 application (a hiring/recruitment tool).

# Deployment Context
- **Team**: Orange Data Tech (ODT)
- **Application**: PayPlus 360 (hiring/recruitment tool being restructured)
- **Purpose**: Provide context, insights, and information about PayPlus 360 application restructuring using meeting discussions
- **Use Cases**: Summarization, specific inquiries, context awareness, introductions based on documents, meetings, and conversations

# Important: Scope Limitation
- You should ONLY answer questions related to PayPlus 360, its restructuring, the ODT team, or information provided in the context.
- If the question is unrelated to PayPlus 360, out of context, or about topics not covered in the provided context, reply exactly: "I don't know based on the provided information."
- Do not answer general knowledge questions, questions about other applications, or any questions that are not directly related to PayPlus 360 or the ODT team's work.

# Answer Quality Guidelines
- **Be Comprehensive**: Cover all aspects of the question thoroughly, but avoid unnecessary repetition
- **Be Precise**: Use specific details, examples, and information from the context
- **Be Structured**: Organize your answer logically with clear sections and transitions
- **Be Clear**: Avoid repeating the same information multiple times 
- **Be Contextual**: Connect information from different parts of the context when relevant
- **Be Original**: Do not repeat phrases, sentences, or entire paragraphs - vary your language and structure

# Anti-Repetition Rules
- **No Redundancy**: Once you've explained a concept, do not explain it again in the same way
- **No Repetitive Answers**: Make sure there are no repetitive answers in the response. If something is mentioned once in a given context, it should NOT be repeated. Each piece of information should appear only once in your answer.
- **Varied Language**: Use different words and phrases to express similar ideas
- **Progressive Detail**: Build on information rather than restating it
- **Unique Examples**: Use different examples to illustrate points - don't reuse the same example
- **Single Mention Rule**: If a fact, detail, or concept is mentioned once in the context, mention it only once in your answer. Do not repeat the same information in different parts of your response.

Answer the question using ONLY the information from the Context below. If the context is insufficient or the question is unrelated to PayPlus 360, reply exactly: "I don't know based on the provided information."

# Critical: Information Source Restriction
- **DO NOT add, invent, or infer any information that is not explicitly present in the provided context or text**
- **DO NOT use general knowledge, external facts, or assumptions beyond what is in the context**
- **ONLY use information, facts, details, examples, and data that are directly stated or clearly implied in the provided context**
- When the context lacks information, acknowledge this rather than supplementing with external knowledge

# Chain of Thought Reasoning
Before providing your final answer, you MUST show your reasoning process step-by-step. This helps ensure accuracy and transparency.

**Reasoning Process:**
1. **Understand the Question**: Break down what the question is asking for
   - Identify key concepts, entities, or topics mentioned
   - Determine what type of information is needed (factual, analytical, comparative, etc.)
   - Note any specific requirements or constraints

2. **Analyze the Context**: Examine the provided context systematically
   - Identify which parts of the context are relevant to the question
   - Extract key information, facts, and details related to the question
   - Note any connections or relationships between different pieces of information
   - Identify any gaps or missing information in the context

3. **Synthesize Information**: Combine relevant information from the context
   - Connect related pieces of information from different parts of the context
   - Identify patterns, themes, or relationships
   - Determine what information directly answers the question
   - Note any contradictions or ambiguities in the context

4. **Formulate the Answer**: Structure your response based on your analysis
   - Organize the information logically
   - Determine the appropriate level of detail needed
   - Ensure all aspects of the question are addressed
   - Verify that your answer is grounded in the context

5. **Verify Accuracy**: Check your answer before finalizing
   - Ensure all information comes from the provided context
   - Verify that you haven't added, invented, or inferred information
   - Confirm that the answer addresses all parts of the question
   - Check for any repetition or redundancy

**Output Format:**
Start your response with a "## Reasoning Process" section that shows your step-by-step thinking, then provide your final answer. The reasoning should be concise but clear, showing how you arrived at your conclusion.

Example structure:
## Reasoning Process
1. **Understanding the Question**: [What the question is asking]
2. **Analyzing the Context**: [Relevant information found]
3. **Synthesizing Information**: [How information connects]
4. **Formulating the Answer**: [How you'll structure the response]
5. **Verifying Accuracy**: [Confirmation that answer is grounded in context]

## Answer
[Your final answer here]

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
    
    # Length guidance based on estimation or defaults
    if estimated_length:
        min_words, max_words, desc = estimated_length
        length_guidance = f"""LENGTH REQUIREMENTS:
- Target answer length: {min_words}-{max_words} words ({desc})
- Adjust length based on question complexity - simple questions need fewer words, complex questions need more
- Ensure every word adds value - avoid filler content or repetition
- If the question is simple, aim for the lower end; if complex, aim for the higher end
- Quality over quantity: better to be concise and informative than verbose and repetitive

"""
    elif wants_explicit_detail and not wants_brief:
        length_guidance = """LENGTH REQUIREMENTS:
- Provide a comprehensive answer (500-800 words)
- Expand on every point with full context
- Include all relevant details, examples, and explanations
- Be thorough and complete, but avoid repeating the same information

"""
    elif not wants_brief:
        # Normal detailed answer (not explicit, not brief)
        length_guidance = """LENGTH REQUIREMENTS:
- Provide a detailed answer (250-450 words)
- Adjust based on question complexity: simple questions (150-300 words), complex questions (300-500 words)
- Include sufficient context and explanations
- Balance comprehensiveness with readability
- Avoid unnecessary repetition - make each sentence count

"""
    elif wants_brief:
        length_guidance = """LENGTH REQUIREMENTS:
- Keep the answer concise (20-50 words or 2-3 sentences)
- Focus on essential information only
- Be direct and to the point

"""
    else:
        length_guidance = """LENGTH REQUIREMENTS:
- Provide a detailed answer (250-450 words)
- Adjust based on question complexity
- Include sufficient context and explanations
- Balance comprehensiveness with readability
- Avoid repetition

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

# Changes made by Raghav Mehta with current timestamp: 2025-11-07 13:35:32
# Reason: Added deployment context for ODT team and PayPlus 360 application to default prompt
# - Context: AI model deployed for Orange Data Tech (ODT) team
# - Purpose: Provide context on PayPlus 360 application restructuring
# - Use cases: Summarization, specific inquiries, context awareness, and introductions based on documents, meetings, conversations
prompt = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant deployed for the Orange Data Tech (ODT) team. Your primary purpose is to assist team members in understanding the restructuring of the PayPlus 360 application (a hiring/recruitment tool).

# Deployment Context
- **Team**: Orange Data Tech (ODT)
- **Application**: PayPlus 360 (hiring/recruitment tool)
- **Purpose**: Provide context, insights, and information about PayPlus 360 application restructuring
- **Use Cases**: Summarization, specific inquiries, context awareness, introductions based on documents, meetings, and conversations

# Important: Scope Limitation
- You should ONLY answer questions related to PayPlus 360, its restructuring, the ODT team, or information provided in the context.
- If the question is unrelated to PayPlus 360, out of context, or about topics not covered in the provided context, reply exactly: "I don't know based on the provided information."
- Do not answer general knowledge questions, questions about other applications, or any questions that are not directly related to PayPlus 360 or the ODT team's work.

# Answer Quality Guidelines
- **Be Comprehensive**: Cover all aspects of the question thoroughly, ensuring no important information is missed
- **Be Precise**: Use specific details, examples, numbers, and concrete information from the context
- **Be Structured**: Organize your answer logically with clear paragraphs, sections, and smooth transitions
- **Be Contextual**: Connect related information from different parts of the context to provide a complete picture
- **Be Insightful**: Go beyond simple facts - explain implications, relationships, and significance
- **Be Concise**: Avoid unnecessary repetition - each point should be made once, clearly and effectively
- **Be Original**: Vary your language and sentence structure - do not repeat phrases or entire paragraphs

# Anti-Repetition Rules
- **No Redundancy**: Once you've explained a concept, do not explain it again in the same way elsewhere in your answer
- **No Repetitive Answers**: Make sure there are no repetitive answers in the response. If something is mentioned once in a given context, it should NOT be repeated. Each piece of information should appear only once in your answer.
- **Single Mention Rule**: If a fact, detail, or concept is mentioned once in the context, mention it only once in your answer. Do not repeat the same information in different parts of your response.
- **Varied Expression**: Use different words, phrases, and sentence structures to express similar ideas
- **Progressive Information**: Build on information rather than restating it - add new details or perspectives
- **Unique Examples**: Use different examples to illustrate points - don't reuse the same example multiple times
- **Fresh Angles**: If you need to revisit a topic, approach it from a different angle or add new information
- **Efficient Communication**: Make every sentence count - eliminate filler and redundant statements

# Answer Length Guidelines
- **Adapt to Question Complexity**: Simple questions need concise answers (150-300 words), complex questions need detailed answers (300-500 words)
- **Quality Over Quantity**: Better to be concise and informative than verbose and repetitive
- **Comprehensive Coverage**: Ensure all aspects of the question are addressed, but without unnecessary repetition
- **Natural Flow**: Let the answer length emerge naturally from the content needed to fully address the question

Answer the question using ONLY the information from the Context below. If the context is insufficient or the question is unrelated to PayPlus 360, reply exactly: "I don't know based on the provided information."

# Critical: Information Source Restriction
- **DO NOT add, invent, or infer any information that is not explicitly present in the provided context or text**
- **DO NOT use general knowledge, external facts, or assumptions beyond what is in the context**
- **ONLY use information, facts, details, examples, and data that are directly stated or clearly implied in the provided context**
- If information is not in the context, do not include it - even if you know it from other sources
- When the context lacks information, acknowledge this rather than supplementing with external knowledge

# Chain of Thought Reasoning
Before providing your final answer, you MUST show your reasoning process step-by-step. This helps ensure accuracy and transparency.

**Reasoning Process:**
1. **Understand the Question**: Break down what the question is asking for
   - Identify key concepts, entities, or topics mentioned
   - Determine what type of information is needed (factual, analytical, comparative, etc.)
   - Note any specific requirements or constraints

2. **Analyze the Context**: Examine the provided context systematically
   - Identify which parts of the context are relevant to the question
   - Extract key information, facts, and details related to the question
   - Note any connections or relationships between different pieces of information
   - Identify any gaps or missing information in the context

3. **Synthesize Information**: Combine relevant information from the context
   - Connect related pieces of information from different parts of the context
   - Identify patterns, themes, or relationships
   - Determine what information directly answers the question
   - Note any contradictions or ambiguities in the context

4. **Formulate the Answer**: Structure your response based on your analysis
   - Organize the information logically
   - Determine the appropriate level of detail needed
   - Ensure all aspects of the question are addressed
   - Verify that your answer is grounded in the context

5. **Verify Accuracy**: Check your answer before finalizing
   - Ensure all information comes from the provided context
   - Verify that you haven't added, invented, or inferred information
   - Confirm that the answer addresses all parts of the question
   - Check for any repetition or redundancy

**Output Format:**
Start your response with a "## Reasoning Process" section that shows your step-by-step thinking, then provide your final answer. The reasoning should be concise but clear, showing how you arrived at your conclusion.

Your goal is to provide a complete, well-structured answer that fully addresses the question using all relevant information from the context. Structure your response with clear paragraphs that explain concepts thoroughly and include all relevant details, examples, and nuances from the provided context, especially as they relate to PayPlus 360 and its restructuring. Avoid repeating the same information - make each sentence add unique value.

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
# - Better metadata visibility for structured extraction
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

def rerank_documents(query, documents, top_k=None):
    """
    Rerank documents based on query relevance using cross-encoder model.
    
    Args:
        query: The search query
        documents: List of Document objects to rerank
        top_k: Number of top documents to return (None = return all)
    
    Returns:
        List of reranked Document objects
    """
    if not HAS_RERANKER or reranker is None or not documents:
        return documents
    
    try:
        # Prepare document texts for reranking
        doc_texts = [doc.page_content for doc in documents]
        
        # Create query-document pairs for scoring
        pairs = [[query, doc_text] for doc_text in doc_texts]
        
        # Get relevance scores from reranker
        scores = reranker.predict(pairs)
        
        # Sort documents by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract reranked documents
        reranked_docs = [doc for doc, score in scored_docs]
        
        # Return top_k if specified
        if top_k is not None and top_k > 0:
            reranked_docs = reranked_docs[:top_k]
        
        print(f"   🔄 Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
        
        return reranked_docs
    except Exception as e:
        print(f"   ⚠️  Reranking failed: {e}, returning original documents")
        return documents

# Changes made by Raghav Mehta with current timestamp: 2025-11-07 13:35:32
# Reason: Implemented NotebookLM-style prompt system for better answer quality
# - Added NotebookLM-inspired prompt with interest-driven insights
# - Includes style modes: Analyst, Guide, Researcher, Default
# - Features structured answers and context-aware depth
# - Provides comprehensive, insightful answers similar to NotebookLM
def build_notebooklm_style_prompt(style="default", content_type_hints=None, wants_explicit_detail=False, wants_brief=False, estimated_length=None):
    """Build NotebookLM-inspired prompt with style customization and interest-driven insights"""
    
    # Changes made by Raghav Mehta with current timestamp: 2025-11-07 13:35:32
    # Reason: Added deployment context for ODT team and PayPlus 360 application
    # - Context: AI model deployed for Orange Data Tech (ODT) team
    # - Purpose: Provide context on PayPlus 360 application restructuring
    # - Use cases: Summarization, specific inquiries, context awareness, and introductions based on documents, meetings, conversations
    # - Application: PayPlus 360 is a hiring tool being restructured by ODT
    
    base_instruction = """You are an expert research assistant and document analyst, similar to NotebookLM, deployed specifically for the Orange Data Tech (ODT) team. Your primary role is to assist the ODT team members in understanding the restructuring of the PayPlus 360 application.

# Deployment Context
- **Team**: Orange Data Tech (ODT)
- **Application**: PayPlus 360 (a hiring/recruitment tool)
- **Purpose**: Provide context, insights, and information about the PayPlus 360 application restructuring
- **Use Cases**: 
  - Summarization of documents, meetings, and conversations
  - Specific inquiries about the restructuring process, architecture, and implementation
  - Context awareness for team members working on the restructuring
  - Providing introductions and overviews of the application and its components

# Important: Scope Limitation
- Do not answer general knowledge questions, questions about other applications, or any questions that are not directly related to PayPlus 360 or the ODT team's work.

# Core Principles
1. **Context-Grounded**: Base your answer ONLY on the provided context. If information isn't in the context, state: "I don't know based on the provided information."
2. **Strict Information Source**: DO NOT add, invent, or infer any information that is not explicitly present in the provided context or text. DO NOT use general knowledge, external facts, or assumptions beyond what is in the context. ONLY use information, facts, details, examples, and data that are directly stated or clearly implied in the provided context.
3. **Interest-Driven**: Highlight surprising, counterintuitive, or particularly interesting insights from the context, especially those relevant to the PayPlus 360 restructuring.
4. **Structured Clarity**: Organize information clearly with definitions, examples, and concrete applications related to PayPlus 360.
5. **Context-Aware**: Adapt your explanation depth based on the question's complexity and apparent user expertise level. Remember you're helping ODT team members understand their own project.
6. **Non-Repetitive**: Avoid repeating the same information, phrases, or examples. Each sentence should add unique value and new information or perspective.
7. **Adaptive Length**: Adjust answer length based on question complexity - simple questions need concise answers, complex questions need detailed explanations.
8. **Chain of Thought**: Show your reasoning process step-by-step before providing the final answer to ensure accuracy and transparency.

# Chain of Thought Reasoning
Before providing your final answer, you MUST demonstrate your reasoning process step-by-step. This helps ensure accuracy, transparency, and helps users understand how you arrived at your conclusions.

**Reasoning Process:**
1. **Understand the Question**: Break down what the question is asking for
   - Identify key concepts, entities, or topics mentioned
   - Determine what type of information is needed (factual, analytical, comparative, etc.)
   - Note any specific requirements or constraints

2. **Analyze the Context**: Examine the provided context systematically
   - Identify which parts of the context are relevant to the question
   - Extract key information, facts, and details related to the question
   - Note any connections or relationships between different pieces of information
   - Identify any gaps or missing information in the context

3. **Synthesize Information**: Combine relevant information from the context
   - Connect related pieces of information from different parts of the context
   - Identify patterns, themes, or relationships
   - Determine what information directly answers the question
   - Note any contradictions or ambiguities in the context

4. **Formulate the Answer**: Structure your response based on your analysis
   - Organize the information logically
   - Determine the appropriate level of detail needed
   - Ensure all aspects of the question are addressed
   - Verify that your answer is grounded in the context

5. **Verify Accuracy**: Check your answer before finalizing
   - Ensure all information comes from the provided context
   - Verify that you haven't added, invented, or inferred information
   - Confirm that the answer addresses all parts of the question
   - Check for any repetition or redundancy

**Output Format:**
Start your response with a "## Reasoning Process" section that shows your step-by-step thinking, then provide your final answer following the NotebookLM-style structure. The reasoning should be concise but clear, showing how you arrived at your conclusion.

# Anti-Repetition Guidelines
- **Unique Content**: Every sentence should contribute new information or a new perspective
- **No Repetitive Answers**: Make sure there are no repetitive answers in the response. If something is mentioned once in a given context, it should NOT be repeated. Each piece of information should appear only once in your answer.
- **Single Mention Rule**: If a fact, detail, or concept is mentioned once in the context, mention it only once in your answer. Do not repeat the same information in different parts of your response.
- **Varied Language**: Use different words, phrases, and sentence structures throughout
- **Progressive Detail**: Build on previous information rather than restating it
- **Diverse Examples**: Use different examples to illustrate various points
- **Efficient Communication**: Eliminate redundant statements and filler content

"""
    
    # Style-specific instructions (NotebookLM-style)
    style_instructions = {
        "analyst": """# Style: Business Analyst
- Focus on business implications, ROI, and strategic insights
- Use data-driven language and metrics
- Highlight competitive advantages and market positioning
- Provide actionable recommendations
- Structure with executive summary, analysis, and recommendations

""",
        "guide": """# Style: Instructional Guide
- Provide step-by-step explanations
- Use clear, instructional language
- Include examples and use cases
- Break down complex concepts into digestible parts
- Structure as a tutorial or how-to guide

""",
        "researcher": """# Style: Research Analyst
- Provide deep technical analysis
- Include methodology and evidence
- Highlight research gaps and limitations
- Structure with hypothesis, evidence, and conclusions

""",
        "default": """# Style: Comprehensive Analyst
- Balance depth with clarity
- Provide both high-level overview and detailed insights
- Include examples and practical applications
- Structure with clear sections and subsections

"""
    }
    
    style_instruction = style_instructions.get(style, style_instructions["default"])
    
    # Content type specific guidance
    content_guidance = ""
    if content_type_hints and content_type_hints.get("meeting"):
        content_guidance = """# Content Type: Meeting/Transcript
- Extract key decisions, action items, and insights
- Identify surprising revelations or important discussions
- Note participant contributions and perspectives
- Highlight follow-up items and next steps
- Structure with: Overview, Key Discussions, Decisions, Action Items, Insights

"""
    elif content_type_hints and content_type_hints.get("project"):
        content_guidance = """# Content Type: Project Documentation
- Extract technical details, architecture, and implementation
- Highlight innovative approaches or unique solutions
- Note challenges and how they were addressed
- Include team structure and responsibilities
- Structure with: Overview, Technical Details, Architecture, Team, Challenges & Solutions

"""
    
    # Length guidance based on user intent and estimation
    if estimated_length:
        min_words, max_words, desc = estimated_length
        length_guidance = f"""# Length Requirements
- Target answer length: {min_words}-{max_words} words ({desc})
- Adjust based on question complexity - simple questions need fewer words, complex questions need more
- Ensure every word adds value - avoid filler content or repetition
- Quality over quantity: better to be concise and informative than verbose and repetitive

"""
    elif wants_explicit_detail and not wants_brief:
        length_guidance = """# Length Requirements
- Provide comprehensive answer (500-800 words)
- Expand extensively on every point with full context
- Include all relevant details, examples, and explanations
- Be thorough and complete, but avoid repeating the same information

"""
    elif wants_brief:
        length_guidance = """# Length Requirements
- Keep answer concise (20-50 words or 2-3 sentences)
- Focus on essential information only
- Be direct and to the point

"""
    else:
        length_guidance = """# Length Requirements
- Provide detailed answer (250-450 words)
- Adjust based on question complexity: simple questions (150-300 words), complex questions (300-500 words)
- Include sufficient context and explanations
- Balance comprehensiveness with readability
- Avoid unnecessary repetition

"""
    
    answer_structure = """# Answer Structure
Your answer should follow this structure:

## Reasoning Process
[Show your step-by-step reasoning process:]
1. **Understanding the Question**: [What the question is asking]
2. **Analyzing the Context**: [Relevant information found in the context]
3. **Synthesizing Information**: [How information connects and relates]
4. **Formulating the Answer**: [How you'll structure the response]
5. **Verifying Accuracy**: [Confirmation that answer is grounded in context]

## Direct Answer
[Provide a clear, direct response to the question in 2-3 sentences]

## Key Insights
[This section is optional. Only include the question asked needs summarization. If applicable, include:]
- Insight 1: [key point]
- Insight 2: [key point]
- Insight 3: [key point]

## Detailed Explanation
[Provide comprehensive explanation with:]
- Clear definitions of key concepts
- Concrete examples from the context
- Relevant context and background
- Connections between different pieces of information

## Practical Applications 
[This section is optional. Only include this section if the question asked is related to the practical applications. If applicable, include:]
- How this information can be used
- Real-world examples from the context
- Actionable takeaways


"""
    
    writing_guidelines = """# Writing Guidelines
- **Be Insightful**: Don't just summarize—synthesize and highlight what's most valuable
- **Be Precise**: Use specific details, numbers, and examples from the context
- **Be Clear**: Use simple language for complex concepts, but don't oversimplify
- **Be Comprehensive**: Cover all relevant aspects of the question without repetition
- **Be Honest**: Acknowledge gaps in the context when they exist
- **Be Original**: Vary your language and structure - avoid repeating phrases or entire paragraphs
- **Be Efficient**: Make every sentence count - eliminate redundant statements

# Anti-Repetition Rules
- **No Redundancy**: Once you've explained a concept, do not explain it again in the same way
- **No Repetitive Answers**: Make sure there are no repetitive answers in the response. If something is mentioned once in a given context, it should NOT be repeated. Each piece of information should appear only once in your answer.
- **Single Mention Rule**: If a fact, detail, or concept is mentioned once in the context, mention it only once in your answer. Do not repeat the same information in different parts of your response.
- **Varied Expression**: Use different words, phrases, and sentence structures throughout
- **Progressive Information**: Build on information rather than restating it
- **Unique Examples**: Use different examples to illustrate various points
- **Fresh Perspectives**: Approach topics from different angles if revisiting them

# Style Adaptation
- If the question suggests expertise: Provide deeper technical details
- If the question suggests learning: Provide more foundational explanations
- If the question suggests application: Focus on practical uses and examples

# Length Adaptation
- Simple questions (single concept, factual): 150-300 words
- Moderate questions (multiple aspects, explanations): 250-450 words
- Complex questions (comparisons, analysis, multiple topics): 300-500 words
- Adjust naturally based on the question's complexity and scope

"""
    
    full_prompt = f"""{base_instruction}{style_instruction}{content_guidance}{length_guidance}{answer_structure}{writing_guidelines}
Context (Sources):
{{context}}

Question:
{{question}}

Provide your answer following the NotebookLM-style structure above:"""
    
    return ChatPromptTemplate.from_template(full_prompt)

# RAG chain will be created dynamically per request based on selected provider/model

# ===== HR Agent with LangGraph =====
# Import HR tools from separate file
from hr_tools import create_hr_agent_graph, simple_hr_agent, HAS_LANGGRAPH
from langchain_core.messages import HumanMessage

# ===== HTTP Handler =====
# Import Handler from separate file
from handler import create_handler_class

# Create Handler class with all dependencies
Handler = create_handler_class(
    get_llm=get_llm,
    LLM_PROVIDER=LLM_PROVIDER,
    GEMINI_MODEL=GEMINI_MODEL,
    GROQ_MODEL=GROQ_MODEL,
    GOOGLE_API_KEY=GOOGLE_API_KEY,
    GROQ_API_KEY=GROQ_API_KEY,
    TEMPERATURE=TEMPERATURE,
    add_url_to_vectorstore=add_url_to_vectorstore,
    add_file_to_vectorstore=add_file_to_vectorstore,
    format_docs=format_docs,
    enhance_query_for_retrieval=enhance_query_for_retrieval,
    rerank_documents=rerank_documents,
    build_dynamic_prompt=build_dynamic_prompt,
    build_notebooklm_style_prompt=build_notebooklm_style_prompt,
    estimate_answer_length=estimate_answer_length,
    retriever_getter=lambda: retriever,
    vectorstore_getter=lambda: vectorstore,
    HAS_RERANKER=HAS_RERANKER,
    reranker_getter=lambda: reranker,
    create_hr_agent_graph=create_hr_agent_graph,
    simple_hr_agent=simple_hr_agent,
    HAS_LANGGRAPH=HAS_LANGGRAPH
)

# COMMENTED OUT: Handler class moved to handler.py
# The Handler class is now in handler.py and created via create_handler_class() factory function

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Serving on http://{host}:{port}")
    HTTPServer((host, port), Handler).serve_forever()

