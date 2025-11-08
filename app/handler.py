"""
HTTP Request Handler
Handles all HTTP requests for the RAG application.
"""

import os
import json
import cgi
import uuid
import time
from http.server import SimpleHTTPRequestHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Session memory storage: {session_id: [{"question": str, "answer": str, "timestamp": float}, ...]}
# Sessions expire after 1 hour of inactivity
SESSION_STORAGE = {}
SESSION_TIMEOUT = 3600  # 1 hour in seconds

def get_or_create_session(session_id=None):
    """Get existing session or create new one"""
    if session_id and session_id in SESSION_STORAGE:
        # Check if session expired
        if SESSION_STORAGE[session_id]:
            last_timestamp = SESSION_STORAGE[session_id][-1].get("timestamp", 0)
            if time.time() - last_timestamp < SESSION_TIMEOUT:
                return session_id
            else:
                # Session expired, remove it
                del SESSION_STORAGE[session_id]
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    SESSION_STORAGE[new_session_id] = []
    return new_session_id

def add_to_session(session_id, question, answer):
    """Add question-answer pair to session history"""
    if session_id not in SESSION_STORAGE:
        SESSION_STORAGE[session_id] = []
    SESSION_STORAGE[session_id].append({
        "question": question,
        "answer": answer,
        "timestamp": time.time()
    })
    # Keep only last 20 exchanges to prevent memory bloat
    if len(SESSION_STORAGE[session_id]) > 20:
        SESSION_STORAGE[session_id] = SESSION_STORAGE[session_id][-20:]

def get_session_history(session_id, max_exchanges=10):
    """Get conversation history for a session"""
    if session_id not in SESSION_STORAGE:
        return []
    history = SESSION_STORAGE[session_id]
    # Return last N exchanges
    return history[-max_exchanges:] if len(history) > max_exchanges else history

def clear_session(session_id):
    """Clear a session's history"""
    if session_id in SESSION_STORAGE:
        SESSION_STORAGE[session_id] = []

def is_follow_up_question(question, conversation_history):
    """Detect if current question is a follow-up to previous conversation"""
    if not conversation_history:
        return False
    
    question_lower = question.lower().strip()
    follow_up_patterns = [
        "tell me more", "more about", "explain more", "more details",
        "what about", "how about", "what is", "how does", "why does",
        "can you", "could you", "please explain", "elaborate",
        "that", "this", "it", "they", "those", "these",
        "also", "and", "what else", "anything else",
        "in relation to", "related to", "regarding", "concerning"
    ]
    
    # Check for follow-up indicators
    has_follow_up_word = any(pattern in question_lower for pattern in follow_up_patterns)
    
    # Check if question is very short (likely a follow-up)
    is_short = len(question.split()) <= 5
    
    # Check if question references previous answer (pronouns, "that", etc.)
    has_reference = any(word in question_lower for word in ["that", "this", "it", "they", "those", "these", "the above", "mentioned"])
    
    return has_follow_up_word or (is_short and has_reference) or has_reference

def format_conversation_history(history, is_follow_up=False):
    """Format conversation history for inclusion in prompt"""
    if not history:
        return ""
    
    formatted = "\n# Previous Conversation History\n"
    
    if is_follow_up:
        formatted += "**IMPORTANT: This is a follow-up question. The user is asking for more information or clarification about something discussed earlier.**\n"
        formatted += "The following conversation history contains the context you need to answer this follow-up question:\n\n"
    else:
        formatted += "The following is the conversation history for context. Use this to understand what was discussed earlier:\n\n"
    
    # Include full answers (no truncation) for better follow-up context
    for i, exchange in enumerate(history, 1):
        formatted += f"**Exchange {i}:**\n"
        formatted += f"Question: {exchange['question']}\n"
        formatted += f"Answer: {exchange['answer']}\n"  # No truncation - full answer for context
        formatted += "\n"
    
    formatted += "---\n"
    
    if is_follow_up:
        formatted += "**CRITICAL FOR FOLLOW-UP QUESTIONS:**\n"
        formatted += "- This question is a follow-up to the conversation above\n"
        formatted += "- Use the previous answers as PRIMARY context - they contain the information the user is asking about\n"
        formatted += "- Reference specific details, examples, or points from the previous answers\n"
        formatted += "- Build on and expand the information already provided\n"
        formatted += "- If the question is unclear, infer what the user is asking about based on the conversation history\n"
        formatted += "- The document context below provides additional supporting information, but prioritize the conversation history\n\n"
    else:
        formatted += "Now answer the current question below, using both the context provided and the conversation history above for reference.\n\n"
    
    return formatted

def is_last_question_request(question, conversation_history):
    """Check if user is asking about their last question"""
    if not conversation_history:
        return False
    
    question_lower = question.lower().strip()
    last_question_patterns = [
        "what was my last question",
        "what was the last question",
        "repeat my last question",
        "repeat the last question",
        "show me my last question",
        "tell me my last question",
        "what did i ask",
        "what did i ask last",
        "my previous question",
        "the previous question",
        "last question i asked",
        "answer my last question",
        "answer the last question",
        "answer my previous question",
        "answer the previous question",
        "re-answer my last question",
        "re-answer the last question"
    ]
    
    return any(pattern in question_lower for pattern in last_question_patterns)

def get_last_question(conversation_history):
    """Get the last question from conversation history"""
    if not conversation_history:
        return None
    return conversation_history[-1].get("question")

def detect_document_reference(question, vectorstore):
    """Detect if user is asking about a specific document and return the source name"""
    import re
    question_lower = question.lower()
    
    # Patterns that indicate document reference (in order of specificity)
    reference_patterns = [
        # Most specific: "from document X" or "in document X"
        (r"from\s+(?:the\s+)?(?:document|doc|file|pdf|page|source|url)\s+(?:called|named|titled)?\s*['\"]?([^'\"\?]+)['\"]?", 1),
        (r"in\s+(?:the\s+)?(?:document|doc|file|pdf|page|source|url)\s+(?:called|named|titled)?\s*['\"]?([^'\"\?]+)['\"]?", 1),
        # "document X says" or "file X mentions"
        (r"(?:document|doc|file|pdf|page|source|url)\s+(?:called|named|titled)?\s*['\"]?([^'\"\?]+)['\"]?\s+(?:says|mentions|states|contains)", 1),
        # Filename with extension in quotes
        (r"['\"]([^'\"]+\.(?:pdf|docx?|txt|html|md))['\"]", 1),
        # URLs
        (r"https?://[^\s\?]+", 0),
        # "from X" or "in X" (less specific, check last)
        (r"from\s+['\"]?([^'\"\?\s]{3,})['\"]?", 1),
        (r"in\s+['\"]?([^'\"\?\s]{3,})['\"]?", 1),
    ]
    
    for pattern, group_idx in reference_patterns:
        try:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                if group_idx == 0:
                    doc_ref = match.group(0).strip()  # For URL pattern (full match)
                else:
                    # Check if the group exists
                    if match.lastindex and match.lastindex >= group_idx:
                        doc_ref = match.group(group_idx).strip()
                    else:
                        continue  # Skip this pattern if group doesn't exist
                
                # Clean up the reference (remove trailing punctuation, etc.)
                doc_ref = doc_ref.rstrip('.,!?;:')
                
                if doc_ref and len(doc_ref) > 2:  # Valid reference
                    return doc_ref
        except (IndexError, AttributeError) as e:
            # Skip patterns that cause errors
            continue
    
    return None

def filter_documents_by_source(docs, source_reference):
    """Filter documents to only include chunks from the specified source"""
    if not source_reference or not docs:
        return docs
    
    import re
    source_ref_lower = source_reference.lower().strip()
    filtered_docs = []
    
    # Extract key terms from reference (remove common words, get filename/domain)
    # For URLs, extract domain; for filenames, extract name without extension
    if source_ref_lower.startswith(('http://', 'https://')):
        # URL: extract domain or full URL
        key_terms = [source_ref_lower]
        # Also try domain extraction
        domain_match = re.search(r'https?://([^/]+)', source_ref_lower)
        if domain_match:
            key_terms.append(domain_match.group(1))
    else:
        # Filename: extract name parts
        key_terms = [source_ref_lower]
        # Remove extension for matching
        name_without_ext = re.sub(r'\.[^.]+$', '', source_ref_lower)
        if name_without_ext:
            key_terms.append(name_without_ext)
        # Extract words from the reference
        words = re.findall(r'\b\w{3,}\b', source_ref_lower)
        key_terms.extend(words)
    
    for doc in docs:
        source = str(doc.metadata.get("source", "")).lower()
        title = str(doc.metadata.get("title", "")).lower()
        all_metadata = str(doc.metadata).lower()
        
        # Check if any key term matches
        matches = False
        for term in key_terms:
            if (term in source or source in term or
                term in title or title in term or
                term in all_metadata):
                matches = True
                break
        
        if matches:
            filtered_docs.append(doc)
    
    # If we found matching documents, return them; otherwise return all (maybe reference wasn't exact)
    if filtered_docs:
        print(f"   📄 Filtered to {len(filtered_docs)} chunks from referenced document: {source_reference[:50]}")
        return filtered_docs
    
    print(f"   ⚠️  No exact match for document reference '{source_reference[:50]}', using all retrieved documents")
    return docs  # Return all if no match found

def create_handler_class(
    get_llm, LLM_PROVIDER, GEMINI_MODEL, GROQ_MODEL,
    GOOGLE_API_KEY, GROQ_API_KEY, TEMPERATURE,
    add_url_to_vectorstore, add_file_to_vectorstore, format_docs,
    enhance_query_for_retrieval, rerank_documents,
    build_dynamic_prompt, build_notebooklm_style_prompt, estimate_answer_length,
    retriever_getter, vectorstore_getter, HAS_RERANKER, reranker_getter,
    create_hr_agent_graph, simple_hr_agent, HAS_LANGGRAPH
):
    """Factory function to create Handler class with all dependencies"""

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
                            graph = create_hr_agent_graph(service, get_llm, LLM_PROVIDER)
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
                                response_text = simple_hr_agent(service, message, get_llm, LLM_PROVIDER)
                        except Exception as e:
                            print(f"LangGraph error: {e}")
                            response_text = simple_hr_agent(service, message, get_llm, LLM_PROVIDER)
                    else:
                        response_text = simple_hr_agent(service, message, get_llm, LLM_PROVIDER)
                
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
            
                # Session memory: Get or create session
                session_id = get_or_create_session(data.get("session_id"))
                # Get initial history to detect follow-up
                initial_history = get_session_history(session_id, max_exchanges=10)
                
                # Detect if this is a follow-up question
                is_follow_up = is_follow_up_question(q, initial_history)
                
                # For follow-up questions, get more conversation history for better context
                if is_follow_up:
                    conversation_history = get_session_history(session_id, max_exchanges=15)  # More context for follow-ups
                    print(f"   🔄 Detected follow-up question - using extended history")
                else:
                    conversation_history = initial_history
                
                print(f"   💬 Session: {session_id[:8]}... ({len(conversation_history)} previous exchanges)")
            
                # Check if user is asking about their last question
                if is_last_question_request(q, conversation_history):
                    last_question = get_last_question(conversation_history)
                    if last_question:
                        print(f"   🔄 User requested last question: '{last_question}'")
                        # Check if they want to re-answer it or just see it
                        q_lower = q.lower()
                        wants_reanswer = any(phrase in q_lower for phrase in [
                            "answer", "re-answer", "reanswer", "respond to"
                        ])
                        
                        if wants_reanswer:
                            # Re-answer the last question
                            print(f"   ✅ Re-answering last question")
                            q = last_question  # Use the last question as the current question
                        else:
                            # Just show the last question
                            payload = {
                                "answer": f"Your last question was: \"{last_question}\"\n\nWould you like me to answer it again?",
                                "sources": [],
                                "markdown": f"### Answer:\nYour last question was: **\"{last_question}\"**\n\nWould you like me to answer it again?",
                                "needs_url": False,
                                "session_id": session_id
                            }
                            blob = json.dumps(payload).encode("utf-8")
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", str(len(blob)))
                            self.send_header("Access-Control-Allow-Origin", "*")
                            self.end_headers()
                            self.wfile.write(blob)
                            return
                    else:
                        # No previous questions
                        payload = {
                            "answer": "You haven't asked any questions yet in this session.",
                            "sources": [],
                            "markdown": "### Answer:\nYou haven't asked any questions yet in this session.",
                            "needs_url": False,
                            "session_id": session_id
                        }
                        blob = json.dumps(payload).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(blob)))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(blob)
                        return
            
                # Get provider and model from request (default to environment settings)
                provider = data.get("provider") or LLM_PROVIDER
                model = data.get("model")
            
                # Changes made by Raghav Mehta with current timestamp: 2025-11-07 12:12:09
                # Reason: Updated retriever initialization to use MMR for better diversity
                # - Uses MMR retriever by default for better chunk diversity
                # - Falls back to similarity search if MMR not supported
                # Ensure we have the latest retriever (in case it was updated)
                retriever = retriever_getter()
                vectorstore = vectorstore_getter()
                if retriever is None:
                    try:
                        retriever = vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={"k": 20, "fetch_k": 30, "lambda_mult": 0.5}
                        )
                    except Exception:
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
            
                # ===== DOCUMENT REFERENCE DETECTION =====
                # Check if user is asking about a specific document
                referenced_doc = detect_document_reference(q, vectorstore)
                if referenced_doc:
                    print(f"   📄 User referenced specific document: {referenced_doc[:60]}")
            
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
                # For follow-up questions, add context from previous conversation
                if is_follow_up and conversation_history:
                    # Extract key terms from recent answers to enhance the query
                    last_answer = conversation_history[-1].get("answer", "")
                    # Add relevant terms from last answer to query
                    enhanced_q = q
                    # Extract important nouns/phrases from last answer (simple approach)
                    last_answer_words = last_answer.split()[:50]  # First 50 words
                    # Add context terms to query
                    context_terms = " ".join([w for w in last_answer_words if len(w) > 4])[:200]  # Terms longer than 4 chars
                    enhanced_q = f"{q} {context_terms}"
                    enhanced_query = enhance_query_for_retrieval(enhanced_q, content_type_hints_preview)
                    print(f"   🔍 Enhanced follow-up query with conversation context")
                else:
                    enhanced_query = enhance_query_for_retrieval(q, content_type_hints_preview)
            
                # Retrieve documents with enhanced query (fetch more for reranking)
                # Fetch more documents initially, then rerank to get the best ones
                RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))  # Final number of docs after reranking
                RETRIEVE_K = int(os.getenv("RETRIEVE_K", "40"))  # Fetch more for reranking
            
                # Temporarily increase k for retrieval if reranking is enabled
                reranker = reranker_getter() if HAS_RERANKER else None
                if HAS_RERANKER and reranker is not None:
                    # Get more candidates for reranking
                    docs = retriever.get_relevant_documents(enhanced_query)
                    # If retriever returned fewer than RETRIEVE_K, try to get more
                    if len(docs) < RETRIEVE_K:
                        # Try to get more documents by adjusting retriever temporarily
                        try:
                            vectorstore = vectorstore_getter()
                            temp_retriever = vectorstore.as_retriever(
                                search_kwargs={"k": RETRIEVE_K}
                            )
                            docs = temp_retriever.get_relevant_documents(enhanced_query)
                        except:
                            pass  # Use what we got
                    
                    # Rerank documents
                    docs = rerank_documents(q, docs, top_k=RERANK_TOP_K)
                else:
                    # No reranking, use normal retrieval
                    docs = retriever.get_relevant_documents(enhanced_query)
                
                # Filter documents if user referenced a specific document
                if referenced_doc:
                    docs = filter_documents_by_source(docs, referenced_doc)
                    # If filtering removed all docs, try again with more docs
                    if not docs:
                        print(f"   ⚠️  No chunks found for referenced document, trying broader search")
                        try:
                            temp_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
                            docs = temp_retriever.get_relevant_documents(enhanced_query)
                            docs = filter_documents_by_source(docs, referenced_doc)
                        except:
                            pass
                
                if not docs:
                    # No context found - return early
                    payload = {
                        "answer": "I don't know based on the provided information. No relevant documents were found in the knowledge base.",
                        "sources": [],
                        "markdown": "### Answer:\nI don't know based on the provided information. No relevant documents were found in the knowledge base.",
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
            
                # Changes made by Raghav Mehta with current timestamp: 2025-11-07 13:35:32
                # Reason: Integrated NotebookLM-style prompts as default for better answer quality
                # - Uses NotebookLM-style prompts for comprehensive, insightful answers
                # - Falls back to format-specific prompts when format is explicitly requested
                # - Provides interest-driven insights and context-aware responses
                # Estimate answer length based on question complexity
                estimated_length = estimate_answer_length(q, wants_explicit_detail, wants_brief, detected_format)
                min_words, max_words, desc = estimated_length
                print(f"   📏 Estimated answer length: {min_words}-{max_words} words ({desc})")
                
                # ===== DYNAMIC PROMPT SELECTION =====
                # Use NotebookLM-style prompts by default for better quality
                # Use format-specific prompts only when format is explicitly requested
                if detected_format:
                    # Use dynamic prompt builder for detected format (table, list, etc.)
                    enhanced_prompt = build_dynamic_prompt(detected_format, content_type_hints, wants_explicit_detail, wants_brief, estimated_length)
                    print(f"   📝 Using dynamic prompt for format: {detected_format}")
                elif wants_explicit_detail and not wants_brief:
                    # Use the aggressive 500+ words prompt ONLY when explicitly requested
                    enhanced_prompt = ChatPromptTemplate.from_template(
                        """You are a helpful AI assistant deployed for the Orange Data Tech (ODT) team. Your primary purpose is to assist team members in understanding the restructuring of the PayPlus 360 application (a hiring/recruitment tool).

    # Deployment Context
    - **Team**: Orange Data Tech (ODT)
    - **Application**: PayPlus 360 (hiring/recruitment tool being restructured)
    - **Purpose**: Provide context, insights, and information about PayPlus 360 application restructuring
    - **Use Cases**: Summarization, specific inquiries, context awareness, introductions based on documents, meetings, and conversations

    # Important: Scope Limitation
    - You should ONLY answer questions related to PayPlus 360, its restructuring, the ODT team, or information provided in the context.
    - If the question is unrelated to PayPlus 360, out of context, or about topics not covered in the provided context, reply exactly: "I don't know based on the provided information."
    - Do not answer general knowledge questions, questions about other applications, or any questions that are not directly related to PayPlus 360 or the ODT team's work.

    # Answer Quality Guidelines
    - **Be Comprehensive**: Cover all aspects of the question thoroughly, but avoid unnecessary repetition
    - **Be Precise**: Use specific details, examples, and information from the context
    - **Be Structured**: Organize your answer logically with clear sections and transitions
    - **Be Concise**: Avoid repeating the same information multiple times - each point should be made once, clearly
    - **Be Contextual**: Connect information from different parts of the context when relevant
    - **Be Original**: Do not repeat phrases, sentences, or entire paragraphs - vary your language and structure

    # Anti-Repetition Rules
    - **No Redundancy**: Once you've explained a concept, do not explain it again in the same way
    - **No Repetitive Answers**: Make sure there are no repetitive answers in the response. If something is mentioned once in a given context, it should NOT be repeated. Each piece of information should appear only once in your answer.
    - **Single Mention Rule**: If a fact, detail, or concept is mentioned once in the context, mention it only once in your answer. Do not repeat the same information in different parts of your response.
    - **Varied Language**: Use different words and phrases to express similar ideas
    - **Progressive Detail**: Build on information rather than restating it
    - **Unique Examples**: Use different examples to illustrate points - don't reuse the same example
    - **Fresh Perspectives**: Approach the same topic from different angles if you need to revisit it

    Answer the question using ONLY the information from the Context below. If the context is insufficient or the question is unrelated to PayPlus 360, reply exactly: "I don't know based on the provided information."

    # Critical: Information Source Restriction
    - **DO NOT add, invent, or infer any information that is not explicitly present in the provided context or text**
    - **DO NOT use general knowledge, external facts, or assumptions beyond what is in the context**
    - **ONLY use information, facts, details, examples, and data that are directly stated or clearly implied in the provided context**
    - If information is not in the context, do not include it - even if you know it from other sources
    - When the context lacks information, acknowledge this rather than supplementing with external knowledge

    🚨🚨🚨 CRITICAL: The user explicitly requested DETAILED/EXPANDED information. Your response MUST be at least 500 words. This is NOT optional - it is a MANDATORY REQUIREMENT.

    MANDATORY REQUIREMENTS:
    1. Your answer MUST be 500-800 words - adjust based on question complexity
    2. Write ONLY in full paragraphs (4-6 sentences each) - NO bullet points, NO lists, NO short phrases
    3. Expand extensively on EVERY single point, topic, and detail from the context
    4. Include ALL relevant details, context, background, nuances, implications, and examples
    5. Avoid repetition - make each sentence add unique value

    WRITING STYLE:
    - Use well-structured paragraphs with proper transitions
    - Each paragraph should be 4-6 sentences explaining one topic in depth
    - Connect ideas with transition words (Furthermore, Additionally, Moreover, etc.)
    - Include specific examples, quotes, and detailed explanations from the context
    - Do NOT summarize - EXPAND and ELABORATE on everything
    - Vary your language - do not repeat the same phrases or sentences

    Your goal is to provide a complete, extensive, well-structured answer that fully addresses the question using ALL relevant information from the context. Every topic mentioned should be explained in detail with full context. Avoid repeating the same information - make each sentence add unique value.

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
                        """You are a helpful AI assistant deployed for the Orange Data Tech (ODT) team. Your primary purpose is to assist team members in understanding the restructuring of the PayPlus 360 application (a hiring/recruitment tool).

    # Deployment Context
    - **Team**: Orange Data Tech (ODT)
    - **Application**: PayPlus 360 (hiring/recruitment tool being restructured)
    - **Purpose**: Provide context, insights, and information about PayPlus 360 application restructuring

    # Important: Scope Limitation
    - You should ONLY answer questions related to PayPlus 360, its restructuring, the ODT team, or information provided in the context.
    - If the question is unrelated to PayPlus 360, out of context, or about topics not covered in the provided context, reply exactly: "I don't know based on the provided information."
    - Do not answer general knowledge questions, questions about other applications, or any questions that are not directly related to PayPlus 360 or the ODT team's work.

    # Answer Quality Guidelines
    - **Be Precise**: Use specific details from the context
    - **Be Concise**: Provide only essential information
    - **Be Original**: Avoid repetition - make each word count
    - **No Repetitive Answers**: Make sure there are no repetitive answers in the response. If something is mentioned once in a given context, it should NOT be repeated. Each piece of information should appear only once in your answer.

    Answer the question using ONLY the information from the Context below. If the context is insufficient or the question is unrelated to PayPlus 360, reply exactly: "I don't know based on the provided information."

    # Critical: Information Source Restriction
    - **DO NOT add, invent, or infer any information that is not explicitly present in the provided context or text**
    - **DO NOT use general knowledge, external facts, or assumptions beyond what is in the context**
    - **ONLY use information, facts, details, examples, and data that are directly stated or clearly implied in the provided context**
    - If information is not in the context, do not include it - even if you know it from other sources

    Provide a brief, concise answer (20-50 words or 2-3 sentences maximum). Avoid repetition. If something is mentioned once, do not repeat it.

    Context:
    {context}

    Question:
    {question}"""
                    )
                    print(f"   📝 Using brief prompt")
                else:
                    # Use NotebookLM-style prompt by default for comprehensive, insightful answers
                    # Detect style from question (analyst, guide, researcher, or default)
                    style = "default"
                    question_lower_style = q.lower()
                    if any(word in question_lower_style for word in ["analyze", "analysis", "business", "strategy", "roi", "market"]):
                        style = "analyst"
                    elif any(word in question_lower_style for word in ["how to", "guide", "tutorial", "steps", "instructions", "explain"]):
                        style = "guide"
                    elif any(word in question_lower_style for word in ["research", "study", "evidence", "methodology", "hypothesis"]):
                        style = "researcher"
                
                    enhanced_prompt = build_notebooklm_style_prompt(style, content_type_hints, wants_explicit_detail, wants_brief, estimated_length)
                    print(f"   📝 Using NotebookLM-style prompt (style: {style})")
            
                # Create dynamic RAG chain with selected LLM
                try:
                    # Get LLM with potentially adjusted temperature
                    if provider.lower() == "gemini":
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
            
                # Create RAG chain with enhanced context formatting including conversation history
                # Use preretrieved and reranked documents instead of calling retriever again
                # Create a lambda function that returns the already-retrieved documents with conversation history
                def get_preretrieved_docs(_):
                    formatted_context = format_docs(docs)
                    
                    # Add document reference instruction if user specified a document
                    doc_ref_note = ""
                    if referenced_doc:
                        doc_ref_note = f"\n# IMPORTANT: Document Reference\n"
                        doc_ref_note += f"The user specifically asked about the document/source: **{referenced_doc}**\n"
                        doc_ref_note += f"Focus your answer primarily on information from this specific document.\n"
                        doc_ref_note += f"The context below contains chunks from this document (and possibly related documents).\n"
                        doc_ref_note += f"Prioritize information from the referenced document when answering.\n\n"
                    
                    # Add conversation history to context if available
                    if conversation_history:
                        history_text = format_conversation_history(conversation_history, is_follow_up=is_follow_up)
                        # For follow-up questions, prioritize conversation history over documents
                        if is_follow_up:
                            return history_text + doc_ref_note + "\n\n# Additional Document Context\n" + formatted_context
                        else:
                            return history_text + doc_ref_note + formatted_context
                    
                    # No conversation history, but may have document reference
                    return doc_ref_note + formatted_context
            
                # Use preretrieved documents (already reranked) instead of retriever
                rag_chain = ({"context": get_preretrieved_docs, "question": RunnablePassthrough()} | enhanced_prompt | selected_llm)
            
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
                            # Use already-retrieved and reranked documents for expansion
                            # If we have more docs, use them; otherwise use what we have
                            docs_for_expansion = docs[:15] if len(docs) > 15 else docs
                            expanded_context = format_docs(docs_for_expansion)  # Use top 15 chunks
                        
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
                
                # Store question-answer pair in session memory (after any expansion)
                add_to_session(session_id, q, answer)
            
                # Collect sources (documents and links)
                sources = []
                if docs:
                    for d in docs[:10]:  # Collect sources from up to 10 documents
                        src = d.metadata.get("source") or "Unknown"
                        # Clean up the source - extract URL or document name
                        if src and src != "Unknown" and src not in sources:
                            sources.append(src)
                
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
            
                md_lines = ["### Answer:", answer]
                
                # Add Sources section if we have sources
                if sources:
                    md_lines.append("")
                    md_lines.append("### Sources")
                    for i, s in enumerate(sources, 1):
                        # Make URLs clickable
                        if s.startswith(("http://", "https://")):
                            md_lines.append(f"{i}. [{s}]({s})")
                        else:
                            md_lines.append(f"{i}. {s}")
            
                payload = {
                    "answer": answer,
                    "sources": sources,
                    "markdown": "\n".join(md_lines),
                    "needs_url": needs_url,
                    "session_id": session_id
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
    return Handler
