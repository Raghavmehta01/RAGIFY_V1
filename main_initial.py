import os, time, json, requests
from http.server import HTTPServer, SimpleHTTPRequestHandler

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # new import

# ---- Settings ----
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")  # default to Mistral 7B
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
UA = os.getenv("USER_AGENT", "RAGBot/1.0 (+https://example.com/contact) LangChain-WebBaseLoader")

def wait_for_ollama(base=OLLAMA_BASE, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base}/api/tags", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Ollama not responding at 11434")

# ---- Build RAG at startup ----
wait_for_ollama()

loader = WebBaseLoader(
    web_paths=("https://orangedatatech.com/team/",),
    raise_for_status=False, continue_on_failure=True, trust_env=True,
)
loader.requests_kwargs = {"headers": {"User-Agent": UA}}
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
splits = splitter.split_documents(docs)

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
else:
    vectorstore = Chroma.from_documents(splits, embedding=emb, persist_directory=CHROMA_DIR)
    vectorstore.persist()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Use new langchain-ollama LLM
llm = OllamaLLM(model=MODEL, base_url=OLLAMA_BASE, temperature=0.1)

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

rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm)

class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
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

            answer = rag_chain.invoke(q)
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

            md_lines = ["### Answer:", str(answer), "", "### Sources:"]
            for i, s in enumerate(sources, 1):
                md_lines.append(f"{i}. {s}")
            payload = {"answer": str(answer), "sources": sources, "markdown": "\n".join(md_lines)}
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
        if self.path == "/ask":
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
        else:
            super().do_OPTIONS()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"Serving on http://127.0.0.1:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
