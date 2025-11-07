# File: app/keep_only_payplus.py
# Changes made by Raghav Mehta with current timestamp: 2025-01-15 14:30:00
# Reason: Script to remove all documents from ChromaDB except PayPlus-related ones
# - Keeps only documents with "payplus" in source metadata (URLs or filenames)
# - Deletes all other documents (GarnishEdge, OrangeDataTech team page, etc.)
# - Cleans up tracking files to match remaining documents
# - Preserves PayPlus 360 knowledge base for ODT team restructuring project

import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set same paths as main.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("🔍 Loading ChromaDB vector store...")

# Initialize embeddings
emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Load vectorstore
if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    print("❌ ChromaDB directory is empty or doesn't exist.")
    exit(1)

vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)

# Get all documents
print("📚 Retrieving all documents from vector store...")
all_docs = vectorstore.get()

if not all_docs or not all_docs.get('ids'):
    print("❌ No documents found in vector store.")
    exit(1)

total_docs = len(all_docs['ids'])
print(f"   Found {total_docs} total document(s)")

# Identify documents to keep (PayPlus-related) and delete (everything else)
payplus_ids = []
delete_ids = []

print("\n🔍 Analyzing documents...")
for i, doc_id in enumerate(all_docs['ids']):
    metadata = all_docs.get('metadatas', [{}])[i] if all_docs.get('metadatas') else {}
    source = str(metadata.get('source', '')).lower()
    
    # Check if source contains "payplus" (case-insensitive)
    if 'payplus' in source:
        payplus_ids.append(doc_id)
        print(f"   ✅ KEEP: {metadata.get('source', 'Unknown')} (ID: {doc_id[:8]}...)")
    else:
        delete_ids.append(doc_id)
        if i < 10:  # Show first 10 for preview
            print(f"   🗑️  DELETE: {metadata.get('source', 'Unknown')} (ID: {doc_id[:8]}...)")

if len(delete_ids) > 10:
    print(f"   ... and {len(delete_ids) - 10} more documents to delete")

print(f"\n📊 Summary:")
print(f"   ✅ Documents to KEEP (PayPlus): {len(payplus_ids)}")
print(f"   🗑️  Documents to DELETE: {len(delete_ids)}")

if not delete_ids:
    print("\n✅ No documents to delete. All documents are PayPlus-related.")
    exit(0)

# Confirm deletion (non-interactive mode - can be overridden with --yes flag)
skip_confirm = '--yes' in sys.argv or '-y' in sys.argv

if not skip_confirm:
    print(f"\n⚠️  WARNING: This will delete {len(delete_ids)} document(s) from ChromaDB.")
    print("   Only PayPlus-related documents will remain.")
    try:
        response = input("   Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Cancelled. No changes made.")
            exit(0)
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Cancelled. No changes made.")
        print("   Tip: Use --yes or -y flag to skip confirmation: python3 app/keep_only_payplus.py --yes")
        exit(0)
else:
    print(f"\n⚠️  WARNING: Deleting {len(delete_ids)} document(s) from ChromaDB (auto-confirmed).")

# Delete documents
print(f"\n🗑️  Deleting {len(delete_ids)} document(s)...")
try:
    vectorstore.delete(ids=delete_ids)
    print(f"✅ Successfully deleted {len(delete_ids)} document(s)")
except Exception as e:
    print(f"❌ Error deleting documents: {e}")
    exit(1)

# Clean up tracking files
print("\n🧹 Cleaning tracking files...")

# Clean .loaded_urls.txt - keep only PayPlus URLs
urls_file = os.path.join(PERSIST_DIR, ".loaded_urls.txt")
if os.path.exists(urls_file):
    with open(urls_file, 'r') as f:
        lines = f.readlines()
    kept_urls = [line for line in lines if 'payplus' in line.lower()]
    with open(urls_file, 'w') as f:
        f.writelines(kept_urls)
    removed_count = len(lines) - len(kept_urls)
    if removed_count > 0:
        print(f"  ✅ Cleaned .loaded_urls.txt (removed {removed_count} non-PayPlus URLs)")
    else:
        print(f"  ✅ .loaded_urls.txt already clean")

# Clean .loaded_docs.txt - keep only PayPlus-related files
docs_file = os.path.join(PERSIST_DIR, ".loaded_docs.txt")
if os.path.exists(docs_file):
    with open(docs_file, 'r') as f:
        lines = f.readlines()
    kept_docs = [line for line in lines if 'payplus' in line.lower()]
    with open(docs_file, 'w') as f:
        f.writelines(kept_docs)
    removed_count = len(lines) - len(kept_docs)
    if removed_count > 0:
        print(f"  ✅ Cleaned .loaded_docs.txt (removed {removed_count} non-PayPlus files)")
    else:
        print(f"  ✅ .loaded_docs.txt already clean")

# Clean .loaded_files.txt - keep only PayPlus-related files
files_file = os.path.join(PERSIST_DIR, ".loaded_files.txt")
if os.path.exists(files_file):
    with open(files_file, 'r') as f:
        lines = f.readlines()
    kept_files = [line for line in lines if 'payplus' in line.lower()]
    with open(files_file, 'w') as f:
        f.writelines(kept_files)
    removed_count = len(lines) - len(kept_files)
    if removed_count > 0:
        print(f"  ✅ Cleaned .loaded_files.txt (removed {removed_count} non-PayPlus files)")
    else:
        print(f"  ✅ .loaded_files.txt already clean")

print(f"\n✅ Cleanup complete!")
print(f"   📊 Remaining documents: {len(payplus_ids)} PayPlus-related document(s)")
print(f"   💡 Restart your server to see the changes.")
print(f"   🔄 Run: python3 app/main.py")

