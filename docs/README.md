# Docs Folder

Place your PDF, Word (.docx, .doc), and Text (.txt) files in this folder.

## How it works

- Every time `main.py` runs, it automatically scans this folder
- All supported files are automatically loaded into the vector database
- Files are processed only once (tracked to avoid duplicates)
- You can add new files anytime - just restart the server

## Supported file types

- **PDF** (.pdf)
- **Word Documents** (.docx, .doc)
- **Text Files** (.txt)

## Usage

1. Add your documents to this `docs/` folder
2. Restart the server (`python3 app/main.py`)
3. The chatbot will automatically have access to all documents

## Notes

- Files are processed on server startup
- Large files may take a moment to process
- The system tracks which files have been indexed to avoid duplicates
- You can see processing status in the server logs

