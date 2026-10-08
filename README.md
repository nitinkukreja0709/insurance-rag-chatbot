# Insurance RAG Chatbot

A multilingual question-answering chatbot over insurance policy brochures, built with retrieval-augmented generation and a local LLM.

## How it works
1. **Ingest** — `insurance-data-loader.py` / `document_loader-pdf.py` extract text from policy PDFs (`pdfplumber`), chunk it, and embed it with `all-MiniLM-L6-v2` (sentence-transformers).
2. **Index** — chunks are stored in a **Qdrant** vector collection.
3. **Answer** — `insurance-chatbot.py` serves a **Flask** API: it detects the query language (`langdetect`), retrieves the most relevant chunks from Qdrant, and generates a grounded answer with a local model via **Ollama** (default `phi4-mini`), called asynchronously.

## Run
```bash
pip install flask qdrant-client sentence-transformers pdfplumber langdetect aiohttp
ollama pull phi4-mini
python insurance-data-loader.py      # build the index
python insurance-chatbot.py          # start the API
```
Sample documents in `insurance_pdfs/` are publicly available product brochures.
