import os
import uuid
import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Configuration
PDF_DIR = "/home/npci-admin/Desktop/JFS_2/insurance_pdfs"  # Directory with PDFs
COLLECTION_NAME = "medical_insurance"
CHUNK_SIZE = 200  # Words per chunk

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks based on word count."""
    words = text.split()
    chunks = []
    current_chunk = []
    word_count = 0

    for word in words:
        current_chunk.append(word)
        word_count += 1
        if word_count >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def extract_chunks_from_pdfs(pdf_dir):
    """Extract text chunks from all PDFs in the directory."""
    if not os.path.exists(pdf_dir):
        print(f"Error: Directory {pdf_dir} does not exist")
        return []
    
    # Resolve absolute path for clarity
    pdf_dir = os.path.abspath(pdf_dir)
    print(f"Looking for PDFs in: {pdf_dir}")

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Error: No PDF files found in {pdf_dir}")
        return []

    all_chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"Processing {pdf_file} ({len(pdf.pages)} pages)")
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    if not text:
                        print(f"Warning: No text extracted from page {page_num} of {pdf_file}")
                        continue
                    text = " ".join(text.split())  # Normalize whitespace
                    page_chunks = chunk_text(text)
                    for i, chunk in enumerate(page_chunks):
                        all_chunks.append({
                            "content": chunk,
                            "metadata": {
                                "source": pdf_file,
                                "page": page_num,
                                "chunk_id": i
                            }
                        })
            print(f"Extracted {len(page_chunks)} chunks from {pdf_file}")
        except Exception as e:
            print(f"Failed to process {pdf_file}: {str(e)}")
    
    print(f"Total chunks extracted: {len(all_chunks)}")
    return all_chunks

def prepare_data_for_insertion(chunks):
    """Prepare chunks for Qdrant insertion."""
    if not chunks:
        print("Error: No chunks to prepare for insertion")
        return []
    
    vectors = []
    for item in chunks:
        embedding = model.encode(item["content"]).tolist()
        vectors.append((
            str(uuid.uuid4()),  # Unique ID
            embedding,          # Vector
            {                   # Payload
                "content": item["content"],
                "metadata": item["metadata"]
            }
        ))
    print(f"Prepared {len(vectors)} vectors for insertion")
    return vectors

def setup_collection(collection_name=COLLECTION_NAME):
    """Create or recreate the Qdrant collection."""
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    print(f"Collection '{collection_name}' created successfully")

def insert_vectors(vectors, collection_name=COLLECTION_NAME):
    """Insert vectors into Qdrant."""
    if not vectors:
        print(f"Error: No vectors to insert into '{collection_name}'")
        return
    
    ids, embeddings, payloads = zip(*vectors)
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=ids,
            vectors=embeddings,
            payloads=payloads
        ),
    )
    print(f"Inserted {len(vectors)} records into collection '{collection_name}'")

def test_query(query, collection_name=COLLECTION_NAME, limit=2):
    """Test a query against the collection."""
    query_embedding = model.encode(query).tolist()
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit
    )
    print(f"\nQuery: {query}")
    print("Results:")
    for hit in search_results:
        print(f"Score: {hit.score:.4f}")
        print(f"Content: {hit.payload['content'][:100]}...")
        print(f"Metadata: {hit.payload['metadata']}")
        print()
    return search_results

if __name__ == "__main__":
    print("Starting data loading process...")
    chunks = extract_chunks_from_pdfs(PDF_DIR)
    setup_collection()
    vectors = prepare_data_for_insertion(chunks)
    insert_vectors(vectors)
    print("Data loading completed.")
    
    # Test queries
    test_query("What is a deductible and how does it work?")
    test_query("How do I get money back from my insurance?")