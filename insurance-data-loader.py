import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Configuration
COLLECTION_NAME = "medical_insurance_qa"

# Pre-generated Q&A pairs based on the provided documents
qa_pairs = [
    {"content": "Q: What is the minimum age at entry for HDFC Life Click 2 Protect Life? A: 18 years", "metadata": {"source": "HDFC Life Click 2 Protect Life", "page": 2}},
    {"content": "Q: What is the maximum age at entry for HDFC Life Click 2 Protect Life? A: 45 years for any Sum Assured combination as per options; 50 years for the 75 Lacs Sum Assured option", "metadata": {"source": "HDFC Life Click 2 Protect Life", "page": 2}},
    {"content": "Q: What are the key features of HDFC Life Click 2 Protect Life? A: Customized packages with flexibility in Term Cover, protection against 19 Critical Illnesses with Critical Illness Rider, life cover with 2x coverage for accidental death, a regular monthly income payable for 10 years in case of Accidental Total Permanent Disability, and quick issuance with no medicals", "metadata": {"source": "HDFC Life Click 2 Protect Life", "page": 2}},
    {"content": "Q: What is the death benefit under HDFC Life Click 2 Protect Life? A: It provides life cover with 2x coverage for accidental death, as part of the Click 2 Protect Life - Life Protect Option", "metadata": {"source": "HDFC Life Click 2 Protect Life", "page": 2}},
    {"content": "Q: What is the sample premium for a 94 L package with Non-ROP option for a 35-year-old male non-smoker? A: ₹13,092 annually, excluding tax, for a 40-year policy term with regular pay, covering Click 2 Protect Life (28L), Accidental Death Benefit (28L), Income Benefit Rider (28L), and Critical Illness Plus Rider (10L)", "metadata": {"source": "HDFC Life Click 2 Protect Life", "page": 2}},
    {"content": "Q: What is the maximum age at maturity for HDFC Life Click 2 Protect Life? A: 75 years", "metadata": {"source": "HDFC Life Click 2 Protect Life", "page": 2}},
    {"content": "Q: What does the Critical Illness Rider cover in HDFC Life Click 2 Protect Life? A: It provides protection against 19 Critical Illnesses with a sum assured of 10L", "metadata": {"source": "HDFC Life Click 2 Protect Life", "page": 2}},
    {"content": "Q: What is the benefit of the Income Benefit on Accidental Disability Rider? A: A regular monthly income payable for 10 years in case of Accidental Total Permanent Disability, with the same sum assured as Click 2 Protect Life (e.g., 25L or 15L depending on package)", "metadata": {"source": "HDFC Life Click 2 Protect Life", "page": 2}},
    {"content": "Q: Can I get a return of premium with HDFC Life Click 2 Protect Life? A: Yes, both ROP (Return of Premium) and Non-ROP options are allowed, subject to Board Approved Underwriting Policy (BAUP)", "metadata": {"source": "HDFC Life Click 2 Protect Life", "page": 2}},
    {"content": "Q: How can I purchase HDFC Life Click 2 Protect Life? A: Call 1800-266-9777 (toll-free) or visit www.hdfclife.com", "metadata": {"source": "HDFC Life Click 2 Protect Life", "page": 3}},
    {"content": "Q: What is the minimum age at entry for HDFC Life Click 2 Protect Super? A: 18 years", "metadata": {"source": "HDFC Life Click 2 Protect Super", "page": 2}},
    {"content": "Q: What is the maximum age at entry for HDFC Life Click 2 Protect Super? A: 84 years for Life and Life Plus options; 65 years for Life Goal option", "metadata": {"source": "HDFC Life Click 2 Protect Super", "page": 2}},
    {"content": "Q: What are the key features of HDFC Life Click 2 Protect Super? A: Comprehensive financial protection, choice of 3 plan options, return of all premiums paid on survival till maturity with ROP option, additional amount payable for accidental death, acceleration of death benefit on terminal illness diagnosis till age 80, and options to increase death benefit, vary it, or waive premiums on critical illness or disability", "metadata": {"source": "HDFC Life Click 2 Protect Super", "page": 2}},
    {"content": "Q: What is the death benefit under the Life Option of HDFC Life Click 2 Protect Super? A: Payable as a lump sum, it’s the higher of Sum Assured on Death × SA Factor or 105% of Total Premiums Paid; for example, ₹1,00,00,000 for a 35-year-old with a 40-year term if death occurs in the 7th year", "metadata": {"source": "HDFC Life Click 2 Protect Super", "page": 4}},
    {"content": "Q: What is the sample premium for the Life Option with a ₹1 crore cover for a 35-year-old? A: ₹20,033 annually (excluding taxes) for a 40-year policy term with regular pay", "metadata": {"source": "HDFC Life Click 2 Protect Super", "page": 4}},
    {"content": "Q: What is the maximum age at maturity for HDFC Life Click 2 Protect Super? A: 85 years", "metadata": {"source": "HDFC Life Click 2 Protect Super", "page": 2}},
    {"content": "Q: What is the Terminal Illness Benefit in HDFC Life Click 2 Protect Super? A: Sum Assured on Death, up to ₹2 crore, is accelerated if diagnosed with a terminal illness (likely to lead to death within 6 months), provided the diagnosis is before age 80", "metadata": {"source": "HDFC Life Click 2 Protect Super", "page": 6}},
    {"content": "Q: What additional benefit is provided under the Life Plus Option? A: An additional amount equal to the Sum Assured on Death is payable in case of accidental death during the policy term", "metadata": {"source": "HDFC Life Click 2 Protect Super", "page": 7}},
    {"content": "Q: How does the Life Goal Option work in HDFC Life Click 2 Protect Super? A: The sum assured payable on death varies with the policy year based on a chosen Level Cover Period and Amortization Rate (0% to 20% p.a.), starting at 100% and reducing post-level period", "metadata": {"source": "HDFC Life Click 2 Protect Super", "page": 8}},
    {"content": "Q: What is the Smart Exit Benefit in HDFC Life Click 2 Protect Super? A: The policyholder can receive Total Premiums Paid by canceling the policy after 30 years but not in the last 5 policy years, if the policy is in-force and Life Goal or ROP options are not selected", "metadata": {"source": "HDFC Life Click 2 Protect Super", "page": 16}}
]

def prepare_data_for_insertion(qa_pairs):
    """Prepare Q&A pairs for Qdrant insertion."""
    if not qa_pairs:
        print("Error: No Q&A pairs to prepare for insertion")
        return []
    
    vectors = []
    for item in qa_pairs:
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
    print("Starting Q&A data loading process...")
    # Use the pre-generated qa_pairs list
    setup_collection()
    vectors = prepare_data_for_insertion(qa_pairs)
    insert_vectors(vectors)
    print("Data loading completed.")
    
    # Test queries
    test_query("What is the minimum age for insurance?")
    test_query("What are the benefits of the insurance plan?")
    test_query("How much is the premium for a 35-year-old?")