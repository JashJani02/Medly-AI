import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# 1. Init Chroma client (persistent on disk)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="medquad_data")

# 2. Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Load MedQuad dataset (only if DB empty)
if collection.count() == 0:
    print("⚡ Loading MedQuad dataset...")
    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    print("Dataset columns:", dataset.column_names)

    # Use correct keys from dataset
    documents = [f"Q: {row['Question']} A: {row['Answer']}" for row in dataset]

    # Batch insert to avoid Chroma size limits
    BATCH_SIZE = 5000
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_embeds = embedder.encode(batch_docs, show_progress_bar=True).tolist()
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeds,
            ids=[str(j) for j in range(i, i+len(batch_docs))]
        )
        print(f"✅ Added batch {i} - {i+len(batch_docs)}")

    print(f"🎉 Stored {collection.count()} MedQuad QnAs in Chroma")

# 4. Retrieval function
def retrieve_context(query, top_k=3):
    query_vec = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=top_k)
    return " ".join(results["documents"][0])
