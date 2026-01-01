import os
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from PyPDF2 import PdfReader

# 1. Init Chroma client (persistent on disk)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="medical_knowledge")

# 2. Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Helper function for Hugging Face datasets
def load_huggingface_dataset(name, split, tag):
    """Load Hugging Face datasets and normalize format."""
    dataset = load_dataset(name, split=split)
    documents = []

    if "Question" in dataset.column_names and "Answer" in dataset.column_names:
        documents = [f"[{tag}] Q: {row['Question']} A: {row['Answer']}" for row in dataset]
    elif "question" in dataset.column_names and "long_answer" in dataset.column_names:
        documents = [f"[{tag}] Q: {row['question']} A: {row['long_answer']}" for row in dataset]
    elif "question" in dataset.column_names and "answer" in dataset.column_names:
        documents = [f"[{tag}] Q: {row['question']} A: {row['answer']}" for row in dataset]
    elif "text" in dataset.column_names:
        documents = [f"[{tag}] {row['text']}" for row in dataset]
    else:
        print(f"⚠️ Unknown format for {name}, skipping.")
    
    return documents


# 4. New: Helper functions for user-uploaded files
def parse_pdf(path, tag="userfile"):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    chunks = text.split("\n\n")
    return [f"[{tag}] {chunk.strip()}" for chunk in chunks if chunk.strip()]


def parse_txt(path, tag="userfile"):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n\n")
    return [f"[{tag}] {line.strip()}" for line in lines if line.strip()]


def parse_csv(path, tag="userfile"):
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        docs.append(f"[{tag}] " + " ".join(str(v) for v in row.values))
    return docs


def parse_json(path, tag="userfile"):
    df = pd.read_json(path)
    docs = []
    for _, row in df.iterrows():
        docs.append(f"[{tag}] " + " ".join(str(v) for v in row.values))
    return docs


def add_user_file(path, tag="userfile"):
    """Add a user-provided file into ChromaDB so chatbot can chat about it."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        docs = parse_pdf(path, tag)
    elif ext == ".txt":
        docs = parse_txt(path, tag)
    elif ext == ".csv":
        docs = parse_csv(path, tag)
    elif ext == ".json":
        docs = parse_json(path, tag)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if not docs:
        print(f"⚠️ No text extracted from {path}")
        return

    embeds = embedder.encode(docs, show_progress_bar=True).tolist()
    collection.add(
        documents=docs,
        embeddings=embeds,
        ids=[f"{tag}_{i}" for i in range(len(docs))]
    )
    print(f"✅ Added {len(docs)} docs from {path}")


# 5. Hugging Face datasets
HUGGINGFACE_DATASETS = [
    ("keivalya/MedQuad-MedicalQnADataset", "train", "medquad"),
    ("qiaojin/pubmedqa", "pqa_labeled", "pubmedqa"),
    ("bigbio/med_qa", "train", "medqa"),
    ("Malikeh1375/medical-question-answering-datasets", "train", "malikeh_medqa"),
]


# 6. Populate ChromaDB if empty
if collection.count() == 0:
    print("⚡ Building knowledge base...")
    documents = []

    for name, split, tag in HUGGINGFACE_DATASETS:
        try:
            docs = load_huggingface_dataset(name, split, tag)
            documents.extend(docs)
            print(f"✅ Added {len(docs)} docs from HuggingFace {name}")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")

    print(f"📥 Encoding {len(documents)} documents...")
    BATCH_SIZE = 2000
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_embeds = embedder.encode(batch_docs, show_progress_bar=True).tolist()
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeds,
            ids=[f"doc_{i+j}" for j in range(len(batch_docs))]
        )
        print(f"🔹 Inserted batch {i} - {i+len(batch_docs)}")

    print(f"🎉 Knowledge base built with {collection.count()} documents!")


# 7. Retrieval function
def retrieve_context(query, top_k=3):
    query_vec = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=top_k)
    return " ".join(results["documents"][0])
