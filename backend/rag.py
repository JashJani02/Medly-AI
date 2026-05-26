import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths
DB_PATH = "data/chroma_db"
UPLOAD_FOLDER = "data/uploads"

os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


def get_vector_db():
    """Load or create ChromaDB instance"""
    
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

    return db


def retrieve_context(query, k=3):
    """Retrieve relevant context from vector DB"""

    db = get_vector_db()

    docs = db.similarity_search(query, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])

    return context


def add_user_file(filepath):
    """Add uploaded file to vector database"""

    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)

    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath)

    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()

    chunks = text_splitter.split_documents(documents)

    db = get_vector_db()

    db.add_documents(chunks)
    db.persist()