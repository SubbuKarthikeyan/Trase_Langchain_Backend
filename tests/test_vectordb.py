from app.rag.loader import load_document
from app.rag.chunker import create_chunks
from app.rag.embedder import create_embeddings
from app.rag.vectordb import VectorDB

records = load_document("data/raw/bus_data.csv")
chunks = create_chunks(records)
embeddings = create_embeddings(chunks)

db = VectorDB()

db.clear()          # optional for testing
db.store(chunks, embeddings)

data = db.get_all()

print("\nTotal IDs :", len(data["ids"]))

print("\nFirst ID:")
print(data["ids"][0])

print("\nFirst Document:")
print(data["documents"][0])

print("\nEmbedding Dimension:")
print(len(data["embeddings"][0]))

print("\nFirst 5 Embedding Values:")
print(data["embeddings"][0][:5])

