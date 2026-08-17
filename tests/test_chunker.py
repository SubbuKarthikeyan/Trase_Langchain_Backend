from app.rag.loader import load_document
from app.rag.chunker import create_chunks

records = load_document("data/raw/bus_data.csv")

chunks = create_chunks(records)

print(type(chunks))
print(f"Total Chunks: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0])

