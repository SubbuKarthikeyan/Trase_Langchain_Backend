'''
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print(project_root)   # Optional, just to verify
'''

from app.rag.loader import load_document
from app.rag.chunker import create_chunks
from app.rag.embedder import create_embeddings

records = load_document("data/raw/bus_data.csv")

chunks = create_chunks(records)

embeddings = create_embeddings(chunks)

print(type(embeddings))
print(len(embeddings))

print("FIRST CHUNK")
print(chunks[0])

print("\nFIRST EMBEDDING")
print(embeddings[0])

print("\nVECTOR SIZE")
print(len(embeddings[0]))

# execution code ---> python -m tests.test_embedder


