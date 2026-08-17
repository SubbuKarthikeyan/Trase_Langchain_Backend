from app.rag.retriever import Retriever

retriever = Retriever()

question = "Bus from Chennai to Madurai"
#question = "Which buses have AC Sleeper?"
#question = "What is the fare from Madurai to Trichy?"

results = retriever.retrieve(question)

print("Retrieved Chunks:\n")

for i, chunk in enumerate(results, start=1):
    print(f"Chunk {i}")
    print("-" * 40)
    print(chunk)
    print()


