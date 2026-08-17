from app.rag.loader import load_document

data = load_document("data/raw/bus_data.csv")

print(type(data))
print(f"Total Records: {len(data)}")

print("\nFirst Record:")
print(data[0])

print("\nSecond Record:")
print(data[1])

###################################################

print(type(data))
print("-" * 50)
print(f"Total Records: {len(data)}")
print("-" * 50)
print(data[0])

####################################################

