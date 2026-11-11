import uuid
from pathlib import Path

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

client = QdrantClient(path="rag-app/qdrant_db")
if not client.collection_exists("docs"):
    client.create_collection(
        "docs", vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )


def chunk(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step)]


files = list(Path("rag-app/data").glob("*.txt"))
print("Files found:", [f.name for f in files])

for f in files:
    chunks = chunk(f.read_text(encoding="utf-8"))
    print(f"Embedding {len(chunks)} chunk(s) from {f.name}...")
    points = []
    for n, c in enumerate(chunks):
        emb = ollama.embed(model="nomic-embed-text", input=c)["embeddings"][0]
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{f.name}-{n}"))
        points.append(
            PointStruct(id=pid, vector=emb, payload={"text": c, "source": f.name})
        )
    client.upsert("docs", points=points)
    print("Ingested", f.name)

print("Done. Chunks in database:", client.count("docs").count)