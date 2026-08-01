from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

client = Groq(api_key=os.getenv("groq_api_key"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load and chunk (no destructive preprocessing)
with open("C:/Users/nani3/Desktop/sample.txt", "r", encoding="utf-8") as f:
    file_content = f.read()

chunks=  [c.strip() for c in file_content.split("\n\n") if c.strip()]


# Build FAISS index
embeddings = model.encode(chunks,normalize_embeddings=True)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.array(embeddings, dtype="float32"))

print("RAG ready. Type 'exit' to quit.\n")

while True:
    prompt = input("Nani: ").strip()
    if prompt.lower() == "exit":
        break

    q_emb = model.encode([prompt],normalize_embeddings=True)
    _, indices = index.search(np.array(q_emb, dtype="float32"), 5)
    context = "\n".join(chunks[i] for i in indices[0])


    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
    {
        "role": "system",
        "content": f"""
You are Nani.

Answer only using the provided context.

If the answer is not present, say:
"I don't know based on the provided document."

Never make up information.

Context:
{context}
"""
    },
    {"role": "user", "content": prompt}
],
        max_completion_tokens=1000
    )
    print("\nNani:", response.choices[0].message.content, "\n")