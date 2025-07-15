import os


def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


# Example usage:
if __name__ == "__main__":
    path = "data/raw/example.txt"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        with open(f"data/processed/chunk_{i}.txt", "w", encoding="utf-8") as out:
            out.write(chunk)
