import os
import json
from pathlib import Path
from datetime import datetime

from openai import OpenAI
import tiktoken

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === CONFIG ===
CHUNKS_DIR = "chunks"
METADATA_DIR = "metadata"
os.makedirs(METADATA_DIR, exist_ok=True)

# === TOKENIZER ===
encoding = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text):
    return len(encoding.encode(text))

# === LLM-BASED METADATA GENERATION ===
def generate_summary_and_tags(text):
    prompt = f"""
Text:
\"\"\"{text}\"\"\"

Summarize this in 1 sentence and provide 3–5 concise tags related to its content.
Format the output as:

Summary: <summary sentence>
Tags: <tag1>, <tag2>, <tag3>, ...
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    content = response.choices[0].message.content

    # Parse summary and tags
    lines = content.strip().split("\n")
    summary = lines[0].replace("Summary:", "").strip()
    tags_line = next((l for l in lines if "Tags:" in l), "")
    tags = [t.strip().lower() for t in tags_line.replace("Tags:", "").split(",") if t.strip()]
    
    return summary, tags

# === METADATA GENERATION ===
def process_chunk_file(file_path):
    chunk_text = Path(file_path).read_text()
    chunk_filename = Path(file_path).name
    base_name = chunk_filename.replace(".txt", "")
    
    summary, tags = generate_summary_and_tags(chunk_text)
    
    metadata = {
        "chunk_id": base_name,
        "source_file": base_name.rsplit("_chunk_", 1)[0],
        "chunk_index": int(base_name.split("_")[-1]),
        "text": chunk_text,
        "summary": summary,
        "tags": tags,
        "char_count": len(chunk_text),
        "token_count": count_tokens(chunk_text),
        "created_at": datetime.utcnow().isoformat()
    }

    # Save JSON
    metadata_path = os.path.join(METADATA_DIR, f"{base_name}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_path}")

# === MAIN ===
if __name__ == "__main__":
    chunk_files = sorted(Path(CHUNKS_DIR).glob("*.txt"))
    for file_path in chunk_files:
        process_chunk_file(file_path)