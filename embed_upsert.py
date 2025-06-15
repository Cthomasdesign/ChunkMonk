import sqlite3
from datetime import datetime
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

print(f"🧪 PINECONE_API_KEY starts with: {os.environ['PINECONE_API_KEY'][:8]}")
print(f"🌍 PINECONE_ENVIRONMENT: {os.environ['PINECONE_ENVIRONMENT']}")
print(f"📦 PINECONE_INDEX: {os.environ.get('PINECONE_INDEX', 'Not Found')}")

import json
from pathlib import Path
import openai
from openai import OpenAI
from pinecone import Pinecone

# === CONFIG ===
CHUNKS_DIR = "chunks"
METADATA_DIR = "metadata"
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
EMBEDDING_MODEL = "text-embedding-3-large"

# === INIT PINECONE ===
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index = pc.Index(os.getenv("PINECONE_INDEX"))

# === LOAD + EMBED + UPSERT ===
def process_and_upsert(chunk_path):
    chunk_id = chunk_path.stem
    print(f"🔍 Processing chunk: {chunk_id}")
    metadata_path = Path(METADATA_DIR) / f"{chunk_id}.json"
    print(f"📄 Looking for metadata at: {metadata_path}")

    chunk_text = chunk_path.read_text()

    # Load metadata
    if not metadata_path.exists():
        print(f"⚠️ Metadata not found for: {chunk_id}")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"📦 Loaded metadata: {metadata}")

    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("🧠 Requesting embedding from OpenAI...")
    # Embed text
    embedding_response = client.embeddings.create(
        input=chunk_text,
        model=EMBEDDING_MODEL
    )
    embedding = embedding_response.data[0].embedding
    print(f"✅ Received embedding of length: {len(embedding)}")

    print(f"📤 Upserting to Pinecone index: {PINECONE_INDEX}")
    # Upsert to Pinecone
    try:
        index.upsert([
            (chunk_id, embedding, metadata)
        ], namespace=namespace)
        print(f"✅ Upserted: {chunk_id}")
    except Exception as e:
        print(f"❌ Failed to upsert {chunk_id}: {e}")
        return

    log_chunk(chunk_id, metadata.get("source_file", "unknown"))

def log_chunk(chunk_id, source_file):
    conn = sqlite3.connect("chunklog.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            source_file TEXT,
            embedded_at TEXT
        )
    """)
    cursor.execute("""
        INSERT OR IGNORE INTO chunks (chunk_id, source_file, embedded_at)
        VALUES (?, ?, ?)
    """, (chunk_id, source_file, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def is_chunk_logged(chunk_id):
    if not os.path.exists("chunklog.db"):
        return False
    conn = sqlite3.connect("chunklog.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

if __name__ == "__main__":
    namespace = input("Enter namespace for Pinecone (default is 'default'): ").strip() or "default"
    print(f"📛 Using namespace: {namespace}")

    print("Select upsert mode:")
    print("1. None (embed everything)")
    print("2. Incremental (skip logged chunks)")
    print("3. Full (clear Pinecone and log, then re-embed)")
    mode_input = input("Enter number: ").strip()

    if mode_input == "1":
        mode = "none"
    elif mode_input == "2":
        mode = "incremental"
    elif mode_input == "3":
        mode = "full"
    else:
        print("Invalid input. Defaulting to 'none'")
        mode = "none"

    print(f"🚀 Running in '{mode}' mode")
    print(f"📁 Scanning chunks in: {CHUNKS_DIR}")

    if mode == "full":
        print("🧹 Clearing Pinecone index and chunk log...")
        try:
            index.delete(delete_all=True, namespace=namespace)
            print(f"✅ Successfully cleared namespace '{namespace}'")
        except Exception as e:
            print(f"❌ Failed to clear namespace '{namespace}': {e}")
        if os.path.exists("chunklog.db"):
            os.remove("chunklog.db")

    chunk_files = sorted(Path(CHUNKS_DIR).glob("*.txt"))
    print(f"📝 Found {len(chunk_files)} chunk(s) to process")
    if not chunk_files:
        print("⚠️ No chunks found in chunks directory.")
    for chunk_path in chunk_files:
        if mode == "incremental" and is_chunk_logged(chunk_path.stem):
            print(f"⏭️ Skipping (already embedded): {chunk_path.stem}")
            continue
        process_and_upsert(chunk_path)