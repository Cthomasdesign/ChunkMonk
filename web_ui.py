from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
from pathlib import Path
from datetime import datetime
import sqlite3
from dotenv import load_dotenv
import openai
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

print(f"🧪 PINECONE_API_KEY starts with: {os.environ['PINECONE_API_KEY'][:8]}")
print(f"🌍 PINECONE_ENVIRONMENT: {os.environ['PINECONE_ENVIRONMENT']}")
print(f"📦 PINECONE_INDEX: {os.environ.get('PINECONE_INDEX', 'Not Found')}")

app = Flask(__name__)

# Configuration
CHUNKS_DIR = "chunks"
METADATA_DIR = "metadata"
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
EMBEDDING_MODEL = "text-embedding-3-large"
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# === INIT PINECONE ===
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host="upstate-gcjwpkh.svc.aped-4627-b74a.pinecone.io")

@app.route('/')
def index_route():
    """Main page showing all chunks"""
    chunks = []
    
    # Get all chunk files
    chunk_files = sorted(Path(CHUNKS_DIR).glob("*.txt"))
    
    for chunk_file in chunk_files:
        chunk_id = chunk_file.stem
        metadata_file = Path(METADATA_DIR) / f"{chunk_id}.json"
        
        # Read chunk content
        with open(chunk_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Read metadata if exists
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Check if chunk is in Pinecone
        in_pinecone = is_chunk_in_pinecone(chunk_id)
        
        chunks.append({
            'id': chunk_id,
            'content': content,
            'metadata': metadata,
            'char_count': len(content),
            'source_file': metadata.get('source_file', 'Unknown'),
            'summary': metadata.get('summary', 'No summary available'),
            'tags': metadata.get('tags', []),
            'in_pinecone': in_pinecone
        })
    
    return render_template('index.html', chunks=chunks)

@app.route('/chunk/<chunk_id>')
def view_chunk(chunk_id):
    """View/edit a specific chunk"""
    chunk_file = Path(CHUNKS_DIR) / f"{chunk_id}.txt"
    metadata_file = Path(METADATA_DIR) / f"{chunk_id}.json"
    
    if not chunk_file.exists():
        return "Chunk not found", 404
    
    # Read chunk content
    with open(chunk_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Read metadata if exists
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return render_template('chunk_detail.html', 
                         chunk_id=chunk_id,
                         content=content,
                         metadata=metadata)

@app.route('/api/chunk/<chunk_id>', methods=['PUT'])
def update_chunk(chunk_id):
    """Update chunk content and metadata, and upsert to Pinecone"""
    data = request.json
    
    # Update chunk content
    chunk_file = Path(CHUNKS_DIR) / f"{chunk_id}.txt"
    if chunk_file.exists():
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(data['content'])
    
    # Update metadata
    metadata_file = Path(METADATA_DIR) / f"{chunk_id}.json"
    metadata = data.get('metadata', {})
    metadata['updated_at'] = datetime.utcnow().isoformat()
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # === Upsert to Pinecone ===
    try:
        # Get OpenAI API key and embedding model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_model = EMBEDDING_MODEL
        
        # Generate new embedding for the updated content
        client = OpenAI(api_key=openai_api_key)
        embedding_response = client.embeddings.create(
            input=data['content'],
            model=embedding_model
        )
        embedding = embedding_response.data[0].embedding
        
        # Upsert to Pinecone
        index.upsert([
            (chunk_id, embedding, metadata)
        ], namespace=PINECONE_NAMESPACE)
        print(f"✅ Upserted updated chunk to Pinecone: {chunk_id}")
    except Exception as e:
        print(f"❌ Error upserting updated chunk to Pinecone: {e}")
        return jsonify({'success': False, 'message': f'Error upserting to Pinecone: {str(e)}'}), 500
    
    return jsonify({'success': True, 'message': 'Chunk updated successfully and upserted to Pinecone'})

@app.route('/api/chunk/<chunk_id>', methods=['DELETE'])
def delete_chunk(chunk_id):
    """Delete a chunk and its metadata from both local files and Pinecone"""
    chunk_file = Path(CHUNKS_DIR) / f"{chunk_id}.txt"
    metadata_file = Path(METADATA_DIR) / f"{chunk_id}.json"
    
    # Use environment variable for namespace
    namespace = PINECONE_NAMESPACE
    
    try:
        # Delete from Pinecone if it exists there
        if namespace:
            print(f"🗑️ Deleting from Pinecone namespace: {namespace}")
            index.delete(ids=[chunk_id], namespace=namespace)
            print(f"✅ Deleted from Pinecone: {chunk_id}")
        
        # Remove from chunk log
        remove_chunk_from_log(chunk_id)
        
        # Delete local files
        if chunk_file.exists():
            chunk_file.unlink()
            print(f"✅ Deleted local chunk file: {chunk_id}")
        
        if metadata_file.exists():
            metadata_file.unlink()
            print(f"✅ Deleted local metadata file: {chunk_id}")
        
        return jsonify({'success': True, 'message': 'Chunk deleted successfully from both local files and Pinecone'})
        
    except Exception as e:
        print(f"❌ Error deleting chunk {chunk_id}: {e}")
        return jsonify({'success': False, 'message': f'Error deleting chunk: {str(e)}'}), 500

def get_chunk_namespace(chunk_id):
    """Get the namespace where a chunk was stored in Pinecone"""
    if not os.path.exists("chunklog.db"):
        return None
    
    conn = sqlite3.connect("chunklog.db")
    cursor = conn.cursor()
    
    # Check if namespace column exists, if not return 'default'
    cursor.execute("PRAGMA table_info(chunks)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'namespace' in columns:
        cursor.execute("SELECT namespace FROM chunks WHERE chunk_id = ?", (chunk_id,))
        result = cursor.fetchone()
        namespace = result[0] if result else 'default'
    else:
        # If no namespace column, assume 'default'
        namespace = 'default'
    
    conn.close()
    return namespace

def remove_chunk_from_log(chunk_id):
    """Remove a chunk from the SQLite log"""
    if not os.path.exists("chunklog.db"):
        return
    
    conn = sqlite3.connect("chunklog.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))
    conn.commit()
    conn.close()
    print(f"✅ Removed from chunk log: {chunk_id}")

@app.route('/search')
def search():
    """Search chunks by content or metadata"""
    query = request.args.get('q', '').lower()
    chunks = []
    
    chunk_files = sorted(Path(CHUNKS_DIR).glob("*.txt"))
    
    for chunk_file in chunk_files:
        chunk_id = chunk_file.stem
        
        # Read chunk content
        with open(chunk_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Read metadata
        metadata_file = Path(METADATA_DIR) / f"{chunk_id}.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Search in content, summary, and tags
        searchable_text = f"{content} {metadata.get('summary', '')} {' '.join(metadata.get('tags', []))}".lower()
        
        if query in searchable_text:
            # Check if chunk is in Pinecone
            in_pinecone = is_chunk_in_pinecone(chunk_id)
            
            chunks.append({
                'id': chunk_id,
                'content': content[:200] + "..." if len(content) > 200 else content,
                'metadata': metadata,
                'char_count': len(content),
                'source_file': metadata.get('source_file', 'Unknown'),
                'summary': metadata.get('summary', 'No summary available'),
                'tags': metadata.get('tags', []),
                'in_pinecone': in_pinecone
            })
    
    return render_template('search.html', chunks=chunks, query=query)

def log_chunk(chunk_id, source_file, namespace="default"):
    conn = sqlite3.connect("chunklog.db")
    cursor = conn.cursor()
    
    # Create table with namespace column if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            source_file TEXT,
            embedded_at TEXT,
            namespace TEXT DEFAULT 'default'
        )
    """)
    
    # Add namespace column if it doesn't exist (for backward compatibility)
    cursor.execute("PRAGMA table_info(chunks)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'namespace' not in columns:
        cursor.execute("ALTER TABLE chunks ADD COLUMN namespace TEXT DEFAULT 'default'")
    
    cursor.execute("""
        INSERT OR IGNORE INTO chunks (chunk_id, source_file, embedded_at, namespace)
        VALUES (?, ?, ?, ?)
    """, (chunk_id, source_file, datetime.utcnow().isoformat(), namespace))
    conn.commit()
    conn.close()

def is_chunk_in_pinecone(chunk_id):
    namespace = PINECONE_NAMESPACE
    try:
        response = index.fetch(ids=[chunk_id], namespace=namespace)
        return chunk_id in response.vectors
    except Exception as e:
        print(f"Error checking Pinecone for {chunk_id}: {e}")
        return False

@app.route('/api/clear_namespace', methods=['POST'])
def clear_namespace():
    """Delete all vectors from the current Pinecone namespace and remove local chunk files and metadata."""
    try:
        # Delete all vectors from Pinecone namespace
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print(f"✅ Cleared all vectors from Pinecone namespace: {PINECONE_NAMESPACE}")
        
        # Optionally, clear local chunk files and metadata
        for chunk_file in Path(CHUNKS_DIR).glob("*.txt"):
            chunk_file.unlink()
        for meta_file in Path(METADATA_DIR).glob("*.json"):
            meta_file.unlink()
        print("✅ Cleared all local chunk and metadata files")
        
        # Optionally, clear chunk log
        if os.path.exists("chunklog.db"):
            os.remove("chunklog.db")
            print("✅ Removed chunklog.db")
        
        return jsonify({'success': True, 'message': f'All chunks cleared from namespace {PINECONE_NAMESPACE} and local storage.'})
    except Exception as e:
        print(f"❌ Error clearing namespace: {e}")
        return jsonify({'success': False, 'message': f'Error clearing namespace: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 