#!/usr/bin/env python3
"""
ChunkMunk Web UI Startup Script

This script starts the web interface for previewing and editing chunks.
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required directories exist"""
    chunks_dir = Path("chunks")
    metadata_dir = Path("metadata")
    
    if not chunks_dir.exists():
        print("❌ Error: 'chunks' directory not found!")
        print("   Please run the chunking script first to create chunks.")
        return False
    
    if not metadata_dir.exists():
        print("❌ Error: 'metadata' directory not found!")
        print("   Please run the metadata generation script first.")
        return False
    
    chunk_files = list(chunks_dir.glob("*.txt"))
    if not chunk_files:
        print("❌ Error: No chunk files found in 'chunks' directory!")
        print("   Please run the chunking script first to create chunks.")
        return False
    
    print(f"✅ Found {len(chunk_files)} chunk files")
    return True

def main():
    print("🚀 Starting ChunkMunk Web UI...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if Flask is installed
    try:
        import flask
        print("✅ Flask is installed")
    except ImportError:
        print("❌ Flask is not installed!")
        print("   Please run: pip install flask")
        sys.exit(1)
    
    print("\n📋 Web UI Features:")
    print("   • View all chunks in a card-based layout")
    print("   • Edit chunk content and metadata")
    print("   • Search chunks by content, summary, or tags")
    print("   • Delete chunks")
    print("   • Real-time character counting")
    print("   • Auto-save functionality")
    
    print("\n🌐 Starting server...")
    print("   The web UI will be available at: http://localhost:8080")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run the web UI
    from web_ui import app
    app.run(debug=True, host='0.0.0.0', port=8080)

if __name__ == "__main__":
    main() 