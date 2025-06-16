# ChunkMonk 🧩

ChunkMonk is a powerful document processing pipeline that chunks documents, generates metadata, and stores embeddings in Pinecone vector database. It's designed to help you prepare your documents for RAG (Retrieval-Augmented Generation) applications.

## Features ✨

- **Multiple Document Support**: Process PDF, DOCX, TXT, MD, CSV, and JSON files
- **Flexible Chunking Strategies**:
  - LLM-based semantic chunking
  - Fixed-size token chunking
  - Sentence-based chunking
  - Heading-based chunking
- **Intelligent Metadata Generation**:
  - Automatic summary generation
  - Content-based tagging
  - Token and character counting
- **Vector Database Integration**:
  - Seamless Pinecone integration
  - Efficient batch processing
  - Incremental updates
  - Namespace support

## Prerequisites 📋

- Python 3.8+
- OpenAI API key
- Pinecone API key and environment
- Pinecone index

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/Cthomasdesign/ChunkMonk.git
cd ChunkMonk
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX=your_pinecone_index
```

## Usage 📖

### 1. Prepare Your Documents

Place your documents in the `documents` directory. Supported formats:
- PDF (.pdf)
- Word (.docx)
- Text (.txt)
- Markdown (.md)
- CSV (.csv)
- JSON (.json)

### 2. Chunk Your Documents

```bash
python chunk_documents.py --method [chunking_method]
```

Available chunking methods:
- `llm`: Uses GPT-4 for semantic chunking
- `fixed`: Fixed-size token chunks
- `sentence`: Sentence-based chunks
- `heading`: Heading-based chunks

Example with options:
```bash
python chunk_documents.py --method fixed --chunk_size 300 --overlap 50
```

### 3. Generate Metadata

```bash
python generate_metadata.py
```

This will:
- Generate summaries for each chunk
- Create relevant tags
- Calculate token counts
- Save metadata as JSON files

### 4. Create Embeddings and Upsert to Pinecone

```bash
python embed_upsert.py
```

The script will prompt you for:
1. Pinecone namespace (default: "default")
2. Upsert mode:
   - None: Process all chunks
   - Incremental: Skip already processed chunks
   - Full: Clear index and reprocess everything

## Project Structure 📁

```
ChunkMonk/
├── documents/     # Input documents
├── chunks/        # Generated chunks
├── metadata/      # Generated metadata
├── chunk_documents.py
├── generate_metadata.py
├── embed_upsert.py
└── chunklog.db    # Processing log
```