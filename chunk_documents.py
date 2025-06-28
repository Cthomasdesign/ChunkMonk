import os
from dotenv import load_dotenv
load_dotenv()

import re
import argparse
import json
from pathlib import Path
from datetime import datetime

import openai
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import os
import pandas as pd
from docx import Document as DocxDocument
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import tiktoken
import csv

# === CONFIG ===
CHUNKS_DIR = "chunks"
os.makedirs(CHUNKS_DIR, exist_ok=True)

# === TEXT EXTRACTION ===
def extract_text_from_file(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif ext == ".docx":
        doc = DocxDocument(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext in [".txt", ".md"]:
        return Path(file_path).read_text()
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    elif ext == ".json":
        data = json.load(open(file_path))
        return json.dumps(data, indent=2)
    else:
        return ""

# === CHUNKING STRATEGIES ===
def llm_chunk(text, prompt_template):
    prompt = prompt_template.format(text=text)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    content = response.choices[0].message.content
    return [chunk.strip() for chunk in content.split('---') if chunk.strip()]

def fixed_chunk(text, max_tokens, overlap=0):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens - overlap)]
    return [encoding.decode(chunk) for chunk in chunks]

def sentence_chunk(text, max_sentences):
    sentences = sent_tokenize(text)
    return [' '.join(sentences[i:i+max_sentences]) for i in range(0, len(sentences), max_sentences)]

def heading_chunk(text, heading_level="#"):
    return [s.strip() for s in re.split(rf"\n{re.escape(heading_level)}+", text) if s.strip()]

def csv_row_chunk(file_path):
    """Chunk a CSV file so each row is a pretty-printed .txt file with field names and values."""
    base_name = Path(file_path).stem
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            pretty_lines = [f"{field}: {row[field]}" for field in reader.fieldnames]
            pretty_content = "\n".join(pretty_lines)
            chunk_filename = f"{base_name}_chunk_{i:03}.txt"
            with open(os.path.join(CHUNKS_DIR, chunk_filename), "w", encoding="utf-8") as f:
                f.write(pretty_content)

# === SAVE CHUNKS ===
def save_chunks(chunks, source_filename):
    base_name = Path(source_filename).stem
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{base_name}_chunk_{i:03}.txt"
        with open(os.path.join(CHUNKS_DIR, chunk_filename), "w") as f:
            f.write(chunk.strip())

# === MAIN ===
def chunk_file(file_path, method, **kwargs):
    text = extract_text_from_file(file_path)
    if not text.strip():
        print(f"Skipped empty or unsupported file: {file_path}")
        return

    if method == "llm":
        chunks = llm_chunk(text, kwargs.get("llm_prompt"))
    elif method == "fixed":
        chunks = fixed_chunk(text, kwargs.get("chunk_size", 300), kwargs.get("overlap", 0))
    elif method == "sentence":
        chunks = sentence_chunk(text, kwargs.get("max_sentences", 5))
    elif method == "heading":
        chunks = heading_chunk(text, kwargs.get("heading_level", "#"))
    elif method == "csv-row":
        csv_row_chunk(file_path)
        return
    else:
        raise ValueError(f"Unknown chunking method: {method}")

    save_chunks(chunks, os.path.basename(file_path))
    print(f"✅ Chunked {file_path} into {len(chunks)} chunks")

def has_been_chunked(file_path):
    base_name = Path(file_path).stem
    # Look for any chunk files that start with this base name
    chunk_files = list(Path(CHUNKS_DIR).glob(f"{base_name}_chunk_*.txt"))
    return len(chunk_files) > 0

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk documents for embedding.")
    parser.add_argument("--method", type=str, choices=["llm", "fixed", "sentence", "heading", "csv-row"], help="Chunking method to use")
    parser.add_argument("--chunk_size", type=int, default=300, help="Token size for fixed chunking")
    parser.add_argument("--max_sentences", type=int, default=5, help="Max sentences per sentence-based chunk")
    parser.add_argument("--heading_level", type=str, default="#", help="Markdown heading level for heading splitting")
    parser.add_argument("--overlap", type=int, default=0, help="Token overlap for fixed chunking")
    parser.add_argument("--llm_prompt", type=str, help="Custom prompt template for LLM chunking. Use {text} as placeholder.")

    args = parser.parse_args()

    args.input_folder = "documents"

    if not args.method:
        print("Choose a chunking method:")
        methods = ["llm", "fixed", "sentence", "heading", "csv-row"]
        for i, m in enumerate(methods):
            print(f"{i + 1}. {m}")
        choice = int(input("Enter number: "))
        args.method = methods[choice - 1]

        if args.method == "fixed":
            args.chunk_size = int(input("Enter chunk size (tokens): "))
            args.overlap = int(input("Enter token overlap: "))
        elif args.method == "sentence":
            args.max_sentences = int(input("Enter number of sentences per chunk: "))
        elif args.method == "heading":
            args.heading_level = input("Enter markdown heading level to split on (e.g., #, ##): ")
        elif args.method == "llm":
            print("Enter your custom prompt (use {text} where the document should be inserted):")
            args.llm_prompt = input("Prompt: ")

    chunk_files = sorted(Path(CHUNKS_DIR).glob("*.txt"))
    for file in Path(args.input_folder).iterdir():
        if file.suffix.lower() in [".pdf", ".docx", ".txt", ".md", ".csv", ".json"]:
            if has_been_chunked(file):
                print(f"⏭️ Skipping already chunked document: {file.name}")
                continue
            if file.suffix.lower() == ".csv" and args.method == "csv-row":
                csv_row_chunk(str(file))
                print(f"✅ Chunked {file} into pretty-printed row chunks")
            else:
                chunk_file(
                    str(file),
                    method=args.method,
                    chunk_size=args.chunk_size,
                    max_sentences=args.max_sentences,
                    heading_level=args.heading_level,
                    overlap=args.overlap,
                    llm_prompt=args.llm_prompt
                )