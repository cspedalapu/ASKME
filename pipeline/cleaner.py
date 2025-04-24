# pipeline/cleaner.py

import os
import json
import csv
import pdfplumber
from docx import Document
from transformers import AutoTokenizer
import torch

import csv
import sys


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the correct tokenizer for BAAI/bge-m3
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", use_fast=True)

# tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def clean_text(text):
    return ' '.join(text.strip().split())

def count_tokens(text):
    return len(tokenizer.encode(text, truncation=False))

def chunk_text(text, chunk_size=512, stride=20):
    input_ids = tokenizer.encode(text, truncation=False)
    chunks = [
        input_ids[i:i+chunk_size]
        for i in range(0, len(input_ids), chunk_size - stride)
    ]
    return chunks

# Reader functions
def read_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            return '\n'.join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        print(f" Error reading PDF {file_path}: {e}")
        return ""

def read_docx(file_path):
    try:
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f" Error reading DOCX {file_path}: {e}")
        return ""

def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f" Error reading TXT {file_path}: {e}")
        return ""

csv.field_size_limit(2**31 - 1)  # Increase field size limit
def read_csv(file_path):
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            return '\n'.join([', '.join(row) for row in reader])
    except Exception as e:
        print(f" Error reading CSV {file_path}: {e}")
        return ""

def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    except Exception as e:
        print(f" Error reading JSON {file_path}: {e}")
        return ""

def read_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return '\n'.join([json.dumps(json.loads(line)) for line in f])
    except Exception as e:
        print(f" Error reading JSONL {file_path}: {e}")
        return ""

# Main cleaner
def read_and_clean(file_paths, chunk_size=512, stride=20):
    cleaned_docs = []

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        text = ""

        if ext == '.pdf':
            text = read_pdf(file_path)
        elif ext == '.docx':
            text = read_docx(file_path)
        elif ext == '.txt':
            text = read_txt(file_path)
        elif ext == '.csv':
            text = read_csv(file_path)
        elif ext == '.json':
            text = read_json(file_path)
        elif ext == '.jsonl':
            text = read_jsonl(file_path)
        else:
            print(f" Unsupported file type: {file_path}")
            continue

        cleaned = clean_text(text)

        if cleaned.strip():
            input_ids = tokenizer.encode(cleaned, truncation=False)
            tokens = len(input_ids)
            chunks = chunk_text(cleaned, chunk_size=chunk_size, stride=stride)

            cleaned_docs.append({
                'content': cleaned,
                'source': file_path,
                'file_type': ext,
                'file_name': os.path.basename(file_path),
                'num_tokens': tokens,
                'num_chunks': len(chunks)
            })

    print(f" Cleaned and processed {len(cleaned_docs)} documents.")
    return cleaned_docs

# Test
if __name__ == "__main__":
    from loader import get_all_files
    file_list = get_all_files("../data/raw/")
    docs = read_and_clean(file_list)

    if docs:
        print(" Preview of first document:")
        print("File:", docs[0]['file_name'])
        print("Type:", docs[0]['file_type'])
        print("Tokens:", docs[0]['num_tokens'])
        print("Chunks:", docs[0]['num_chunks'])
        print("Text preview:\n", docs[0]['content'][:500], "...")
    else:
        print(" No documents processed.")
