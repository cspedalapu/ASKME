# This script loads all files from the data directory and its subdirectories.
# It supports various file formats including PDF, JSON, JSONL, CSV, TXT, and DOCX.

import os

SUPPORTED_EXTENSIONS = ['.pdf', '.json', '.jsonl', '.csv', '.txt', '.docx']

def get_all_files(data_dir='../data/raw/'):
    loaded_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, file)
                loaded_files.append(full_path)
    
    print(f" Total files found: {len(loaded_files)}")
    for path in loaded_files:
        print("", path)
    
    return loaded_files

# Run directly
if __name__ == "__main__":
    get_all_files()
