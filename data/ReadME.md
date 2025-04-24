**Note:** The data/ folder is intentionally left empty in this repository to avoid uploading institution-specific, sensitive, or copyrighted content.

### Prepare Your Own Dataset
To use this project effectively, you must create and place your own data inside the data/ folder. This should include documents relevant to your university or domain such as:

Academic advising guides (PDF, DOCX, TXT)

Departmental policies and program info (JSON, CSV)

Webpages saved as HTML or text

Any other structured/unstructured knowledge sources


### Tips for Preparing Data: 

Ensure files are readable and clean (avoid scanned images unless OCR is used)

Recommended formats: .pdf, .txt, .json, .csv, .docx

Avoid uploading confidential student or employee data

Once your documents are ready, the pipeline scripts (like recursive_embedder.py) will automatically load, clean, chunk, and embed them.