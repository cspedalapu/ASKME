**Note**: This folder `output/` containing model embeddings, chunk logs, and QA traces. Due to the size and sensitivity of these artifacts (e.g., `recursive_vectors.json` > 1.4 GB), actual files are excluded to maintain cybersecurity hygiene and repository performance. Users are expected to generate their own outputs by running the pipeline locally.

### Output Directory – Explanation and Usage
The output/ folder is excluded from version control because it contains large, dynamically generated files and sensitive artifacts resulting from model execution.

### Why Not Included?
Security & Privacy:
Some files may contain institution-specific or sensitive academic data (e.g., question-answer logs, user interactions, internal document structures) that shouldn't be shared publicly.

### Storage & Performance:
The file recursive_vectors.json alone exceeds 1.4 GB, and the persistent vector databases like chroma_store_recursive/ are not suited for Git-based versioning due to their binary nature.

### Reproducibility over Replication:
This project follows the principle of reproducibility. All necessary scripts and instructions are included so others can generate their own outputs from raw data, maintaining compliance and control.