# UOE Lahore Academic AI Assistant - Technical Documentation

This document provides a comprehensive record of the steps taken to build and configure the backend ingestion pipeline for the UOE Lahore Academic AI Assistant.

---

## 🏗️ 1. Environment & Dependency Setup

### Python Environment
- **Version**: Python 3.12 (Optimized for performance and latest library compatibility).
- **Manager**: `uv` for fast, reproducible virtual environment management.

### Dependency Modularization
The project uses the latest modular LangChain stack to ensure long-term stability and compatibility with Python 3.12.
- **`pinecone` (v3.0.0+)**: Migrated from the deprecated `pinecone-client` to avoid namespace conflicts.
- **`langchain-core`**: Core abstractions.
- **`langchain-community`**: Document loaders (PyPDF).
- **`langchain-openai`**: High-performance embeddings using `text-embedding-3-large`.
- **`langchain-pinecone`**: Vector store integration.
- **`langchain-text-splitters`**: Advanced text partitioning logic.

---

## 📂 2. Data Architecture & Namespaces

The system implements strict **Domain Isolation** across three namespaces to prevent context contamination during retrieval:

| Namespace | Data Source | Focus |
|-----------|-------------|-------|
| `bs-adp-schemes` | `/Data/BS&ADP` | Undergrad course outlines, CLOs, and prerequisites. |
| `ms-phd-schemes` | `/Data/Ms&Phd` | Graduate & Post-grad research, thesis rules, and courses. |
| `rules-regulations` | `/Data/Rules` | University statutes, admission policies, and fee structures. |

---

## 🧠 3. Ingestion Pipeline Features

### A. Improved Semantic Chunking
Instead of naive character splitting, we implemented an `ImprovedSemanticChunker` that:
- Detects academic boundaries like `Course Code:`, `Article`, `Rule`, and `Semester`.
- Uses optimized sizes (700-1000 characters) to fit within Reranker context windows.
- Filters out "junk" chunks (under 80-100 characters) to maintain high vector density.

### B. Rich Metadata Extraction
The pipeline automatically extracts 20-25 fields per chunk, including:
- **Course Metadata**: Code, Title, Credit Hours, Semesters, Prerequisites.
- **Rule Categorization**: Admission, Fee, Grading, Probation, Discipline.
- **Technical Flags**: Language detection (English/Urdu), Page numbers, File hashes.
- **Reranker Optimization**: Text previews stored in metadata for faster top-k reranking.

### C. Reliability & Performance
- **Deduplication**: Uses MD5 file hashes to skip unchanged documents.
- **Resume Capability**: Progress is persisted to `processed_files.json`. If a run is interrupted, it can be resumed with `python pinecone_ingestion.py --resume`.
- **Batching**: Optimized batch size (50) to balance ingestion speed with OpenAI/Pinecone rate limits.
- **Logging**: Full audit trail in `ingestion.log`.

---

## 🛠️ 4. Execution Guide

### Initial Setup
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Install latest dependencies
uv add -r requirements.txt
```

### Running Ingestion
- **Ingest Everything**: `python pinecone_ingestion.py`
- **Ingest BS/ADP Only**: `python pinecone_ingestion.py bs`
- **Resume Last Run**: `python pinecone_ingestion.py --resume`

---

## 📝 5. Development History (Change Log)
1. **Initial Research**: Mapped `/Data` subfolders to Pinecone namespaces.
2. **Core Scripting**: Built `pinecone_ingestion.py` with custom extraction logic.
3. **Refinement**: Switched to `text-embedding-3-large` with **3072 dimensions** for maximum accuracy.
4. **Cleanup**: Removed all emojis for production-grade readability.
5. **Debug Phase**: Resolved package conflicts between `pinecone` and `pinecone-client`.
6. **Modernization**: Updated imports to modular LangChain paths for Python 3.12 support.

---

## 📊 6. Deployment & Ingestion Audit (Real-world Stats)

Based on the baseline ingestion run, the system achieved the following performance metrics:

- **Total Vectors Synchronized**: 28,848
- **Average Ingestion Speed**: ~5.83 chunks per second
- **Total Depth of Audit**: 141 PDF source files processed with **0% failure rate**.
- **Namespace Distribution**:
    - `bs-adp-schemes`: 19,962 vectors
    - `ms-phd-schemes`: 7296 vectors
    - `rules-regulations`: 1590 vectors

---

## 🛠️ 7. Troubleshooting & Error Resolution

During the deployment phase, the following critical issues were identified and resolved to ensure Python 3.12 compatibility:

### 1. Pinecone Package Rename Conflict
- **Issue**: `Exception: The official Pinecone python package has been renamed from pinecone-client to pinecone.`
- **Resolution**: Forcefully uninstalled `pinecone-client` using `uv pip uninstall pinecone-client` and ensured only `pinecone>=3.0.0` is present in `requirements.txt`.

### 2. LangChain Modular Import Errors
- **Issue**: `ModuleNotFoundError: No module named 'langchain.text_splitter'`.
- **Resolution**: Updated all imports in `pinecone_ingestion.py` to use modular packages:
    - From `langchain.text_splitter` to `langchain_text_splitters`.
    - From `langchain.schema` to `langchain_core.documents`.

### 3. Environment Context Issues
- **Issue**: Standardizing the virtual environment path for consistent terminal usage.
- **Resolution**: Initialized `.venv` using `uv venv` and standardized all execution commands to use `source .venv/bin/activate`.
