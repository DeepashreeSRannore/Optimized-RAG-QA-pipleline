# RAG QA Pipeline with LlamaIndex

A end-to-end Retrieval-Augmented Generation (RAG) pipeline built on a medical document — the WebMD Real Stories: Chronic Migraine Report (Winter 2025) — using LlamaIndex. This project explores, compares, and iteratively improves multiple retrieval strategies to build a smarter, more accurate Q&A system over real-world medical content.

---

## What This Project Does

This pipeline ingests a multi-modal PDF document (text, tables, and embedded images), indexes it using different chunking and embedding strategies, and evaluates retrieval quality across four retrieval methods: Base Vector, Auto-Merging, BM25, and Hybrid Fusion.

The core focus is not just building a working RAG system, but understanding *why* certain retrieval strategies perform better than others on medical content, and systematically improving the pipeline based on those observations.

---

## Project Structure

```
rag-qa-pipeline/
│
├── HW4_Deepashree.ipynb        # Main notebook with all implementations
├── my_questions.txt            # Structured test set (10 questions across 5 categories)
├── WebMD.pdf                   # Source document (Chronic Migraine Report)
└── README.md
```

---

## Pipeline Overview

### Document Ingestion
- Loads the WebMD PDF using `SimpleDirectoryReader`
- Extracts structured table data using `tabula` and converts it to markdown
- Analyzes document structure using `pdfplumber`
- Experiments with four chunking strategies: baseline (chunk size 256, no overlap), small chunks (128, overlap 20), medium chunks (256, overlap 40), and large chunks (512, overlap 50)
- Large chunks with 50-token overlap performed best for this document type, preserving contextual continuity across long medical explanations

### Retrieval Methods
- **Base Vector Retriever** — semantic similarity using vector embeddings
- **BM25 Retriever** — keyword-based lexical matching
- **Auto-Merging Retriever** — dynamically groups related chunks for richer context
- **Hybrid Fusion Retriever** — combines vector and BM25 results using `QueryFusionRetriever` with weighted scores (0.6 vector, 0.4 BM25)

### OCR Integration
- Uses `PyMuPDF` to extract embedded images from the PDF
- Runs `Tesseract OCR` on meaningful images (filters out decorative graphics under 100x100 pixels)
- Aggregates extracted text into the ingestion pipeline
- Includes a post-processing noise filter to remove garbled OCR artifacts before indexing

### Pipeline Enhancements 
Four targeted improvements were implemented on top of the Task 4 baseline:

1. **OCR Noise Reduction** — A character-threshold filter removes unintelligible text blocks, reducing 10 noisy OCR outputs to 3 clean, meaningful documents
2. **Context Enrichment with Metadata Tagging** *(Novel)* — Each document chunk is enriched with metadata at indexing time, including source name, section type, page number, and content category. This helps the retriever make more informed decisions, especially in multi-document settings where the same phrase may carry different meanings
3. **Domain-Adapted Embeddings** — Replaces the generic sentence embedding model with `pritamdeka/S-PubMedBert-MS-MARCO`, a biomedical model fine-tuned on passage retrieval, improving performance on medical terminology in keyword-heavy and semantic queries
4. **Multi-Query Expansion** — Generates 2 paraphrased versions of each user query using the LLM before retrieval, addressing vocabulary mismatch (e.g., matching "prevent migraines" to "prophylactic treatments")

---

## Test Set

A structured set of 10 questions across 5 categories is used to evaluate retrieval performance:

| Category | Focus |
|---|---|
| Cross-sectional | Questions requiring information from multiple sections |
| Keyword-heavy | Specific medical terms like CGRP monoclonal antibodies, drug names |
| Semantic/Contextual | Meaning-based queries without exact keyword matches |
| Structured Data | Table and list extraction |
| Image/Multimedia | Content embedded in figures and infographics |

---

## Evaluation Metrics

Each retrieval method is evaluated on:
- **Faithfulness** — Are the answers grounded in the retrieved context?
- **Relevancy** — Is the retrieved content relevant to the query?
- **MRR** (Mean Reciprocal Rank)
- **Hit Rate**
- **Precision**
- **Recall**

---

## Key Findings

- Large chunks (512 tokens, 50 overlap) consistently outperformed smaller chunks for this document type, as medical content tends to carry meaning across multiple sentences
- Auto-Merging achieved 1.0 relevancy across all question categories but showed faithfulness gaps on keyword-heavy queries
- BM25 performed unexpectedly well on semantic queries but struggled with specialized drug names, likely due to low term frequency
- Hybrid Fusion excelled on structured data and semantic queries, but occasionally underperformed on cross-sectional questions where a single focused retriever would have been more precise
- After all the improvements, the Hybrid Fusion retriever reached 0.80 faithfulness and 0.80 relevancy, transforming a zero-performing baseline (Task 4) into a functional, evaluated system

---

## Tech Stack

- **LlamaIndex** — Core RAG framework
- **OpenAI** — LLM for generation and query expansion
- **HuggingFace Transformers** — Domain-specific biomedical embeddings
- **PyMuPDF** — Image extraction from PDF
- **Tesseract OCR** — Text extraction from images
- **tabula-py** — Table extraction from PDF
- **pdfplumber** — Document structure analysis
- **Python** — pandas, matplotlib, scikit-learn

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-qa-pipeline.git
cd rag-qa-pipeline

# Install dependencies
pip install llama-index openai pymupdf pytesseract tabula-py pdfplumber transformers

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run the notebook
jupyter notebook HW4_Deepashree.ipynb
```

---

## Author

**Deepashree Srinivasa Rao Rannore**    
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
