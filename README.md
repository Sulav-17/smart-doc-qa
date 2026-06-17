# Smart Document Q&A System

A RAG-based document question-answering system that allows users to upload PDFs, ask natural language questions, and receive grounded answers with source passages.

## Problem

People often have PDFs, notes, reports, and course materials but cannot easily ask questions across them with accurate citations.

## Solution

This app will let users upload a PDF, extract the text, chunk the content, create embeddings, store them in a vector database, retrieve relevant chunks, and generate grounded answers using an LLM.

## Tech Stack

- Python
- Streamlit
- pypdf
- ChromaDB
- OpenAI API
- python-dotenv
- pytest

## Planned Architecture

PDF Upload  
↓  
Text Extraction  
↓  
Chunking  
↓  
Embedding Model  
↓  
Vector Database  
↓  
User Question  
↓  
Retrieve Relevant Chunks  
↓  
LLM Answer With Sources  
↓  
Streamlit Chat UI  

## Current Status

Milestone 7 complete:

- Project folder created
- Virtual environment set up
- Basic dependencies installed
- Initial folder structure created
- Basic Streamlit app created
- Environment variable example added
- PDF upload added
- PDF text extraction added
- Extracted text preview added
- Document chunking added
- Chunk settings added to sidebar
- Chunk previews added
- OpenAI embeddings module added
- ChromaDB vector storage added
- Vector similarity search added
- RAG answer generation added
- Source passage formatting added
- Basic ingestion, chunking, embeddings, retriever, search, citation, and QA tests added

## How to Run

Create and activate a virtual environment:

```bash
python -m venv .venv

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

## Future Features

- Better Streamlit chat UI
- Conversation memory
- Multi-document support
- PDF page citations
- Local embedding model option

## Local AI Setup

This project runs without paid API access.

### Local Embeddings

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2


## RAG Evaluation

The repository includes a small end-to-end evaluation set covering:

- Supported factual questions
- Paraphrased questions
- Expected source pages
- Unsupported questions
- Misleading premises
- Hallucination refusal behavior

Run the evaluation:

```bash
python scripts/run_evaluation.py --pdf "sample_docs/Build With AI Playbook.pdf"