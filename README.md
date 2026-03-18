# Document Intelligence POC: Multi-Agent RAG Reseacher

## Overview
Multi-agent Retrieval-Augmented Generation document intelligence POC . It elevates standard PDF parsing by utilizing Document Layout Analysis to preserve complex structures (like tables and images), employs LangGraph for deterministic agent routing, and integrates LangSmith for full-stack observability and automated evaluation.

## Key Features
* **Document Intelligence Features:** Uses IBM Docling to bypass standard OCR limitations, explicitly extracting and tagging tables and visual elements as isolated Markdown to prevent Vector Dilution.
* **Multi-Agent State Machine:** Replaces unpredictable autonomous agents with a deterministic LangGraph workflow (`Researcher` -> `Synthesizer` -> `Evaluator`).
* **Automated QA & Evaluation:** Utilizes LangChain's native `CriteriaEvalChain` for Chain-of-Thought (CoT) grading, outputting strict binary scores for *Faithfulness* (hallucination checks) and *Relevance*.
* **Full-Stack Observability:** Native LangSmith integration traces node execution, token economics, and pushes programmatic feedback scores directly to the cloud dashboard.

## Tech Stack
* **UI/Frontend:** Streamlit
* **Orchestration:** LangGraph, LangChain (`langchain-classic`)
* **Local LLM & Embeddings:** Ollama (Llama 3, Nomic-Embed-Text)
* **Vector Database:** ChromaDB
* **Document Processing:** IBM Docling
* **Observability:** LangSmith

## Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed locally with `llama3` and `nomic-embed-text` models pulled.
* A [LangSmith](https://smith.langchain.com/) API Key for tracing.

## Environment Setup
Create a `.env` file in the root directory. **Never commit this file to version control.**

```env
# LangSmith Observability Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=[https://api.smith.langchain.com](https://api.smith.langchain.com)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=Doc_Intelligence_POC

