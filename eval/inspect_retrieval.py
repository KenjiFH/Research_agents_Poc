"""Look inside ingestion and retrieval for a PDF, in the terminal (needs Ollama with nomic-embed-text).

Run from the project root:

    python3 eval/inspect_retrieval.py path/to/file.pdf "what does error 429 mean?" "who is liable?"

It processes the PDF exactly as the app does, prints the chunking report, then for each question
prints the BM25 breakdown and the stage-by-stage retrieval trace (semantic, keyword, fusion, dedup,
rerank, final) and how much context the answerer would receive. Add --bm25 to also print the full
BM25 term table, --no-rerank to skip the cross-encoder, or --chunks N to list the first N chunks.
"""
import argparse
import os
import sys

os.environ["RAG_DEBUG"] = "1"  # the pipeline prints its own chunking report and traces
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dev_insight
import document_processor as dp
from agents import format_context


class LocalUpload:
    """Just enough of Streamlit's UploadedFile for process_document_and_create_vdb."""

    def __init__(self, path):
        self.path = path

    def getvalue(self):
        with open(self.path, "rb") as handle:
            return handle.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf")
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--bm25", action="store_true", help="print the BM25 term table for each question again, standalone")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    parser.add_argument("--chunks", type=int, default=0, metavar="N", help="list the first N chunks")
    args = parser.parse_args()

    retriever, chunks, store = dp.process_document_and_create_vdb(LocalUpload(args.pdf))
    try:
        if args.no_rerank:
            retriever = dp.make_retriever(store, chunks)
        for chunk in chunks[: args.chunks]:
            print(f"{chunk.metadata.get('chunk_id'):>5} p{chunk.metadata.get('page_no')} "
                  f"{chunk.metadata.get('type')}: {dev_insight.snippet(chunk.page_content, 100)}")
        index = dp.KeywordIndex(chunks)
        for question in args.questions:
            if args.bm25:
                dev_insight.print_bm25_report(index.explain(question, k=10), top=10)
            docs = retriever.invoke(question)  # prints the trace because RAG_DEBUG is on
            context, retrieved = format_context(docs)
            dev_insight.print_context_report(context, retrieved)
    finally:
        dp.delete_vector_store(store)


if __name__ == "__main__":
    main()
