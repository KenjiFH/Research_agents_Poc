import tempfile
import os
import pandas as pd
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Using our local Ollama embedding model
EMBEDDING_MODEL = "nomic-embed-text"


def extract_safe_text(obj, default=""):
    """
    Safely extract text/content/value from Docling/Pydantic objects.
    Prevents AttributeError crashes from schema variations.
    """
    if obj is None:
        return default

    if hasattr(obj, "text") and obj.text:
        return obj.text

    if hasattr(obj, "content") and obj.content:
        return obj.content

    if hasattr(obj, "value") and obj.value:
        return obj.value

    try:
        dumped = obj.model_dump()

        for key in ["text", "content", "value", "caption"]:
            if key in dumped and dumped[key]:
                return str(dumped[key])

        return str(dumped)

    except Exception:
        return str(obj)


def process_document_and_create_vdb(uploaded_file):
    """
    Takes a Streamlit UploadedFile, processes it with Docling (OCR enabled),
    isolates tables and images with look-back metadata tagging,
    and returns a Chroma retriever and raw chunks.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        print("Running AI Document Intelligence with OCR and Look-Back Heuristics...")

        # 1. Enable OCR and Image Processing in Docling
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        # 2. Convert the document
        result = converter.convert(temp_file_path)

        raw_documents = []
        current_text_buffer = ""
        last_seen_text = "Untitled Document Element"

        # 3. Iterate through elements in strict reading order
        for item, level in result.document.iterate_items():
            item_label = getattr(item, "label", "")

            # --- INTERCEPT TABLES ---
            if item_label == "table":
                # Flush text buffer FIRST so we can use its tail end as context
                preceding_context = ""
                if current_text_buffer.strip():
                    # Save the last ~200 characters of the buffer as semantic context
                    preceding_context = current_text_buffer.strip()[-200:]
                    
                    raw_documents.append(
                        Document(
                            page_content=current_text_buffer.strip(),
                            metadata={"type": "text"}
                        )
                    )
                    current_text_buffer = ""

                # Convert table to markdown
                try:
                    df = item.export_to_dataframe()
                    table_md = df.to_markdown(index=False)
                except Exception as e:
                    print(f"Skipping malformed table: {e}")
                    continue

                # Smarter caption extraction
                if hasattr(item, "captions") and item.captions:
                    caption = extract_safe_text(
                        item.captions[0],
                        default="Untitled Table"
                    )
                else:
                    caption = last_seen_text.strip()

                # Inject the preceding paragraph directly into the table chunk
                enriched_table_text = (
                    f"DOCUMENT TABLE: {caption}\n"
                    f"Preceding Context: {preceding_context}\n\n"
                    f"{table_md}"
                )

                raw_documents.append(
                    Document(
                        page_content=enriched_table_text,
                        metadata={
                            "type": "table",
                            "title": caption
                        }
                    )
                )

            # --- INTERCEPT IMAGES / FIGURES ---
            elif item_label == "picture":
                preceding_context = ""
                if current_text_buffer.strip():
                    preceding_context = current_text_buffer.strip()[-200:]
                    
                    raw_documents.append(
                        Document(
                            page_content=current_text_buffer.strip(),
                            metadata={"type": "text"}
                        )
                    )
                    current_text_buffer = ""

                if hasattr(item, "captions") and item.captions:
                    caption = extract_safe_text(
                        item.captions[0],
                        default="Untitled Image/Figure"
                    )
                else:
                    caption = last_seen_text.strip()

                enriched_image_text = (
                    "DOCUMENT VISUAL ELEMENT (Image/Graph/Diagram)\n"
                    f"Context/Title: {caption}\n"
                    f"Preceding Context: {preceding_context}\n"
                )

                # OCR text
                if hasattr(item, "text") and item.text:
                    enriched_image_text += f"\nExtracted Text from Image: {item.text}"

                # Image annotations
                if hasattr(item, "annotations") and item.annotations:
                    enriched_image_text += f"\nExtracted Image Details: {item.annotations}"

                raw_documents.append(
                    Document(
                        page_content=enriched_image_text.strip(),
                        metadata={
                            "type": "image",
                            "title": caption,
                            "source_element": "picture"
                        }
                    )
                )

            # --- COLLECT NORMAL TEXT ---
            elif item_label in ["text", "paragraph", "section_header", "title", "list_item"]:
                text = extract_safe_text(item)

                if text:
                    current_text_buffer += text + "\n\n"
                    
                    # HEURISTIC FIX: Only update last_seen_text if the chunk is meaningful
                    cleaned_text = text.strip()
                    is_meaningful_length = len(cleaned_text) > 20
                    contains_keyword = any(kw in cleaned_text.lower() for kw in ["table", "figure", "chart", "shows", "illustrates"])
                    
                    if is_meaningful_length or contains_keyword:
                        last_seen_text = cleaned_text

        # 4. Flush remaining text
        if current_text_buffer.strip():
            raw_documents.append(
                Document(
                    page_content=current_text_buffer.strip(),
                    metadata={"type": "text"}
                )
            )

        # 5. Selective chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_chunks = []

        for doc in raw_documents:
            # Do not re-chunk tables or images to preserve structural integrity
            if doc.metadata.get("type") in ["table", "image"]:
                final_chunks.append(doc)
            else:
                split_docs = text_splitter.split_documents([doc])
                final_chunks.extend(split_docs)

        print(f"Pipeline yielded {len(final_chunks)} context-aware chunks.")

        # 6. Embed and store in ChromaDB
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL
        )

        vector_store = Chroma.from_documents(
            documents=final_chunks,
            embedding=embeddings,
            collection_name="poc_rag_collection"
        )

        return (
            vector_store.as_retriever(search_kwargs={"k": 4}),
            final_chunks
        )

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)