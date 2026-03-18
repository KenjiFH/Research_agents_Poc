
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
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # 2. Convert the document
        result = converter.convert(temp_file_path)
        
        raw_documents = []
        current_text_buffer = ""
        last_seen_text = "Untitled Document Element" # Our short-term memory variable

        # 3. Iterate through elements in strict reading order
        for item, level in result.document.iterate_items():
            
            # --- INTERCEPT TABLES ---
            if item.label == "table":
                # Flush the text buffer first
                if current_text_buffer.strip():
                    raw_documents.append(Document(
                        page_content=current_text_buffer.strip(), 
                        metadata={"type": "text"}
                    ))
                    current_text_buffer = ""
                
                # Convert table to Markdown
                try:
                    df = item.export_to_dataframe()
                    table_md = df.to_markdown(index=False)
                except Exception as e:
                    print(f"Skipping malformed table: {e}")
                    continue
                
                # The Look-Back Heuristic for Tables
                caption = "Untitled Table"
                if hasattr(item, "captions") and item.captions:
                    caption = item.captions[0].text
                else:
                    caption = last_seen_text.strip()
                
                enriched_table_text = f"DOCUMENT TABLE: {caption}\n\n{table_md}"
                
                raw_documents.append(Document(
                    page_content=enriched_table_text, 
                    metadata={"type": "table", "title": caption}
                ))

            # --- INTERCEPT IMAGES / FIGURES ---
            elif item.label == "picture":
                # Flush the text buffer first
                if current_text_buffer.strip():
                    raw_documents.append(Document(
                        page_content=current_text_buffer.strip(), 
                        metadata={"type": "text"}
                    ))
                    current_text_buffer = ""
                
                # The Look-Back Heuristic for Images
                caption = "Untitled Image/Figure"
                if hasattr(item, "captions") and item.captions:
                    caption = item.captions[0].text
                else:
                    caption = last_seen_text.strip()
                
                enriched_image_text = f"DOCUMENT VISUAL ELEMENT (Image/Graph/Diagram)\n"
                enriched_image_text += f"Context/Title: {caption}\n"
                
                # Catch OCR text or VLM annotations if Docling extracted them
                if hasattr(item, "text") and item.text:
                    enriched_image_text += f"\nExtracted Text from Image: {item.text}"
                if hasattr(item, "annotations") and item.annotations:
                    enriched_image_text += f"\nExtracted Image Details: {item.annotations}"

                raw_documents.append(Document(
                    page_content=enriched_image_text.strip(), 
                    metadata={"type": "image", "title": caption, "source_element": "picture"}
                ))
            
            # --- COLLECT NORMAL TEXT ---
            elif item.label in ["text", "paragraph", "section_header", "title", "list_item"]:
                if hasattr(item, "text") and item.text:
                    current_text_buffer += item.text + "\n\n"
                    # Update our memory for the next potential table or image
                    last_seen_text = item.text

        # 4. Flush any remaining text at the very end
        if current_text_buffer.strip():
            raw_documents.append(Document(
                page_content=current_text_buffer.strip(), 
                metadata={"type": "text"}
            ))

        # 5. Selective Chunking 
        # Only split normal text; keep our beautifully formatted tables and image contexts intact
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_chunks = []
        
        for doc in raw_documents:
            if doc.metadata.get("type") in ["table", "image"]:
                final_chunks.append(doc)
            else:
                split_docs = text_splitter.split_documents([doc])
                final_chunks.extend(split_docs)

        print(f"Pipeline yielded {len(final_chunks)} context-aware chunks.")

        # 6. Embed and Store in ChromaDB
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma.from_documents(
            documents=final_chunks,
            embedding=embeddings,
            collection_name="poc_rag_collection"
        )
        
        # Return both the retriever and the chunks so Streamlit can display them in the debug UI
        return vector_store.as_retriever(search_kwargs={"k": 4}), final_chunks

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
