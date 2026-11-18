import re
import tempfile
from typing import Callable, Optional
import os
import uuid
import pandas as pd
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings

# Using our local Ollama embedding model
EMBEDDING_MODEL = "nomic-embed-text"
# Retrieval (AD-9): draw many candidates, drop near-duplicates, hand the best few to the answerer
CANDIDATE_K = 20       # candidates drawn from each search before filtering
FINAL_K = 6            # chunks the answerer receives
DEDUP_THRESHOLD = 0.6  # word-shingle containment at or above this counts as a near-duplicate
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "").strip() or "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60             # reciprocal-rank-fusion constant used to merge keyword and semantic rankings


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


def get_page_no(item, default=None):
    """Return the 1-based page number of a Docling item, or default if it has none."""
    for prov in getattr(item, "prov", None) or []:
        page_no = getattr(prov, "page_no", None)
        if isinstance(page_no, int):
            return page_no
    return default


def tag_chunks(chunks):
    """Give every chunk a stable, ordered chunk_id (c1, c2, ...) and a 1-based page_no (default 1)."""
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"c{index}"
        if not isinstance(chunk.metadata.get("page_no"), int):
            chunk.metadata["page_no"] = 1
    return chunks


def new_collection_name():
    """A unique Chroma collection name, one per upload (AD-6)."""
    return f"poc_rag_{uuid.uuid4().hex[:12]}"


def build_vector_store(chunks, embeddings):
    """Embed chunks into a brand-new, uniquely named collection and return the store."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=new_collection_name()
    )


def _shingles(text, size=4):
    """Lower-cased word shingles of a text, used to compare chunks for near-duplication."""
    words = re.findall(r"\w+", text.lower())
    if not words:
        return set()
    if len(words) <= size:
        return {" ".join(words)}
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def remove_near_duplicates(docs, threshold=DEDUP_THRESHOLD):
    """Drop a chunk that mostly repeats an earlier, better-ranked one (overlap, boilerplate, repeats).

    A chunk is dropped when at least `threshold` of its own word shingles already appear in one
    kept chunk. A longer chunk that merely contains an earlier short one (a repeated header, say)
    stays, because most of its text is new. The first chunk always stays, and so does a chunk with
    no words, since there is nothing to compare.
    """
    kept, kept_shingles = [], []
    for doc in docs:
        shingles = _shingles(doc.page_content)
        duplicate = bool(shingles) and any(
            len(shingles & other) / len(shingles) >= threshold for other in kept_shingles
        )
        if not duplicate:
            kept.append(doc)
            kept_shingles.append(shingles)
    return kept


COMPOUND = re.compile(r"\w+(?:[.,\-/]\w+)+")
WORD = re.compile(r"\w+")


def tokenize(text):
    """Lower-cased tokens for keyword search.

    Compounds such as '4.2', 'abc-123' or '1,200' are kept whole and also split into their parts,
    so an exact section number or code matches precisely and a partial query still matches.
    """
    text = text.lower()
    tokens = []
    for compound in COMPOUND.findall(text):
        tokens.append(compound)
    tokens.extend(WORD.findall(text))
    return tokens


class KeywordIndex:
    """BM25 keyword search over one upload's chunks, so exact terms are found (R11)."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        corpus = [tokenize(chunk.page_content) for chunk in self.chunks]
        self.token_sets = [set(tokens) for tokens in corpus]
        self.bm25 = BM25Okapi(corpus) if any(corpus) else None

    def search(self, question, k=CANDIDATE_K):
        """Chunks sharing at least one term with the question, best BM25 score first.

        Membership is decided by term overlap, not by the score sign: Okapi idf is zero or
        negative for terms found in half the chunks or more, which would hide hits in small uploads.
        """
        tokens = set(tokenize(question))
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(list(tokens))
        ranked = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        return [self.chunks[i] for i in ranked if tokens & self.token_sets[i]][:k]


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Merge ranked lists of chunks: each list adds 1 / (k + rank) to a chunk's score.

    Chunks are identified by chunk_id; ties keep the order of the earlier list.
    """
    scores, first = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc.metadata.get("chunk_id") or doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            first.setdefault(key, doc)
    return [first[key] for key in sorted(scores, key=lambda key: -scores[key])]


_MODEL_CACHE = {}


def _load_cross_encoder(model_name):
    """One loaded cross-encoder per model name, shared by every upload in this process."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder
        _MODEL_CACHE[model_name] = CrossEncoder(model_name).predict
    return _MODEL_CACHE[model_name]


class CrossEncoderReranker:
    """Re-scores (question, chunk) pairs with a local cross-encoder and sorts best first (R12).

    The model loads on first use and is shared across uploads, so importing this module stays fast
    and it is downloaded once from Hugging Face. A retrieval aid must not break the app: if the
    model cannot load (offline on a first run, say) reranking is switched off for this retriever
    with one warning; if one scoring call fails or returns unusable scores, only that call keeps the
    earlier order. `scorer` lets tests supply a function from (question, text) pairs to scores.
    """

    def __init__(self, model_name=None, scorer=None):
        self.model_name = model_name or RERANKER_MODEL
        self._scorer = scorer
        self._failed = False

    @property
    def unavailable(self):
        """True once the model failed to load, so callers can tell a real rerank from a fallback."""
        return self._failed

    def _load(self):
        if self._scorer is None:
            self._scorer = _load_cross_encoder(self.model_name)
        return self._scorer

    def __call__(self, question, docs):
        docs = list(docs)
        if len(docs) < 2 or self._failed:
            return docs
        try:
            scorer = self._load()
        except Exception as error:
            self._failed = True
            print(f"Reranker unavailable ({error}); keeping the earlier order.")
            return docs
        try:
            scores = [float(score) for score in scorer([(question, doc.page_content) for doc in docs])]
            if len(scores) != len(docs) or any(score != score for score in scores):
                raise ValueError(f"expected {len(docs)} usable scores, got {scores!r}")
        except Exception as error:
            print(f"Reranking failed for this question ({error}); keeping the earlier order.")
            return docs
        order = sorted(range(len(docs)), key=lambda i: -scores[i])
        return [docs[i] for i in order]


class ComposedRetriever(BaseRetriever):
    """The one retriever the graph sees (AD-9): invoke(question) returns chunks in final order.

    A LangChain retriever, so its calls are traced and take callbacks like the one it replaced.
    Candidate search, de-duplication and any later stage run inside; every stage keeps each
    chunk's chunk_id and page_no. `semantic_search` and the optional `keyword_search` each map a
    question to a ranked list of documents; when both are given their rankings are fused.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    semantic_search: Callable[[str], list]
    keyword_search: Optional[Callable[[str], list]] = None
    reranker: Optional[Callable[[str, list], list]] = None
    final_k: int = FINAL_K
    dedup_threshold: float = DEDUP_THRESHOLD

    def __init__(self, semantic_search, keyword_search=None, reranker=None,
                 final_k=FINAL_K, dedup_threshold=DEDUP_THRESHOLD):
        super().__init__(
            semantic_search=semantic_search, keyword_search=keyword_search, reranker=reranker,
            final_k=final_k, dedup_threshold=dedup_threshold,
        )

    def candidates(self, question):
        semantic = list(self.semantic_search(question))
        if self.keyword_search is None:
            return semantic
        return reciprocal_rank_fusion([semantic, list(self.keyword_search(question))])

    def _get_relevant_documents(self, query, *, run_manager=None):
        candidates = remove_near_duplicates(self.candidates(query), self.dedup_threshold)
        if self.reranker is not None:
            candidates = self.reranker(query, candidates)
        return candidates[: self.final_k]


def make_retriever(vector_store, chunks=None, reranker=None):
    """The retriever bound to one upload's store.

    With chunks it also searches them by keyword; with a reranker it re-orders the candidates.
    """
    keyword_search = KeywordIndex(chunks).search if chunks else None
    return ComposedRetriever(
        lambda question: vector_store.similarity_search(question, k=CANDIDATE_K),
        keyword_search=keyword_search,
        reranker=reranker,
    )


def delete_vector_store(vector_store):
    """Delete an upload's collection so a later upload never sees its chunks."""
    if vector_store is not None:
        vector_store.delete_collection()


def items_to_raw_documents(items):
    """Turn Docling (item, level) pairs, in reading order, into typed raw documents.

    Text is buffered and flushed at tables, pictures and page changes; every document gets page_no.
    """
    raw_documents = []
    current_text_buffer = ""
    buffer_page = 1  # page the buffered text started on
    last_page = 1  # last page seen; page-less items inherit it so page_no is always an int
    last_seen_text = "Untitled Document Element"

    # Iterate through elements in strict reading order
    for item, level in items:
        item_label = getattr(item, "label", "")
        item_page = get_page_no(item, default=last_page)
        last_page = item_page

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
                        metadata={"type": "text", "page_no": buffer_page}
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
                        "title": caption,
                        "page_no": item_page
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
                        metadata={"type": "text", "page_no": buffer_page}
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
                        "source_element": "picture",
                        "page_no": item_page
                    }
                )
            )

        # --- COLLECT NORMAL TEXT ---
        elif item_label in ["text", "paragraph", "section_header", "title", "list_item"]:
            text = extract_safe_text(item)

            if text:
                # Never let one text chunk span two pages, so page_no stays exact
                if current_text_buffer.strip() and item_page != buffer_page:
                    raw_documents.append(
                        Document(
                            page_content=current_text_buffer.strip(),
                            metadata={"type": "text", "page_no": buffer_page}
                        )
                    )
                    current_text_buffer = ""

                if not current_text_buffer.strip():
                    buffer_page = item_page
                current_text_buffer += text + "\n\n"
                
                # HEURISTIC FIX: Only update last_seen_text if the chunk is meaningful
                cleaned_text = text.strip()
                is_meaningful_length = len(cleaned_text) > 20
                contains_keyword = any(kw in cleaned_text.lower() for kw in ["table", "figure", "chart", "shows", "illustrates"])
                
                if is_meaningful_length or contains_keyword:
                    last_seen_text = cleaned_text

    # Flush remaining text
    if current_text_buffer.strip():
        raw_documents.append(
            Document(
                page_content=current_text_buffer.strip(),
                metadata={"type": "text", "page_no": buffer_page}
            )
        )

    return raw_documents


def process_document_and_create_vdb(uploaded_file):
    """
    Takes a Streamlit UploadedFile, processes it with Docling (OCR enabled),
    isolates tables and images with look-back metadata tagging,
    and returns a Chroma retriever, the chunks, and the per-upload vector store.
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

        raw_documents = items_to_raw_documents(result.document.iterate_items())

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

        tag_chunks(final_chunks)

        print(f"Pipeline yielded {len(final_chunks)} context-aware chunks.")

        # 6. Embed and store in ChromaDB
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL
        )

        vector_store = build_vector_store(final_chunks, embeddings)

        return (
            make_retriever(vector_store, final_chunks, CrossEncoderReranker()),
            final_chunks,
            vector_store
        )

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)