"""Developer insight: print what chunking, BM25 and the retrieval stages are doing, to the console.

Everything here only formats data it is handed and imports nothing from the project, so it sits
beside the three layers without coupling them (AD-2). Set RAG_DEBUG=1 and the app prints a chunking
report after every upload and a stage-by-stage trace for every question. Each print_* function
takes `out`, a function that receives one line of text (default `print`), so tests can capture it.

    RAG_DEBUG=1 streamlit run app.py
    python3 eval/inspect_retrieval.py some.pdf "what does error 429 mean?"
"""
import os
import statistics

TOKEN_CHARS = 4  # rough characters per token, only for size estimates


def debug_enabled():
    """True when RAG_DEBUG is set to 1, true or yes. Read at call time, so it can change per run."""
    return os.getenv("RAG_DEBUG", "").strip().lower() in ("1", "true", "yes")


def snippet(text, limit=70):
    """One line of a chunk's text: whitespace squashed, cut with an ellipsis."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _emit(out, text=""):
    (out or print)(text)


def _heading(out, title):
    _emit(out)
    _emit(out, f"== {title} " + "=" * max(3, 70 - len(title)))


def _label(doc):
    meta = doc.metadata
    return f"{meta.get('chunk_id', '?'):>5}  p{meta.get('page_no', '?'):<3} {str(meta.get('type', '?')):<5}"


# ---------------------------------------------------------------- chunking


def print_chunking_report(raw_documents, chunks, chunk_size=None, chunk_overlap=None, out=None):
    """What ingestion produced: counts by type and page, size spread, the biggest and tiniest chunks."""
    _heading(out, "CHUNKING")
    setting = f" (text split at {chunk_size} chars, overlap {chunk_overlap})" if chunk_size else ""
    _emit(out, f"{len(raw_documents)} raw elements became {len(chunks)} chunks{setting}")

    by_type, by_page = {}, {}
    for chunk in chunks:
        kind = chunk.metadata.get("type", "?")
        by_type[kind] = by_type.get(kind, 0) + 1
        page = chunk.metadata.get("page_no", "?")
        by_page[page] = by_page.get(page, 0) + 1
    _emit(out, "by type: " + ", ".join(f"{kind} {count}" for kind, count in sorted(by_type.items())))
    pages = sorted(by_page.items(), key=lambda item: (not isinstance(item[0], int), item[0]))
    _emit(out, "by page: " + "  ".join(f"p{page}:{count}" for page, count in pages))

    for kind in sorted(by_type):
        sizes = [len(c.page_content) for c in chunks if c.metadata.get("type", "?") == kind]
        _emit(out, f"{kind:<6} size chars  min {min(sizes)}  median {int(statistics.median(sizes))}  "
                   f"mean {int(statistics.mean(sizes))}  max {max(sizes)}")

    split_sources = sum(1 for raw in raw_documents if raw.metadata.get("type") == "text"
                        and chunk_size and len(raw.page_content) > chunk_size)
    if split_sources:
        _emit(out, f"{split_sources} text elements were longer than {chunk_size} chars and were split")

    _emit(out, "largest chunks (tables and images are never re-split):")
    for chunk in sorted(chunks, key=lambda c: -len(c.page_content))[:5]:
        chars = len(chunk.page_content)
        warn = "  <-- long: embedding may truncate it and it can crowd the model context" if chars > 4000 else ""
        _emit(out, f"  {_label(chunk)} {chars:>6} chars ~{chars // TOKEN_CHARS:>5} tokens  {snippet(chunk.page_content, 40)}{warn}")

    tiny = [c for c in chunks if len(c.page_content.strip()) < 40]
    if tiny:
        _emit(out, f"{len(tiny)} tiny chunks (under 40 chars), usually headers or page furniture:")
        for chunk in tiny[:5]:
            _emit(out, f"  {_label(chunk)} {snippet(chunk.page_content, 40)!r}")


# ---------------------------------------------------------------- BM25


def print_bm25_report(explanation, top=5, out=None):
    """Why BM25 ranked chunks the way it did: each query term's weight, then per-term contributions."""
    _heading(out, "BM25 KEYWORD SEARCH")
    terms = explanation["terms"]
    corpus = explanation.get("corpus_size")
    _emit(out, "query terms: " + (", ".join(t["term"] for t in terms) or "(none)"))
    if corpus is not None and corpus < 10:
        _emit(out, f"note: only {corpus} chunks in this upload; BM25 weights need a larger corpus to mean much")
    _emit(out, f"{'term':<14}{'in chunks':>10}{'idf':>8}  note")
    for term in terms:
        if term["chunks_with_term"] == 0:
            note = "not in any chunk, cannot match"
        elif term["idf"] <= 0:
            note = "in half the chunks or more: weight 0, ranks nothing"
        elif term["idf"] > 3:
            note = "rare: strong signal"
        else:
            note = ""
        _emit(out, f"{term['term']:<14}{term['chunks_with_term']:>10}{term['idf']:>8.2f}  {note}")

    results = explanation["results"]
    _emit(out, f"{explanation['matches']} chunks share at least one term; top {min(top, len(results))}:")
    for rank, row in enumerate(results[:top], start=1):
        parts = " ".join(f"{term}={score:.2f}" for term, score in row["term_scores"] if score)
        _emit(out, f"  #{rank} {_label(row['doc'])} score {row['score']:6.2f}  [{parts}]")
        _emit(out, f"       {snippet(row['doc'].page_content)}")
    if not results:
        _emit(out, "  no chunk shares a term with the question, so BM25 contributes nothing here")


# ---------------------------------------------------------------- retrieval


def _ids(docs):
    return [d.metadata.get("chunk_id") for d in docs]


def print_retrieval_trace(question, stages, final_k=None, top=6, out=None):
    """Follow one question through semantic search, BM25, fusion, dedup, rerank and the final cut."""
    _heading(out, f"RETRIEVAL  {snippet(question, 60)!r}")
    semantic, keyword = stages["semantic"], stages["keyword"]
    funnel = [f"semantic {len(semantic)}"]
    if keyword is not None:
        funnel.append(f"keyword {len(keyword)}")
    funnel.append(f"fused {len(stages['fused'])}")
    funnel.append(f"after dedup {len(stages['deduped'])}")
    funnel.append(f"final {len(stages['final'])}")
    _emit(out, "funnel: " + " -> ".join(funnel))

    _emit(out, f"1. semantic top {min(top, len(semantic))} (embedding similarity):")
    for rank, doc in enumerate(semantic[:top], start=1):
        _emit(out, f"   #{rank} {_label(doc)} {snippet(doc.page_content)}")

    if stages.get("keyword_explanation"):
        _emit(out, "2. keyword search (BM25):")
        print_bm25_report(stages["keyword_explanation"], top=min(top, 3), out=out)
    elif keyword is None:
        _emit(out, "2. keyword search is off for this retriever")

    if keyword is not None:
        _emit(out, "3. fusion (reciprocal rank: each list adds 1/(60+rank)):")
        for row in stages["fused"][:top]:
            sem = f"semantic #{row['ranks'][0]}" if row["ranks"][0] else "not in semantic"
            kw = f"keyword #{row['ranks'][1]}" if row["ranks"][1] else "not in keyword"
            rescue = "  <-- found by keyword only" if not row["ranks"][0] and row["ranks"][1] else ""
            _emit(out, f"   {_label(row['doc'])} rrf {row['score']:.4f}  {sem}, {kw}{rescue}")

    dropped = stages["dropped"]
    _emit(out, f"4. dedup: {len(dropped)} near-duplicates removed")
    for item in dropped[:top]:
        _emit(out, f"   dropped {item['doc'].metadata.get('chunk_id')} "
                   f"({item['containment']:.0%} of its text is already in {item['duplicate_of']})")

    reranked = stages.get("reranked")
    if reranked is None:
        _emit(out, "5. rerank: off")
    else:
        before = {doc.metadata.get("chunk_id"): pos for pos, doc in enumerate(stages["deduped"], start=1)}
        scores = stages.get("rerank_scores") or {}
        _emit(out, "5. rerank (cross-encoder score, higher is better):")
        for pos, doc in enumerate(reranked[:top], start=1):
            cid = doc.metadata.get("chunk_id")
            move = before.get(cid, pos) - pos
            arrow = f"moved up {move}" if move > 0 else (f"moved down {-move}" if move < 0 else "same place")
            score = f"{scores[cid]:7.2f}" if cid in scores else "    n/a"
            _emit(out, f"   #{pos} {_label(doc)} score {score}  (was #{before.get(cid, pos)}, {arrow})")

    _emit(out, f"6. sent to the answerer ({len(stages['final'])}"
               + (f" of at most {final_k}" if final_k else "") + "):")
    for rank, doc in enumerate(stages["final"], start=1):
        _emit(out, f"   #{rank} {_label(doc)} {snippet(doc.page_content)}")


def print_context_report(context, retrieved_chunks, window_tokens=4096, out=None):
    """How much text the answerer is given, against a typical model context window."""
    _heading(out, "CONTEXT SENT TO THE ANSWERER")
    chars = len(context)
    tokens = chars // TOKEN_CHARS
    share = tokens / window_tokens if window_tokens else 0
    _emit(out, f"{len(retrieved_chunks)} chunks, {chars} chars, about {tokens} tokens "
               f"({share:.0%} of a {window_tokens}-token window, before the prompt and the answer)")
    for chunk in retrieved_chunks:
        size = len(chunk.get("content", ""))
        _emit(out, f"   {chunk.get('chunk_id', '?'):>5} p{chunk.get('page_no', '?'):<3} {str(chunk.get('type')):<5} {size:>6} chars")
    if share > 0.75:
        _emit(out, "   WARNING: the context is near the window size, so the model may lose the start of it")
