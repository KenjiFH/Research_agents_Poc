# Development process: hardening the RAG POC

This note records how the project went from "scores shown on screen" to a pipeline that grades,
gates and cites its answers, what went wrong along the way, and how each problem was handled.
It is written from the working log of the build, and every number comes from a test or script that
is in the repository.

## Where it started

The POC ran Researcher, then Synthesizer, then Evaluator. Review of the code and an outside critique
(`updates.md`) found four weaknesses:

1. **The model graded its own work.** Llama 3 wrote the answer and judged it, and nothing showed the
   judge agreed with a person.
2. **Scores changed nothing.** The UI showed "Faithfulness 0%" or "100%", but each score was a single
   binary pass or fail shown as a percentage, and a low score triggered no warning or action.
3. **No source citations.** Chunks carried only a type and a title, and answers did not say where
   they came from.
4. **The docs had drifted from the code.** The diagram said "returns JSON scores" while the code used
   `CriteriaEvalChain`, and the README claimed page-number metadata the code did not produce.

A review of the code added three more: every upload shared one Chroma collection (so a second PDF
leaked into the first), LangSmith scores were pushed on a 0 to 100 scale where 0 to 1 was expected,
and an evaluator failure silently became a score of 0.

## How the work was run

The work followed the BMad method, one step at a time:

- **Requirements first.** The weaknesses became numbered requirements (R1 to R9), each with a
  checkable result. Later a retrieval review added R10 to R13.
- **An architecture spine.** A short document of binding decisions (AD-1 to AD-9) so separate pieces
  of work could not drift: deterministic graph with one bounded retry, a dependency direction
  between modules, a tri-state verdict, a citation contract, one Chroma collection per upload, one
  shared judge function, an abstention flag, and one composed retriever.
- **Epics and stories.** Six epics and 22 stories in build order, each story small enough for one
  sitting and each with a plain "verify" statement.
- **A review per story.** After each story was built, an independent reviewer read the diff against
  the plan and the spine. Findings were triaged as real, rejected, or deferred, and the triage was
  written into the story's plan file.
- **Measure before and after.** Where a claim could be measured with the local models, a script in
  `eval/` measured it (see below).

The suite grew to 158 tests, all passing. Real Ollama, Chroma, Docling and cross-encoder runs were
used for the measurements, but no test depends on a network or a running model.

## What was built

| Area | Change |
|---|---|
| Ingestion | Every chunk carries `page_no` (real page from Docling) and a stable `chunk_id`; text never spans a page; one uniquely named Chroma collection per upload, deleted on reset or when a different file is uploaded |
| Judge | `JUDGE_MODEL` (default `qwen2.5:14b`) is separate from `ANSWER_MODEL` (default `llama3`); one shared `judge()` function; a clear message if the judge model was never pulled |
| Verdicts | Faithfulness and relevance are `pass`, `fail` or `error`, never a percentage; an evaluator failure is an error, not a silent 0 |
| Gate | A failed faithfulness check retries once, then abstains; off by default behind `GATE_ENABLED` |
| Citations | Context is numbered by chunk id; the model returns `chunk_id` and `quote`; code adds the page and drops any quote that is not verbatim text of its chunk; the UI shows source cards |
| Retrieval | Larger candidate set, near-duplicate removal, BM25 keyword search fused with the embeddings, local cross-encoder reranking, and a panel showing the retrieved chunks for every answer |
| Answers | Schema-constrained structured output with a prompt that asks for complete sentences |

## Challenges and how they were handled

### 1. "The app only matches keywords and sometimes misses the answer entirely"

This started the retrieval work. The first move was **not** to guess a fix.

- **Test of the obvious suspects.** A small experiment against the real `nomic-embed-text` model
  showed that on clean passages retrieval was perfect in every configuration (with or without
  prefixes, cosine or Chroma's default metric, and with `mxbai-embed-large`). So the embedding model
  and its settings were not the cause.
- **Answer side.** With the right passage in context, llama3 answered paraphrased questions 12 of 12
  with the plain prompt and 11 of 12 with the JSON-mode prompt added earlier. The one miss was a bare
  "Yes". That pointed at the answer style, not retrieval.
- **Conclusion.** The failure had to come from real-document characteristics. The likely causes, from
  the code: overlapping and repeated chunks filling all four result slots; purely semantic search
  that blurs exact terms; large unsplit tables; and terse structured answers.
- **First action.** Before changing retrieval, add a **retrieved-chunks panel** so a failing question
  could be diagnosed: is the answer chunk retrieved or not?

### 2. Larger k and near-duplicate removal (story 6.2)

- **Change.** Draw 20 candidates instead of returning the top 4, remove near-duplicates, and pass the
  best 6 to the answerer.
- **A reviewer found a real bug in the first dedup.** It divided by the smaller of two chunks, so a
  short repeated header ranked first could cause the long chunk containing both the header and the
  answer to be dropped. That is exactly the case the change was meant to help. It now measures how much
  of the later chunk's own text is already in a kept chunk, with a test for the header-first case.
- **Other findings fixed.** Chunks with no words no longer collapse into one empty fingerprint, and
  the retriever became a proper LangChain retriever so LangSmith keeps its retriever trace (the first
  version was a plain class and would have lost it).
- **Result.** On a synthetic 60-chunk document with repeated boilerplate and real embeddings
  (`eval/retrieval_recall.py`), the old top-4 retriever found the answer chunk for 4 of 6 questions
  and the new one found all 6.

### 3. BM25 keyword search (story 6.3)

- **Change.** Each upload also builds a BM25 index; semantic and keyword rankings are merged with
  reciprocal rank fusion, then de-duplicated. This needed one new dependency, `rank-bm25`.
- **Challenge: exact figures were split apart.** The first tokenizer turned `4.2` into `4` and `2`,
  and `ABC-123` into two fragments, so exact section numbers and codes matched too loosely. Tokens are
  now kept whole and also split into parts.
- **Challenge: BM25 silently returned nothing on tiny uploads.** Okapi's term weight is zero or
  negative for a term found in half the chunks or more, and the first version dropped any hit with a
  score at or below zero. Membership now depends on sharing a term; the score only orders results.
- **Challenge: a claim without evidence.** The plan cited the 60-chunk experiment, but it only lived in
  a scratch script. The reviewer caught that, and the script is now committed as
  `eval/retrieval_recall.py`.
- **Honest result.** BM25 added nothing on that synthetic test, because the larger k had already found
  every answer. Its value is for an exact term that semantic search ranks outside the top 20. It was
  built because the user asked for it and because the mechanism is tested with scripted rankings, but
  the benefit on a real document is still to be confirmed.

### 4. Reranking with a local cross-encoder (story 6.4)

- **Change.** `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores each (question, chunk) pair after fusion
  and dedup. On the synthetic document the answer chunk ranked first for all six questions. Scoring 40
  chunk pairs takes about a third of a second, after a one-time model load of 4 to 7 seconds.
- **Challenge: a reranker must not break the app.** The reviewer found that one scoring error would have
  disabled reranking for the rest of the upload, that the model reloaded on every upload, and that a
  wrong number of scores, a `None` or a `NaN` could crash retrieval. Now a model that cannot load
  switches reranking off with one warning, a failed or unusable scoring call falls back for that
  question only, and one loaded model is shared across uploads.
- **Challenge: a pass that could be a fallback.** The recall script now says when the reranker could
  not load, so a quiet fallback cannot look like a success.

### 5. A dependency upgrade broke Docling

Installing `sentence-transformers` 6.3.0 upgraded `transformers` from the pinned 4.57.6 to 5.23.0
(along with `tokenizers`, `huggingface_hub` and `safetensors`). Docling's dependencies cannot use that
version, and the app failed on start with
`ImportError: cannot import name 'divide_to_patches' from 'transformers.image_transforms'`.

- **Cause.** The install was done without checking what else it would change. This was a mistake in the
  process, not in the design.
- **Fix.** The pinned versions were restored, and `sentence-transformers==5.7.0`, the newest release
  that works with them, was pinned in `requirements.txt`. `pip check` is clean, and a real Docling
  conversion of a small PDF passes with correct page numbers.
- **Lesson.** Check what an install will upgrade, and run `pip check` and one real end-to-end
  conversion after changing dependencies.

### 6. Terse answers and the answering model (story 6.5)

- **Problem.** Loose JSON mode made llama3 answer in a few words, and one answer was a bare "Yes".
- **Change.** `ANSWER_MODEL` became a separate setting from `JUDGE_MODEL`, and the synthesizer switched
  to schema-constrained output with a prompt that asks for complete sentences. The user confirmed
  `qwen2.5:14b` as the judge, so llama3 stays the answerer.
- **Challenge: the fix introduced regressions that were measured, not assumed.** The first prompt
  answered correctly 12 of 12 and was much longer (about 16 words instead of 6), but llama3 returned
  fewer verified citations (8 instead of 10). The reviewer also warned that the prompt's example
  sentence about leave could be echoed on unrelated documents.
- **Response.** The example was made generic (with a test that the prompt holds no domain wording), and
  the citation instruction now says to cite whenever the answer comes from the context. Re-measured
  with llama3 (`eval/answer_compare.py`): 12 of 12 correct, about 16 words, a verified citation on all
  12 answers, and no echo of the example.
- **Side finding.** With the final prompt, llama3 answered as well as `qwen2.5:14b` (12 of 12 for
  both), so the stronger model is not needed as the answerer.

### 7. Problems found earlier in the hardening work

| Problem | How it was found | Response |
|---|---|---|
| LangSmith feedback was never being sent | A test of the verdict-to-LangSmith mapping failed; a probe showed LangGraph does not pass `run_id` to nodes | The app generates the run id and passes it through the graph config; feedback is only attempted when tracing is on, and a retried run sends only its final verdict |
| A page-less item could give `page_no=None` | Reviewer of story 1.1 | Page-less items inherit the last seen page |
| A failed workflow build leaked its Chroma collection | Reviewer of story 1.2 | The store is saved before the workflow is built, and cleanup keys on the store |
| An unparseable score looked like an unfaithful answer | Design of the tri-state verdict | `error` is its own verdict and never retries or abstains |
| A withheld answer still showed a relevance grade | Reviewer of the gate | After abstaining, relevance is cleared and LangSmith gets only the faithfulness fail |
| `llama3:latest` or `Llama3` would self-grade with no warning | Reviewer of story 3.1 | Model names are compared with the `:latest` tag and case normalized |

## Developer insight in the terminal

After retrieval work, "why did it miss?" needed more than the chunk panel in the UI, so `dev_insight.py`
prints what chunking, BM25 and each retrieval stage are doing. It imports nothing from the project and
changes no behavior; `ComposedRetriever.stages()` exposes the intermediate results, and `invoke` returns
exactly the same chunks as before.

| What | How |
|---|---|
| Chunking report after every upload | `RAG_DEBUG=1 streamlit run app.py`: counts by type and page, size spread, the biggest chunks (flags tables over 4000 chars) and tiny ones |
| Per-question retrieval trace | same switch: funnel (semantic, keyword, fused, after dedup, final), semantic top hits, BM25, fusion with each list's rank (and chunks rescued by keyword only), what dedup dropped and why, rerank scores and movement, what the answerer receives |
| BM25 breakdown | each query term's chunk count and idf (terms in no chunk, and terms in half the chunks or more, are called out), then the top chunks with each term's share of the score; the shares add up to BM25's own score (tested) |
| Context size | how many tokens the answerer is given against a 4096-token window, with a warning near the limit |
| Inspect any PDF without the app | `python3 eval/inspect_retrieval.py file.pdf "question" ...` (add `--bm25`, `--no-rerank` or `--chunks N`) |

A real run on a two-chunk test PDF showed the value straight away: every BM25 term had weight 0
because a corpus that small cannot give BM25 anything to rank with, and the report said so instead of
leaving a silent empty result.

## Decisions the user made

- Build the gate before the judge is validated, but keep it **off** by default (`GATE_ENABLED`) until
  the agreement script has measured the judge against human labels.
- Keep `qwen2.5:14b` as the judge and llama3 as the answerer.
- Use a local cross-encoder for reranking, accepting a one-time model download.
- Add BM25, larger k, dedup and reranking as separate, testable stages behind a single
  `retriever.invoke` interface (AD-9).

## Limits and what is still open

- **The measurements are small and synthetic.** The recall and answer comparisons use a 60-chunk
  document and 12 questions written for the purpose. They show direction, not a benchmark. The real
  test is a real PDF and a question that was failing, using the retrieved-chunks panel.
- **The judge is not yet validated.** Story 3.2 (labeling 20 to 50 cases by hand from a real PDF) and
  story 3.3 (the agreement script) are open. Until they are done, the gate should stay off. The
  20-case file in `eval/` is illustrative; it was written by the assistant and must be re-labeled by a
  person before it counts.
- **LangSmith feedback has not been checked against a real trace.** The unit tests prove the run id is
  passed through and the right scores are sent, but whether feedback lands on the root run of a live
  trace is unconfirmed (story 4.2).
- **No live end-to-end run of the finished app on a real PDF** was done in this log. Docling conversion,
  retrieval, reranking and the answer prompts were each run for real against local models.
- **Documentation is the next epic.** The README and `Architecture.png` do not yet describe the gate,
  BM25, reranking or the new settings (`ANSWER_MODEL`, `JUDGE_MODEL` with `ollama pull qwen2.5:14b`,
  `RERANKER_MODEL`, `GATE_ENABLED`). The old README errors (an unclosed code fence, typos, a false
  page-number claim) are also still there.
- **Deferred ideas:** make the retry tell the model its first answer was unsupported; delete a
  collection when a browser session ends; a refactor sweep, including removing the unused `llm_json`.

## Commit history of the work

| Commit | Change |
|---|---|
| `d70eb1c` | Chunk `page_no` and `chunk_id` (1.1) |
| `48ba36f` | One Chroma collection per upload (1.2) |
| `bee152e` | Separate judge and one judge function (3.1) |
| `6fcc086` | Testable Docling logic, retriever binding, upload lifecycle |
| `b855700` | Numbered context and citations (2.1) |
| `c22d120` | Verify quotes against their chunk (2.2) |
| `8f27546` | Tri-state verdicts; fix LangSmith feedback (4.1) |
| `f5baac1` | Source cards (2.3) |
| `908f96e` | Gate and abstention, off by default (4.3, 4.4) |
| `601bc86` | Retrieved-chunks panel (6.1) |
| `d1c0a53` | Larger k, dedup, BM25 and fusion (6.2, 6.3) |
| `bcdb845` | Cross-encoder reranking (6.4) |
| `60930bc` | Separate answering model, structured output (6.5) |
| (this change) | Developer insight into chunking, BM25 and retrieval stages |

## Reproducing the measurements

```bash
python3 -m unittest discover -s tests -t .     # 158 tests, no network or model needed
python3 eval/retrieval_recall.py               # needs Ollama with nomic-embed-text; first run downloads the reranker
python3 eval/answer_compare.py llama3          # needs Ollama with llama3 and nomic-embed-text
```
