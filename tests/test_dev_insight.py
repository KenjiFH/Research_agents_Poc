import os
import unittest
from unittest import mock

from langchain_core.documents import Document

import dev_insight
from document_processor import (
    ComposedRetriever, CrossEncoderReranker, KeywordIndex, explain_near_duplicates, fusion_details,
    reciprocal_rank_fusion, remove_near_duplicates, tokenize,
)


def doc(chunk_id, text, page=1, kind="text"):
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "page_no": page, "type": kind})


CORPUS = [
    doc("c1", "The warranty covers manufacturing defects for twenty four months.", 1),
    doc("c2", "Revenue for Q3 2024 was 4.2 million dollars driven by subscription growth.", 2),
    doc("c3", "The API rate limit is 100 requests per minute and exceeding it returns HTTP 429.", 3),
    doc("c4", "Shipping takes seven to fourteen business days for international orders.", 3),
    doc("c5", "A quorum requires at least five of the nine board members.", 4),
    doc("c6", "Refunds are issued to the original payment method within ten days.", 5),
]


def captured(function, *args, **kwargs):
    lines = []
    function(*args, out=lines.append, **kwargs)
    return "\n".join(lines)


class DebugSwitchTests(unittest.TestCase):
    def test_off_by_default_and_on_for_common_values(self):
        with mock.patch.dict(os.environ, {"RAG_DEBUG": ""}):
            self.assertFalse(dev_insight.debug_enabled())
        for value in ("1", "true", "YES"):
            with mock.patch.dict(os.environ, {"RAG_DEBUG": value}):
                self.assertTrue(dev_insight.debug_enabled(), value)
        with mock.patch.dict(os.environ, {"RAG_DEBUG": "0"}):
            self.assertFalse(dev_insight.debug_enabled())

    def test_snippet_squashes_whitespace_and_truncates(self):
        self.assertEqual(dev_insight.snippet("a\n\n  b"), "a b")
        self.assertTrue(dev_insight.snippet("word " * 100, 20).endswith("…"))
        self.assertLessEqual(len(dev_insight.snippet("word " * 100, 20)), 20)

    def test_default_output_goes_to_the_console(self):
        with mock.patch("builtins.print") as fake_print:
            dev_insight.print_context_report("abc", [])
        self.assertTrue(fake_print.called)


class ChunkingReportTests(unittest.TestCase):
    def test_reports_counts_sizes_and_the_biggest_chunks(self):
        raw = [doc("r1", "x" * 1500), doc("r2", "table")]
        chunks = [doc("c1", "a" * 900), doc("c2", "b" * 600, 2), doc("c3", "| a |\n" * 1000, 2, "table"), doc("c4", "hi", 3)]
        text = captured(dev_insight.print_chunking_report, raw, chunks, chunk_size=1000, chunk_overlap=200)
        self.assertIn("2 raw elements became 4 chunks (text split at 1000 chars, overlap 200)", text)
        self.assertIn("by type: table 1, text 3", text)
        self.assertIn("p1:1  p2:2  p3:1", text)
        self.assertIn("1 text elements were longer than 1000 chars and were split", text)
        self.assertIn("tiny chunks", text)  # the two-character chunk
        self.assertIn("c3", text.split("largest chunks")[1].splitlines()[1])  # biggest first

    def test_a_very_long_table_chunk_is_flagged(self):
        chunks = [doc("c1", "x" * 6000, 1, "table")]
        self.assertIn("long: embedding may truncate it", captured(dev_insight.print_chunking_report, chunks, chunks))


class Bm25ExplainTests(unittest.TestCase):
    def setUp(self):
        self.index = KeywordIndex(CORPUS)

    def test_term_shares_add_up_to_the_bm25_score(self):
        explanation = self.index.explain("what does error 429 mean for the rate limit")
        question_tokens = list(dict.fromkeys(tokenize("what does error 429 mean for the rate limit")))
        raw_scores = self.index.bm25.get_scores(question_tokens)
        for row in explanation["results"]:
            position = self.index.chunks.index(row["doc"])
            self.assertAlmostEqual(row["score"], raw_scores[position], places=6)

    def test_the_best_chunk_and_the_driving_term_are_reported(self):
        explanation = self.index.explain("error 429")
        top = explanation["results"][0]
        self.assertEqual(top["doc"].metadata["chunk_id"], "c3")
        shares = dict(top["term_scores"])
        self.assertGreater(shares["429"], 0)
        self.assertEqual(shares["error"], 0)  # "error" is in no chunk

    def test_terms_report_how_many_chunks_contain_them(self):
        by_term = {t["term"]: t for t in self.index.explain("the warranty zebra")["terms"]}
        self.assertEqual(by_term["warranty"]["chunks_with_term"], 1)
        self.assertEqual(by_term["zebra"]["chunks_with_term"], 0)
        self.assertGreater(by_term["the"]["chunks_with_term"], 1)

    def test_explain_agrees_with_search_on_which_chunks_match(self):
        question = "board members quorum"
        explained = [r["doc"] for r in self.index.explain(question)["results"]]
        self.assertEqual(explained, self.index.search(question)[: len(explained)])

    def test_empty_inputs_are_safe(self):
        self.assertEqual(KeywordIndex([]).explain("anything")["results"], [])
        self.assertEqual(self.index.explain("???")["results"], [])

    def test_report_names_unmatched_and_rare_terms_and_the_per_term_shares(self):
        text = captured(dev_insight.print_bm25_report, self.index.explain("error 429 quorum"))
        self.assertIn("not in any chunk, cannot match", text)  # "error"
        self.assertIn("429=", text)
        self.assertIn("BM25 KEYWORD SEARCH", text)

    def test_a_tiny_corpus_is_flagged(self):
        text = captured(dev_insight.print_bm25_report, self.index.explain("quorum"))  # 6 chunks
        self.assertIn("only 6 chunks in this upload", text)
        big = KeywordIndex([doc(f"c{i}", f"passage number {i} about topic {i}") for i in range(12)])
        self.assertNotIn("only", captured(dev_insight.print_bm25_report, big.explain("topic")))

    def test_report_says_when_nothing_matches(self):
        text = captured(dev_insight.print_bm25_report, self.index.explain("zebra xylophone"))
        self.assertIn("BM25 contributes nothing here", text)

    def test_a_term_in_half_the_chunks_is_called_out_as_ranking_nothing(self):
        chunks = [doc(f"c{i}", "common word here" + (" rare" if i == 0 else "")) for i in range(4)]
        text = captured(dev_insight.print_bm25_report, KeywordIndex(chunks).explain("common rare"))
        self.assertIn("weight 0, ranks nothing", text)


class FusionAndDedupDetailTests(unittest.TestCase):
    def test_fusion_details_report_ranks_in_each_list_and_match_the_plain_function(self):
        a = [doc("c1", "a"), doc("c2", "b"), doc("c3", "c")]
        b = [doc("c3", "c"), doc("c9", "z")]
        details = fusion_details([a, b])
        by_id = {d["doc"].metadata["chunk_id"]: d for d in details}
        self.assertEqual(by_id["c3"]["ranks"], [3, 1])
        self.assertEqual(by_id["c9"]["ranks"], [0, 2])
        self.assertEqual([d["doc"] for d in details], reciprocal_rank_fusion([a, b]))

    def test_explain_near_duplicates_names_what_each_dropped_chunk_repeats(self):
        header = "acme corp annual report confidential page footer text goes here"
        kept, dropped = explain_near_duplicates([doc("c1", header), doc("c2", header + " again")])
        self.assertEqual([d.metadata["chunk_id"] for d in kept], ["c1"])
        self.assertEqual(dropped[0]["duplicate_of"], "c1")
        self.assertGreaterEqual(dropped[0]["containment"], 0.6)
        self.assertEqual(remove_near_duplicates([doc("c1", header), doc("c2", header + " again")]), kept)


class StagesTests(unittest.TestCase):
    def build(self, reranker=None):
        index = KeywordIndex(CORPUS)
        semantic = lambda q: [CORPUS[0], CORPUS[3], CORPUS[4]]
        return ComposedRetriever(semantic, index.search, reranker, explain_keyword=index.explain)

    def test_the_final_stage_is_exactly_what_invoke_returns(self):
        retriever = self.build()
        stages = retriever.stages("error 429")
        self.assertEqual([d.metadata["chunk_id"] for d in stages["final"]],
                         [d.metadata["chunk_id"] for d in retriever.invoke("error 429")])

    def test_stages_expose_each_step(self):
        stages = self.build().stages("error 429")
        self.assertEqual(len(stages["semantic"]), 3)
        self.assertIn("c3", [d.metadata["chunk_id"] for d in stages["keyword"]])
        rescued = [e for e in stages["fused"] if e["doc"].metadata["chunk_id"] == "c3"][0]
        self.assertEqual(rescued["ranks"][0], 0)  # semantic missed it, keyword found it
        self.assertIsNone(stages["reranked"])
        self.assertTrue(stages["keyword_explanation"]["results"])

    def test_rerank_scores_are_kept_when_a_reranker_runs(self):
        reranker = CrossEncoderReranker(scorer=lambda pairs: [float(i) for i in range(len(pairs))])
        stages = self.build(reranker).stages("error 429")
        self.assertEqual(set(stages["rerank_scores"]), {d.metadata["chunk_id"] for d in stages["deduped"]})
        self.assertEqual(stages["reranked"][0].metadata["chunk_id"], stages["deduped"][-1].metadata["chunk_id"])

    def test_semantic_only_retriever_reports_keyword_as_off(self):
        stages = ComposedRetriever(lambda q: [CORPUS[0]]).stages("q")
        self.assertIsNone(stages["keyword"])
        self.assertIsNone(stages["keyword_explanation"])


class TraceTests(unittest.TestCase):
    def trace(self, reranker=None, question="error 429"):
        retriever = StagesTests().build(reranker)
        return captured(dev_insight.print_retrieval_trace, question, retriever.stages(question), final_k=6)

    def test_trace_walks_every_stage_and_shows_the_funnel(self):
        text = self.trace()
        for expected in ("RETRIEVAL", "funnel: semantic 3 -> keyword", "1. semantic top", "BM25 KEYWORD SEARCH",
                         "3. fusion", "4. dedup", "5. rerank: off", "6. sent to the answerer"):
            self.assertIn(expected, text)

    def test_a_chunk_found_only_by_keyword_is_marked_as_rescued(self):
        self.assertIn("found by keyword only", self.trace())

    def test_rerank_movement_is_shown_with_scores(self):
        reranker = CrossEncoderReranker(scorer=lambda pairs: [float(i) for i in range(len(pairs))])
        text = self.trace(reranker)
        self.assertIn("cross-encoder score", text)
        self.assertIn("moved up", text)

    def test_dedup_drops_are_listed(self):
        header = "acme corp annual report confidential page footer text goes here"
        retriever = ComposedRetriever(lambda q: [doc("c1", header), doc("c2", header + " again")])
        text = captured(dev_insight.print_retrieval_trace, "q", retriever.stages("q"))
        self.assertIn("dropped c2", text)
        self.assertIn("already in c1", text)


class DebugWiringTests(unittest.TestCase):
    def test_invoke_prints_the_trace_only_when_debugging(self):
        retriever = StagesTests().build()
        with mock.patch.dict(os.environ, {"RAG_DEBUG": ""}), mock.patch("builtins.print") as quiet:
            retriever.invoke("error 429")
        quiet.assert_not_called()
        with mock.patch.dict(os.environ, {"RAG_DEBUG": "1"}), mock.patch("builtins.print") as loud:
            retriever.invoke("error 429")
        printed = "\n".join(str(call.args[0]) for call in loud.call_args_list if call.args)
        self.assertIn("RETRIEVAL", printed)


class ContextReportTests(unittest.TestCase):
    def test_reports_size_against_the_window_and_warns_when_close(self):
        chunks = [{"chunk_id": "c1", "page_no": 2, "type": "table", "content": "x" * 9000}]
        text = captured(dev_insight.print_context_report, "x" * 9000, chunks, window_tokens=4096)
        self.assertIn("about 2250 tokens (55% of a 4096-token window", text)
        self.assertNotIn("WARNING", text)
        near = captured(dev_insight.print_context_report, "x" * 14000, chunks, window_tokens=4096)
        self.assertIn("WARNING", near)


if __name__ == "__main__":
    unittest.main()
