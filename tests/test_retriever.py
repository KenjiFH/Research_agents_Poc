import unittest

from langchain_core.documents import Document

from document_processor import (
    CANDIDATE_K, ComposedRetriever, FINAL_K, KeywordIndex, make_retriever, reciprocal_rank_fusion, remove_near_duplicates, tokenize,
)


def doc(chunk_id, text, page=1):
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "page_no": page, "type": "text"})


PASSAGE = "the warranty covers manufacturing defects for a period of twenty four months from purchase date"


class NearDuplicateTests(unittest.TestCase):
    def test_a_repeat_of_a_better_ranked_chunk_is_dropped(self):
        docs = remove_near_duplicates([doc("c1", PASSAGE), doc("c2", PASSAGE + " extra"), doc("c3", "shipping takes seven days")])
        self.assertEqual([d.metadata["chunk_id"] for d in docs], ["c1", "c3"])

    def test_a_short_chunk_inside_a_longer_one_is_dropped(self):
        longer = PASSAGE + " and water damage is excluded from all claims made under this policy"
        docs = remove_near_duplicates([doc("c1", longer), doc("c2", PASSAGE)])
        self.assertEqual([d.metadata["chunk_id"] for d in docs], ["c1"])

    def test_a_long_chunk_containing_an_earlier_short_header_is_kept(self):
        header = "acme corp annual report confidential page footer"
        long_chunk = header + " the refund is issued within ten business days after the returned item is inspected by staff"
        docs = remove_near_duplicates([doc("c1", header), doc("c2", long_chunk)])
        self.assertEqual([d.metadata["chunk_id"] for d in docs], ["c1", "c2"])

    def test_chunks_without_words_are_kept_and_never_collapse_each_other(self):
        docs = remove_near_duplicates([doc("c1", "---"), doc("c2", "***"), doc("c3", "real words here for once")])
        self.assertEqual(len(docs), 3)

    def test_distinct_chunks_with_a_little_shared_text_are_kept(self):
        a = doc("c1", "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu")
        b = doc("c2", "iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon")
        self.assertEqual(len(remove_near_duplicates([a, b])), 2)

    def test_order_and_metadata_are_preserved(self):
        docs = remove_near_duplicates([doc("c9", "first unique passage about apples", 3), doc("c2", "second unique passage about pears", 7)])
        self.assertEqual([(d.metadata["chunk_id"], d.metadata["page_no"]) for d in docs], [("c9", 3), ("c2", 7)])

    def test_the_first_chunk_always_stays_even_when_short(self):
        self.assertEqual(len(remove_near_duplicates([doc("c1", "x")])), 1)


class ComposedRetrieverTests(unittest.TestCase):
    def test_duplicates_no_longer_crowd_out_the_answer_chunk(self):
        crowd = [doc(f"c{i}", PASSAGE + f" variant {i}") for i in range(1, 8)]  # seven near-copies
        answer = doc("c40", "the refund is issued within ten business days after inspection of the item")
        retriever = ComposedRetriever(lambda q: crowd + [answer])
        ids = [d.metadata["chunk_id"] for d in retriever.invoke("when is my refund issued")]
        self.assertIn("c40", ids)
        self.assertEqual(len([i for i in ids if i != "c40"]), 1)

    def test_a_larger_candidate_set_reaches_past_the_old_top_four(self):
        docs = [doc(f"c{i}", f"completely different topic number {i} with unique words w{i} x{i} y{i}") for i in range(1, 21)]
        retriever = ComposedRetriever(lambda q: docs)
        self.assertEqual(len(retriever.invoke("q")), FINAL_K)
        self.assertGreater(FINAL_K, 4)

    def test_it_is_a_traced_langchain_retriever(self):
        from langchain_core.retrievers import BaseRetriever
        self.assertIsInstance(ComposedRetriever(lambda q: []), BaseRetriever)

    def test_invoke_matches_the_interface_the_graph_uses(self):
        retriever = ComposedRetriever(lambda q: [doc("c1", "only chunk")])
        self.assertEqual([d.metadata["chunk_id"] for d in retriever.invoke("q", config={"tags": ["t"]})], ["c1"])


CORPUS = [
    doc("c1", "The warranty covers manufacturing defects for twenty four months."),
    doc("c2", "Revenue for Q3 2024 was 4.2 million dollars driven by subscription growth."),
    doc("c3", "The API rate limit is 100 requests per minute and exceeding it returns HTTP 429."),
    doc("c4", "Shipping takes seven to fourteen business days for international orders."),
    doc("c5", "A quorum requires at least five of the nine board members."),
]


class KeywordSearchTests(unittest.TestCase):
    def test_tokenize_keeps_numbers_and_codes(self):
        self.assertEqual(tokenize("What was Q3 revenue? HTTP 429!"), ["what", "was", "q3", "revenue", "http", "429"])

    def test_tokenize_keeps_compounds_whole_and_split(self):
        tokens = tokenize("See section 4.2, part ABC-123 and $1,200.")
        for expected in ("4.2", "abc-123", "abc", "123", "1,200", "section"):
            self.assertIn(expected, tokens)

    def test_a_trailing_period_is_not_part_of_a_term(self):
        self.assertIn("429", tokenize("It returns HTTP 429."))
        self.assertNotIn("429.", tokenize("It returns HTTP 429."))

    def test_an_exact_section_number_beats_a_chunk_that_only_has_its_digits(self):
        filler = [doc(f"f{i}", f"unrelated passage number {i} about shipping boxes and trucks") for i in range(8)]
        chunks = filler + [doc("c1", "Section 3.1 covers refunds."), doc("c2", "Section 4.2 covers indemnity for 3 claims and 1 appeal.")]
        self.assertEqual(KeywordIndex(chunks).search("what does section 3.1 say")[0].metadata["chunk_id"], "c1")

    def test_hits_are_returned_even_in_tiny_uploads(self):
        for chunks in ([doc("c1", "the warranty lasts 24 months")], [doc("c1", "the warranty lasts 24 months"), doc("c2", "the warranty is void if wet")]):
            self.assertIn("c1", [d.metadata["chunk_id"] for d in KeywordIndex(chunks).search("warranty months")])

    def test_an_exact_code_finds_its_chunk_first(self):
        index = KeywordIndex(CORPUS)
        self.assertEqual(index.search("what does error 429 mean")[0].metadata["chunk_id"], "c3")
        self.assertEqual(index.search("Q3 revenue")[0].metadata["chunk_id"], "c2")

    def test_a_question_sharing_no_terms_returns_nothing(self):
        self.assertEqual(KeywordIndex(CORPUS).search("zebra xylophone"), [])

    def test_empty_inputs_are_safe(self):
        self.assertEqual(KeywordIndex([]).search("anything"), [])
        self.assertEqual(KeywordIndex(CORPUS).search("???"), [])
        self.assertEqual(KeywordIndex([doc("c1", "")]).search("x"), [])


class FusionTests(unittest.TestCase):
    def ids(self, docs):
        return [d.metadata["chunk_id"] for d in docs]

    def test_a_chunk_ranked_by_both_lists_beats_one_ranked_by_a_single_list(self):
        a = [doc("c1", "a"), doc("c2", "b"), doc("c3", "c")]
        b = [doc("c3", "c"), doc("c9", "z")]
        self.assertEqual(self.ids(reciprocal_rank_fusion([a, b]))[:2], ["c3", "c1"])

    def test_each_chunk_appears_once_and_keeps_its_metadata(self):
        fused = reciprocal_rank_fusion([[doc("c1", "a", 4)], [doc("c1", "a", 4), doc("c2", "b", 7)]])
        self.assertEqual([(d.metadata["chunk_id"], d.metadata["page_no"]) for d in fused], [("c1", 4), ("c2", 7)])

    def test_ties_keep_the_earlier_list_first(self):
        fused = reciprocal_rank_fusion([[doc("c1", "a")], [doc("c2", "b")]])
        self.assertEqual(self.ids(fused), ["c1", "c2"])


class HybridRetrieverTests(unittest.TestCase):
    def test_keyword_search_rescues_an_exact_term_the_semantic_search_missed(self):
        semantic = lambda q: [CORPUS[0], CORPUS[3], CORPUS[4]]  # misses c3 entirely
        keyword = KeywordIndex(CORPUS).search
        without = ComposedRetriever(semantic).invoke("what does error 429 mean")
        with_bm25 = ComposedRetriever(semantic, keyword).invoke("what does error 429 mean")
        self.assertNotIn("c3", [d.metadata["chunk_id"] for d in without])
        ids = [d.metadata["chunk_id"] for d in with_bm25]
        self.assertIn("c3", ids[:2])  # the keyword list's top hit ties the semantic top hit, which wins ties

    def test_dedup_still_runs_after_fusion(self):
        dup = doc("c8", CORPUS[2].page_content + " again")
        retriever = ComposedRetriever(lambda q: [CORPUS[2], dup], KeywordIndex(CORPUS + [dup]).search)
        ids = [d.metadata["chunk_id"] for d in retriever.invoke("rate limit 429")]
        self.assertEqual(ids.count("c3") + ids.count("c8"), 1)

    def test_chunk_ids_survive_the_chroma_round_trip_so_fusion_dedups_by_id(self):
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from document_processor import build_vector_store, delete_vector_store, tag_chunks
        chunks = tag_chunks([Document(page_content=d.page_content, metadata={"type": "text", "page_no": i + 1}) for i, d in enumerate(CORPUS)])
        store = build_vector_store(chunks, DeterministicFakeEmbedding(size=16))
        try:
            docs = make_retriever(store, chunks).invoke("what does error 429 mean")
            ids = [d.metadata["chunk_id"] for d in docs]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(d.metadata["page_no"] in range(1, 6) for d in docs))
            self.assertIn("c3", ids[:2])  # the keyword hit for 429 survives fusion with Chroma's own results
        finally:
            delete_vector_store(store)

    def test_make_retriever_wires_the_keyword_index_from_the_chunks(self):
        asked = {}

        class Store:
            def similarity_search(self, question, k):
                asked["k"] = k
                return [CORPUS[0]]
        retriever = make_retriever(Store(), CORPUS)
        self.assertIn("c3", [d.metadata["chunk_id"] for d in retriever.invoke("error 429")][:2])
        self.assertEqual(asked["k"], CANDIDATE_K)  # the larger candidate set is really requested
        self.assertIsNone(make_retriever(Store()).keyword_search)


if __name__ == "__main__":
    unittest.main()
