import json
import unittest
from unittest import mock

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agents


def doc(chunk_id, page, content):
    return Document(page_content=content, metadata={"type": "text", "chunk_id": chunk_id, "page_no": page})


class FormatContextTests(unittest.TestCase):
    def test_numbers_context_by_chunk_id_and_keeps_pages(self):
        context, retrieved = agents.format_context([doc("c3", 2, "Alpha text."), doc("c7", 5, "Beta text.")])
        self.assertIn("[c3] (page 2)\nAlpha text.", context)
        self.assertIn("[c7] (page 5)\nBeta text.", context)
        self.assertEqual(retrieved[1], {"chunk_id": "c7", "page_no": 5, "content": "Beta text."})

    def test_falls_back_to_position_when_a_chunk_has_no_id(self):
        _, retrieved = agents.format_context([Document(page_content="x", metadata={})])
        self.assertEqual(retrieved[0]["chunk_id"], "r1")


class MissingPageTests(unittest.TestCase):
    def test_context_omits_the_page_label_when_there_is_no_page(self):
        context, retrieved = agents.format_context([Document(page_content="x", metadata={"chunk_id": "c1"})])
        self.assertTrue(context.startswith("[c1]\nx"))
        self.assertIsNone(retrieved[0]["page_no"])


class ParseSynthesisTests(unittest.TestCase):
    RETRIEVED = [
        {"chunk_id": "c1", "page_no": 4, "content": "The sky is blue."},
        {"chunk_id": "c2", "page_no": 9, "content": "Grass is green."},
    ]

    def test_returns_answer_and_citations_with_pages_from_the_chunks(self):
        raw = json.dumps({"answer": "Blue.", "citations": [{"chunk_id": "c1", "quote": "The sky is blue."}]})
        answer, citations = agents.parse_synthesis(raw, self.RETRIEVED)
        self.assertEqual(answer, "Blue.")
        self.assertEqual(citations, [{"chunk_id": "c1", "page_no": 4, "quote": "The sky is blue."}])

    def test_model_supplied_page_is_ignored(self):
        raw = json.dumps({"answer": "x", "citations": [{"chunk_id": "c2", "page_no": 99, "quote": "Grass is green."}]})
        _, citations = agents.parse_synthesis(raw, self.RETRIEVED)
        self.assertEqual(citations[0]["page_no"], 9)

    def test_unknown_chunk_and_empty_quote_are_dropped(self):
        raw = json.dumps({"answer": "x", "citations": [
            {"chunk_id": "c99", "quote": "q"}, {"chunk_id": "c1", "quote": " "}, "junk",
        ]})
        self.assertEqual(agents.parse_synthesis(raw, self.RETRIEVED)[1], [])

    def test_plain_text_reply_becomes_the_answer_with_no_citations(self):
        self.assertEqual(agents.parse_synthesis("I do not know.", self.RETRIEVED), ("I do not know.", []))

    def test_json_without_an_answer_never_shows_raw_json(self):
        for raw in (json.dumps({"citations": []}), json.dumps({"answer": "", "citations": []})):
            self.assertEqual(agents.parse_synthesis(raw, self.RETRIEVED), ("I do not know.", []))


class GraphTests(unittest.TestCase):
    """The whole graph with the LLMs, the judge and LangSmith replaced."""

    def _run(self, reply, judge_result):
        class Retriever:
            def invoke(self, question):
                return [doc("c1", 4, "The sky is blue."), doc("c2", 9, "Grass is green.")]

        fake = FakeListChatModel(responses=[reply])
        with mock.patch.object(agents, "ChatOllama", side_effect=lambda **kwargs: fake), \
             mock.patch.object(agents, "judge", return_value=judge_result) as judge, \
             mock.patch.object(agents, "ls_client"):
            workflow = agents.create_multi_agent_workflow(Retriever())
            state = workflow.invoke({
                "question": "What colour is the sky?", "context": "", "retrieved_chunks": [],
                "answer": "", "citations": [], "faithfulness_score": 0, "relevance_score": 0,
            })
        return state, judge

    def test_final_state_holds_citations_and_the_judge_sees_the_numbered_context(self):
        reply = json.dumps({"answer": "Blue.", "citations": [{"chunk_id": "c1", "quote": "The sky is blue."}]})
        scores = {"faithfulness_score": 100, "relevance_score": 100, "faithfulness_reasoning": "", "relevance_reasoning": ""}
        state, judge = self._run(reply, scores)
        self.assertEqual(state["answer"], "Blue.")
        self.assertEqual(state["citations"], [{"chunk_id": "c1", "page_no": 4, "quote": "The sky is blue."}])
        self.assertEqual((state["faithfulness_score"], state["relevance_score"]), (100, 100))
        args = judge.call_args.args
        self.assertEqual(args[0], "What colour is the sky?")
        self.assertEqual(args[1], "Blue.")
        self.assertIn("[c1] (page 4)", args[2])

    def test_a_missing_score_still_becomes_zero_until_story_4_1(self):
        reply = json.dumps({"answer": "Blue.", "citations": []})
        scores = {"faithfulness_score": None, "relevance_score": None, "faithfulness_reasoning": "", "relevance_reasoning": ""}
        state, _ = self._run(reply, scores)
        self.assertEqual((state["faithfulness_score"], state["relevance_score"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
