import json
import os
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
        self.assertEqual(retrieved[1], {"chunk_id": "c7", "page_no": 5, "type": "text", "content": "Beta text."})

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


class VerifyCitationsTests(unittest.TestCase):
    RETRIEVED = [{"chunk_id": "c1", "page_no": 4, "content": "The sky is\nblue on clear days."}]

    def cite(self, quote, chunk_id="c1"):
        return {"chunk_id": chunk_id, "page_no": 4, "quote": quote}

    def test_verbatim_quote_stays(self):
        self.assertEqual(agents.verify_citations([self.cite("blue on clear days")], self.RETRIEVED),
                         [self.cite("blue on clear days")])

    def test_quote_across_a_line_break_still_verifies(self):
        self.assertEqual(len(agents.verify_citations([self.cite("The sky is blue")], self.RETRIEVED)), 1)

    def test_altered_quote_is_dropped(self):
        self.assertEqual(agents.verify_citations([self.cite("The sky is green")], self.RETRIEVED), [])

    def test_case_change_is_dropped(self):
        self.assertEqual(agents.verify_citations([self.cite("the sky is blue")], self.RETRIEVED), [])

    def test_quote_from_a_different_chunk_is_dropped(self):
        retrieved = self.RETRIEVED + [{"chunk_id": "c2", "page_no": 5, "content": "Grass is green."}]
        self.assertEqual(agents.verify_citations([self.cite("Grass is green.", "c1")], retrieved), [])

    def test_unknown_chunk_and_empty_quote_are_dropped(self):
        self.assertEqual(agents.verify_citations([self.cite("x", "c9"), self.cite("  ")], self.RETRIEVED), [])


RUN_ID = "11111111-1111-1111-1111-111111111111"
PASS = {"faithfulness": "pass", "relevance": "pass", "faithfulness_reasoning": "", "relevance_reasoning": ""}


class GraphTests(unittest.TestCase):
    """The whole graph with the LLMs, the judge and LangSmith replaced."""

    tracing = "true"

    def _run(self, reply, judge_result, config=None, judge_side_effect=None):
        class Retriever:
            def invoke(self, question):
                return [doc("c1", 4, "The sky is blue."), doc("c2", 9, "Grass is green.")]

        fake = FakeListChatModel(responses=[reply])
        with mock.patch.object(agents, "ChatOllama", side_effect=lambda **kwargs: fake), \
             mock.patch.object(agents, "judge", return_value=judge_result, side_effect=judge_side_effect) as judge, \
             mock.patch.object(agents, "ls_client") as ls_client, \
             mock.patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": self.tracing}):
            self.ls_client = ls_client
            workflow = agents.create_multi_agent_workflow(Retriever())
            state = workflow.invoke({
                "question": "What colour is the sky?", "context": "", "retrieved_chunks": [],
                "answer": "", "citations": [], "faithfulness": "", "relevance": "",
            }, config=config)
        return state, judge

    def test_final_state_holds_citations_and_the_judge_sees_the_numbered_context(self):
        reply = json.dumps({"answer": "Blue.", "citations": [{"chunk_id": "c1", "quote": "The sky is blue."}]})
        scores = PASS
        state, judge = self._run(reply, scores)
        self.assertEqual(state["answer"], "Blue.")
        self.assertEqual(state["citations"], [{"chunk_id": "c1", "page_no": 4, "quote": "The sky is blue."}])
        self.assertEqual((state["faithfulness"], state["relevance"]), ("pass", "pass"))
        args = judge.call_args.args
        self.assertEqual(args[0], "What colour is the sky?")
        self.assertEqual(args[1], "Blue.")
        self.assertIn("[c1] (page 4)", args[2])

    def test_an_altered_quote_is_removed_from_the_final_state(self):
        reply = json.dumps({"answer": "Blue.", "citations": [
            {"chunk_id": "c1", "quote": "The sky is blue."}, {"chunk_id": "c1", "quote": "The sky is purple."},
        ]})
        scores = PASS
        state, _ = self._run(reply, scores)
        self.assertEqual([c["quote"] for c in state["citations"]], ["The sky is blue."])

    def test_a_failed_evaluator_is_an_error_verdict(self):
        reply = json.dumps({"answer": "Blue.", "citations": []})
        state, _ = self._run(reply, None, judge_side_effect=RuntimeError("connection refused"))
        self.assertEqual((state["faithfulness"], state["relevance"]), ("error", "error"))

    def test_a_missing_judge_model_is_not_swallowed(self):
        reply = json.dumps({"answer": "Blue.", "citations": []})
        with self.assertRaises(agents.JudgeModelNotFound):
            self._run(reply, None, judge_side_effect=agents.JudgeModelNotFound("ollama pull x"))

    def test_no_feedback_is_attempted_when_tracing_is_off(self):
        reply = json.dumps({"answer": "Blue.", "citations": []})
        self.tracing = "false"
        try:
            self._run(reply, PASS, config={"run_id": RUN_ID, "configurable": {"ls_run_id": RUN_ID}})
        finally:
            self.tracing = "true"
        self.ls_client.create_feedback.assert_not_called()

    def test_langsmith_gets_the_same_verdicts_and_nothing_for_an_error(self):
        reply = json.dumps({"answer": "Blue.", "citations": []})
        verdicts = {"faithfulness": "pass", "relevance": "fail", "faithfulness_reasoning": "", "relevance_reasoning": ""}
        self._run(reply, verdicts, config={"run_id": RUN_ID, "configurable": {"ls_run_id": RUN_ID}})
        sent = {c.kwargs["key"]: c.kwargs["score"] for c in self.ls_client.create_feedback.call_args_list}
        self.assertEqual(sent, {"faithfulness": 1, "relevance": 0})
        self.assertEqual(self.ls_client.create_feedback.call_args.args[0], RUN_ID)

        errors = {"faithfulness": "error", "relevance": "pass", "faithfulness_reasoning": "", "relevance_reasoning": ""}
        self._run(reply, errors, config={"run_id": RUN_ID, "configurable": {"ls_run_id": RUN_ID}})
        sent = {c.kwargs["key"]: c.kwargs["score"] for c in self.ls_client.create_feedback.call_args_list}
        self.assertEqual(sent, {"relevance": 1})


if __name__ == "__main__":
    unittest.main()
