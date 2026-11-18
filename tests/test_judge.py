import unittest
from unittest import mock

import ollama

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agents


class JudgeTests(unittest.TestCase):
    def test_judge_uses_the_supplied_llm_and_maps_scores(self):
        llm = FakeListChatModel(responses=["The answer sticks to the context.\nY", "It answers the question.\nN"])
        result = agents.judge("q?", "a", "ctx", llm=llm)
        self.assertEqual(result["faithfulness"], "pass")
        self.assertEqual(result["relevance"], "fail")
        self.assertIn("context", result["faithfulness_reasoning"])

    def test_an_unparseable_score_is_an_error_verdict_not_a_fail(self):
        with mock.patch.object(agents, "load_evaluator") as load:
            load.return_value.evaluate_strings.return_value = {"score": None, "reasoning": "unclear"}
            result = agents.judge("q", "a", "c", llm=object())
        self.assertEqual((result["faithfulness"], result["relevance"]), ("error", "error"))

    def test_verdict_mapping(self):
        self.assertEqual([agents.verdict_from_score(v) for v in (1, 0, None, 0.5)], ["pass", "fail", "error", "error"])
        self.assertEqual([agents.verdict_to_score(v) for v in ("pass", "fail", "error", "")], [1, 0, None, None])

    def test_default_judge_builds_the_judge_model_not_the_answer_model(self):
        with mock.patch.object(agents, "ChatOllama") as chat, \
             mock.patch.object(agents, "load_evaluator") as load:
            load.return_value.evaluate_strings.return_value = {"score": 1, "reasoning": "r"}
            agents.judge("q", "a", "c")
        chat.assert_called_once()
        self.assertEqual(chat.call_args.kwargs["model"], agents.JUDGE_MODEL)

    def test_warning_only_when_judge_equals_answerer(self):
        with mock.patch.object(agents, "JUDGE_MODEL", agents.ANSWER_MODEL):
            self.assertIn("self-grading", agents.get_judge_warning())
        with mock.patch.object(agents, "JUDGE_MODEL", "some-other-model"):
            self.assertIsNone(agents.get_judge_warning())

    def test_unpulled_judge_model_raises_a_friendly_error(self):
        with mock.patch.object(agents, "load_evaluator") as load:
            load.return_value.evaluate_strings.side_effect = ollama.ResponseError("model not found", 404)
            with self.assertRaises(agents.JudgeModelNotFound) as caught:
                agents.judge("q", "a", "c")
        self.assertIn("ollama pull", str(caught.exception))

    def test_other_ollama_errors_are_not_swallowed(self):
        with mock.patch.object(agents, "load_evaluator") as load:
            load.return_value.evaluate_strings.side_effect = ollama.ResponseError("boom", 500)
            with self.assertRaises(ollama.ResponseError):
                agents.judge("q", "a", "c")

    def test_equal_models_are_detected_despite_case_and_latest_tag(self):
        for variant in ("llama3:latest", "Llama3", " llama3 "):
            with mock.patch.object(agents, "JUDGE_MODEL", variant):
                self.assertIsNotNone(agents.get_judge_warning(), variant)

    def test_judge_and_answer_models_differ_by_default(self):
        self.assertNotEqual(agents.JUDGE_MODEL, agents.ANSWER_MODEL)


if __name__ == "__main__":
    unittest.main()
