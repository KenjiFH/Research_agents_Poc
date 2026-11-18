import json
import os
import unittest
from unittest import mock

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agents

RUN_ID = "22222222-2222-2222-2222-222222222222"


def verdicts(faithfulness, relevance="pass"):
    return {"faithfulness": faithfulness, "relevance": relevance, "faithfulness_reasoning": "", "relevance_reasoning": ""}


def reply(answer, quote="The sky is blue."):
    return json.dumps({"answer": answer, "citations": [{"chunk_id": "c1", "quote": quote}]})


class GateDecisionTests(unittest.TestCase):
    def decide(self, faithfulness, retries, gate="true"):
        with mock.patch.dict(os.environ, {"GATE_ENABLED": gate}):
            return agents.gate_decision(faithfulness, retries)

    def test_fail_retries_once_then_abstains(self):
        self.assertEqual(self.decide("fail", 0), "retry")
        self.assertEqual(self.decide("fail", 1), "abstain")

    def test_pass_and_error_never_retry_or_abstain(self):
        for verdict in ("pass", "error", ""):
            self.assertEqual(self.decide(verdict, 0), "end")
            self.assertEqual(self.decide(verdict, 1), "end")

    def test_gate_is_off_by_default_and_when_disabled(self):
        self.assertEqual(self.decide("fail", 0, gate=""), "end")
        self.assertEqual(self.decide("fail", 1, gate="false"), "end")


class GateGraphTests(unittest.TestCase):
    def _run(self, replies, judge_results, gate="true", tracing="false"):
        class Retriever:
            def invoke(self, question):
                return [Document(page_content="The sky is blue.", metadata={"chunk_id": "c1", "page_no": 4})]

        fake = FakeListChatModel(responses=replies)
        env = {"GATE_ENABLED": gate, "LANGCHAIN_TRACING_V2": tracing}
        with mock.patch.object(agents, "ChatOllama", side_effect=lambda **kwargs: fake), \
             mock.patch.object(agents, "judge", side_effect=judge_results) as judge, \
             mock.patch.object(agents, "ls_client") as ls_client, \
             mock.patch.dict(os.environ, env):
            workflow = agents.create_multi_agent_workflow(Retriever())
            state = workflow.invoke(
                {"question": "q", "context": "", "retrieved_chunks": [], "answer": "", "citations": [],
                 "faithfulness": "", "relevance": "", "retry_count": 0, "abstained": False, "gate": ""},
                config={"run_id": RUN_ID, "configurable": {"ls_run_id": RUN_ID}},
            )
        return state, judge, ls_client

    def test_a_pass_ends_without_retry(self):
        state, judge, _ = self._run([reply("Blue.")], [verdicts("pass")])
        self.assertEqual((state["retry_count"], state["abstained"], judge.call_count), (0, False, 1))
        self.assertEqual(state["answer"], "Blue.")

    def test_a_fail_retries_once_and_the_second_answer_wins_when_it_passes(self):
        state, judge, _ = self._run([reply("Purple."), reply("Blue.")], [verdicts("fail"), verdicts("pass")])
        self.assertEqual((state["retry_count"], state["abstained"], judge.call_count), (1, False, 2))
        self.assertEqual(state["answer"], "Blue.")
        self.assertEqual(state["faithfulness"], "pass")

    def test_two_fails_abstain_with_no_citations(self):
        state, judge, _ = self._run([reply("Purple."), reply("Green.")], [verdicts("fail"), verdicts("fail")])
        self.assertEqual((state["retry_count"], state["abstained"], judge.call_count), (1, True, 2))
        self.assertEqual(state["answer"], agents.ABSTENTION_MESSAGE)
        self.assertEqual(state["citations"], [])
        self.assertEqual(state["faithfulness"], "fail")
        self.assertEqual(state["relevance"], "")  # the withheld answer's relevance no longer applies

    def test_retry_is_capped_at_one(self):
        _, judge, _ = self._run([reply("a"), reply("b"), reply("c")], [verdicts("fail")] * 3)
        self.assertEqual(judge.call_count, 2)

    def test_an_error_verdict_neither_retries_nor_abstains(self):
        state, judge, _ = self._run([reply("Blue.")], [verdicts("error")])
        self.assertEqual((state["retry_count"], state["abstained"], judge.call_count), (0, False, 1))
        self.assertEqual(state["answer"], "Blue.")

    def test_a_judge_error_after_the_retry_lets_the_retried_answer_through(self):
        state, judge, _ = self._run([reply("a"), reply("Blue.")], [verdicts("fail"), verdicts("error")])
        self.assertEqual((state["retry_count"], state["abstained"], judge.call_count), (1, False, 2))
        self.assertEqual((state["answer"], state["faithfulness"]), ("Blue.", "error"))

    def test_gate_off_leaves_a_failed_answer_alone(self):
        state, judge, _ = self._run([reply("Purple.")], [verdicts("fail")], gate="")
        self.assertEqual((state["retry_count"], state["abstained"], judge.call_count), (0, False, 1))
        self.assertEqual(state["answer"], "Purple.")

    def test_langsmith_receives_only_the_final_verdict(self):
        _, _, ls_client = self._run([reply("a"), reply("b")], [verdicts("fail"), verdicts("pass")], tracing="true")
        sent = [(c.kwargs["key"], c.kwargs["score"]) for c in ls_client.create_feedback.call_args_list]
        self.assertEqual(sent, [("faithfulness", 1), ("relevance", 1)])

    def test_langsmith_gets_only_the_fail_when_the_answer_is_abstained(self):
        _, _, ls_client = self._run([reply("a"), reply("b")], [verdicts("fail"), verdicts("fail")], tracing="true")
        sent = [(c.kwargs["key"], c.kwargs["score"]) for c in ls_client.create_feedback.call_args_list]
        self.assertEqual(sent, [("faithfulness", 0)])


if __name__ == "__main__":
    unittest.main()
