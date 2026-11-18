import importlib
import json
import os
import unittest
from unittest import mock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import agents


class AnswerModelSettingTests(unittest.TestCase):
    def reload_with(self, **env):
        with mock.patch.dict(os.environ, env, clear=False):
            importlib.reload(agents)
            return agents.ANSWER_MODEL, agents.JUDGE_MODEL

    def tearDown(self):
        importlib.reload(agents)

    def test_defaults_are_separate_models(self):
        with mock.patch.dict(os.environ, {"ANSWER_MODEL": "", "JUDGE_MODEL": ""}):
            importlib.reload(agents)
            self.assertEqual((agents.ANSWER_MODEL, agents.JUDGE_MODEL), ("llama3", "qwen2.5:14b"))
            self.assertIsNone(agents.get_judge_warning())

    def test_answer_model_is_set_apart_from_the_judge_by_the_environment(self):
        answer, judge = self.reload_with(ANSWER_MODEL=" qwen3.6:latest ", JUDGE_MODEL="qwen2.5:14b")
        self.assertEqual((answer, judge), ("qwen3.6:latest", "qwen2.5:14b"))

    def test_the_equal_model_warning_follows_the_answer_model_setting(self):
        self.reload_with(ANSWER_MODEL="qwen2.5:14b", JUDGE_MODEL="qwen2.5:14b")
        self.assertIn("self-grading", agents.get_judge_warning())


class SynthesisSchemaTests(unittest.TestCase):
    def test_schema_requires_an_answer_and_citations_with_chunk_id_and_quote(self):
        schema = agents.SYNTHESIS_SCHEMA
        self.assertEqual(schema["required"], ["answer", "citations"])
        item = schema["properties"]["citations"]["items"]
        self.assertEqual(sorted(item["required"]), ["chunk_id", "quote"])
        self.assertIn("complete sentence", schema["properties"]["answer"]["description"])


class SchemaAcceptedTests(unittest.TestCase):
    def test_langchain_ollama_accepts_the_schema_as_its_format(self):
        model = agents.ChatOllama(model="any-model", temperature=0.2, format=agents.SYNTHESIS_SCHEMA)
        self.assertEqual(model.format, agents.SYNTHESIS_SCHEMA)


class PromptTests(unittest.TestCase):
    def test_prompt_asks_for_complete_sentences_and_requires_citations(self):
        prompt = agents.SYNTHESIS_PROMPT
        for expected in ("complete sentences", "cite at least one passage", "verbatim", "{context}", "{question}"):
            self.assertIn(expected, prompt)

    def test_prompt_has_no_domain_specific_example_to_echo(self):
        for leaked in ("leave", "PTO", "vacation", "10 days"):
            self.assertNotIn(leaked, agents.SYNTHESIS_PROMPT)

    def test_warning_names_both_settings(self):
        with mock.patch.object(agents, "JUDGE_MODEL", agents.ANSWER_MODEL):
            message = agents.get_judge_warning()
        self.assertIn("JUDGE_MODEL", message)
        self.assertIn("ANSWER_MODEL", message)


class SynthesizerWiringTests(unittest.TestCase):
    def test_the_synthesizer_uses_the_schema_and_asks_for_complete_sentences(self):
        prompts, built = [], []

        def fake_llm(prompt_value):
            prompts.append(prompt_value.to_string())
            return AIMessage(content=json.dumps({"answer": "Yes, up to 10 days carry over.", "citations": []}))

        def fake_chat(**kwargs):
            built.append(kwargs)
            return RunnableLambda(fake_llm)

        class Retriever:
            def invoke(self, question):
                return [Document(page_content="Unused leave up to 10 days carries over.", metadata={"chunk_id": "c1", "page_no": 2})]

        verdicts = {"faithfulness": "pass", "relevance": "pass", "faithfulness_reasoning": "", "relevance_reasoning": ""}
        with mock.patch.object(agents, "ChatOllama", side_effect=fake_chat), \
             mock.patch.object(agents, "judge", return_value=verdicts), \
             mock.patch.object(agents, "ls_client"):
            state = agents.create_multi_agent_workflow(Retriever()).invoke(
                {"question": "Can I roll over PTO?", "context": "", "retrieved_chunks": [], "answer": "", "citations": [],
                 "faithfulness": "", "relevance": "", "retry_count": 0, "abstained": False, "gate": ""})

        schema_calls = [k for k in built if k.get("format") == agents.SYNTHESIS_SCHEMA]
        self.assertEqual(len(schema_calls), 1)
        self.assertEqual(schema_calls[0]["model"], agents.ANSWER_MODEL)
        self.assertIsInstance(schema_calls[0]["format"], dict)  # a schema, not loose "json" mode
        self.assertIn("complete sentences", prompts[0])
        self.assertIn("[c1] (page 2)", prompts[0])
        self.assertIn("Can I roll over PTO?", prompts[0])
        self.assertEqual(state["answer"], "Yes, up to 10 days carry over.")


if __name__ == "__main__":
    unittest.main()
