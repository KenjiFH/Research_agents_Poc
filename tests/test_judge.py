import json
import unittest
from unittest import mock

import ollama
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agents

CONTEXT = "[c1] (page 1)\nThe warranty covers manufacturing defects for 24 months from purchase. Water damage is excluded."


def claims_reply(*claims):
    return json.dumps({"claims": list(claims)})


def support_reply(*entries):
    return json.dumps({"results": [
        {"claim": i, "verdict": v, "evidence": e, "reason": r} for i, (v, e, r) in enumerate(entries, start=1)]})


def relevance_reply(addresses="yes", asked="The warranty length.", reason="It gives the length."):
    return json.dumps({"asked": asked, "addresses": addresses, "reason": reason})


EVIDENCE = "The warranty covers manufacturing defects for 24 months from purchase."


def scripted(*replies):
    """A fake judge model that answers the three stages in order: claims, support, relevance."""
    return FakeListChatModel(responses=list(replies))


class StagedJudgeTests(unittest.TestCase):
    def test_every_claim_supported_with_verified_evidence_passes(self):
        llm = scripted(claims_reply("The warranty lasts 24 months."),
                       support_reply(("supported", EVIDENCE, "Stated directly.")), relevance_reply())
        result = agents.judge("How long is the warranty?", "It lasts 24 months.", CONTEXT, llm=llm)
        self.assertEqual((result["faithfulness"], result["relevance"]), ("pass", "pass"))
        self.assertEqual(result["claims"][0]["verdict"], "supported")
        self.assertIn("1 of 1 claims supported", result["faithfulness_reasoning"])

    def test_one_unsupported_claim_fails_the_whole_answer(self):
        llm = scripted(claims_reply("The warranty lasts 24 months.", "It also covers water damage."),
                       support_reply(("supported", EVIDENCE, "ok"), ("unsupported", "", "Water damage is excluded.")),
                       relevance_reply())
        result = agents.judge("What does the warranty cover?", "24 months, and water damage.", CONTEXT, llm=llm)
        self.assertEqual(result["faithfulness"], "fail")
        self.assertIn("1 of 2 claims supported", result["faithfulness_reasoning"])
        self.assertIn("water damage", result["faithfulness_reasoning"].lower())

    def test_evidence_that_is_not_in_the_context_downgrades_a_supported_claim(self):
        llm = scripted(claims_reply("The warranty covers cosmetic scratches."),
                       support_reply(("supported", "The warranty covers cosmetic scratches.", "Stated.")), relevance_reply())
        result = agents.judge("What does it cover?", "It covers cosmetic scratches.", CONTEXT, llm=llm)
        self.assertEqual(result["faithfulness"], "fail")
        self.assertEqual(result["claims"][0]["reason"], "the quoted evidence is not in the context")

    def test_evidence_spanning_a_line_break_still_verifies(self):
        context = "The warranty covers\nmanufacturing defects for 24 months."
        llm = scripted(claims_reply("The warranty lasts 24 months."), support_reply(("supported", "The warranty covers manufacturing defects", "ok")), relevance_reply())
        self.assertEqual(agents.judge("q", "a", context, llm=llm)["faithfulness"], "pass")

    def test_relevance_passes_only_on_a_clear_yes(self):
        for addresses, expected in (("yes", "pass"), ("partly", "fail"), ("no", "fail")):
            llm = scripted(claims_reply("The warranty lasts 24 months."), support_reply(("supported", EVIDENCE, "ok")), relevance_reply(addresses))
            self.assertEqual(agents.judge("q", "a", CONTEXT, llm=llm)["relevance"], expected, addresses)

    def test_relevance_reasoning_records_what_was_asked(self):
        llm = scripted(claims_reply("The warranty lasts 24 months."), support_reply(("supported", EVIDENCE, "ok")), relevance_reply(asked="How long the warranty lasts."))
        self.assertIn("How long the warranty lasts.", agents.judge("q", "a", CONTEXT, llm=llm)["relevance_reasoning"])

    def test_an_unusable_reply_is_an_error_verdict_for_that_metric_only(self):
        llm = scripted("not json at all", relevance_reply())  # the support stage is skipped after a failure
        result = agents.judge("q", "a", CONTEXT, llm=llm)
        self.assertEqual((result["faithfulness"], result["relevance"]), ("error", "pass"))
        self.assertIn("not valid JSON", result["faithfulness_reasoning"])

    def test_a_support_reply_that_skips_a_claim_is_an_error_when_nothing_else_failed(self):
        llm = scripted(claims_reply("The warranty lasts 24 months.", "The warranty covers defects."), support_reply(("supported", EVIDENCE, "ok")), relevance_reply())
        result = agents.judge("q", "a", CONTEXT, llm=llm)
        self.assertEqual(result["faithfulness"], "error")
        self.assertIn("skipped 1 claim", result["faithfulness_reasoning"])

    def test_a_bad_relevance_verdict_is_an_error(self):
        llm = scripted(claims_reply("The warranty lasts 24 months."), support_reply(("supported", EVIDENCE, "ok")), json.dumps({"asked": "x", "addresses": "maybe", "reason": "y"}))
        result = agents.judge("q", "a", CONTEXT, llm=llm)
        self.assertEqual((result["faithfulness"], result["relevance"]), ("pass", "error"))

    def test_no_listed_claims_falls_back_to_checking_the_whole_answer(self):
        llm = scripted(claims_reply(), support_reply(("supported", EVIDENCE, "ok")), relevance_reply())
        result = agents.judge("q", "The warranty is 24 months.", CONTEXT, llm=llm)
        self.assertEqual(result["claims"][0]["claim"], "The warranty is 24 months.")


class StageFailureCombinationTests(unittest.TestCase):
    def test_a_known_unsupported_claim_outranks_a_skipped_one(self):
        llm = scripted(claims_reply("The warranty lasts 24 months.", "It covers water damage.", "It is transferable."),
                       support_reply(("supported", EVIDENCE, "ok"), ("unsupported", "", "Water damage is excluded.")),  # claim 3 skipped
                       relevance_reply())
        result = agents.judge("What does it cover?", "x", CONTEXT, llm=llm)
        self.assertEqual(result["faithfulness"], "fail")
        self.assertEqual([c["verdict"] for c in result["claims"]], ["supported", "unsupported", "unchecked"])

    def test_a_claim_number_given_as_a_string_still_counts(self):
        reply = json.dumps({"results": [{"claim": "1", "verdict": "supported", "evidence": EVIDENCE, "reason": "ok"}]})
        llm = scripted(claims_reply("The warranty lasts 24 months."), reply, relevance_reply())
        self.assertEqual(agents.judge("q", "The warranty lasts 24 months.", CONTEXT, llm=llm)["faithfulness"], "pass")

    def test_more_claims_than_the_limit_is_an_error_not_a_silent_pass(self):
        import judging
        many = [f"The warranty lasts {i} months." for i in range(judging.CLAIM_LIMIT + 1)]
        llm = scripted(claims_reply(*many), relevance_reply())
        result = agents.judge("q", "answer", CONTEXT, llm=llm)
        self.assertEqual(result["faithfulness"], "error")
        self.assertIn("more than the", result["faithfulness_reasoning"])


class EvidenceVerificationTests(unittest.TestCase):
    def verdict(self, evidence, claim="The warranty lasts 24 months."):
        llm = scripted(claims_reply(claim), support_reply(("supported", evidence, "ok")), relevance_reply())
        return agents.judge("q", claim, CONTEXT, llm=llm)["claims"][0]

    def test_trivial_evidence_does_not_count(self):
        self.assertEqual(self.verdict("The")["verdict"], "unsupported")
        self.assertEqual(self.verdict("24")["verdict"], "unsupported")  # one word

    def test_the_context_header_lines_are_not_evidence(self):
        self.assertEqual(self.verdict("[c1]")["verdict"], "unsupported")
        self.assertEqual(self.verdict("(page 1)")["verdict"], "unsupported")
        self.assertEqual(self.verdict("[c1] (page 1)")["verdict"], "unsupported")

    def test_evidence_must_relate_to_the_claim(self):
        claim = self.verdict("Water damage is excluded.", claim="The warranty lasts 24 months.")
        self.assertEqual(claim["verdict"], "unsupported")
        self.assertIn("does not relate", claim["reason"])

    def test_smart_quotes_case_markdown_and_table_pipes_do_not_cause_a_wrong_fail(self):
        context = "| Plan | Detail |\n|---|---|\n| Basic | The library's hours are 9 to 5. |"
        llm = scripted(claims_reply("The library hours are 9 to 5."),
                       support_reply(("supported", "the library\u2019s HOURS are 9 to 5.", "ok")), relevance_reply())
        self.assertEqual(agents.judge("q", "The library hours are 9 to 5.", context, llm=llm)["faithfulness"], "pass")

    def test_two_passages_joined_by_an_ellipsis_both_verify(self):
        context = "The shop opens at 9 am. We sell bread. It closes at 5 pm."
        evidence = "The shop opens at 9 am ... It closes at 5 pm."
        llm = scripted(claims_reply("The shop opens at 9 am and closes at 5 pm."), support_reply(("supported", evidence, "ok")), relevance_reply())
        self.assertEqual(agents.judge("q", "The shop opens at 9 am and closes at 5 pm.", context, llm=llm)["faithfulness"], "pass")

    def test_an_invented_second_passage_fails_even_when_the_first_is_real(self):
        context = "The shop opens at 9 am. We sell bread."
        llm = scripted(claims_reply("The shop opens at 9 am and sells cake."),
                       support_reply(("supported", "The shop opens at 9 am ... We sell cake.", "ok")), relevance_reply())
        self.assertEqual(agents.judge("q", "The shop opens at 9 am and sells cake.", context, llm=llm)["faithfulness"], "fail")


class WindowTests(unittest.TestCase):
    def test_a_prompt_too_large_for_the_judge_window_is_an_error_not_a_wrong_fail(self):
        big_context = "The shop opens at nine. " * 400
        llm = scripted(claims_reply("The shop opens at nine."), relevance_reply())
        with mock.patch.object(agents, "JUDGE_NUM_CTX", 1000):
            result = agents.judge("q", "The shop opens at nine.", big_context, llm=llm)
        self.assertEqual(result["faithfulness"], "error")
        self.assertIn("too large for the 1000-token judge window", result["faithfulness_reasoning"])
        self.assertEqual(result["relevance"], "pass")  # relevance does not need the context

    def test_a_normal_prompt_fits(self):
        llm = scripted(claims_reply("The warranty lasts 24 months."), support_reply(("supported", EVIDENCE, "ok")), relevance_reply())
        self.assertEqual(agents.judge("q", "The warranty lasts 24 months.", CONTEXT, llm=llm)["faithfulness"], "pass")

    def test_a_bad_window_setting_falls_back_instead_of_crashing_on_import(self):
        import os
        with mock.patch.dict(os.environ, {"JUDGE_NUM_CTX": "lots"}):
            self.assertEqual(agents._read_int("JUDGE_NUM_CTX", 8192), 8192)
        with mock.patch.dict(os.environ, {"JUDGE_NUM_CTX": " 16384 "}):
            self.assertEqual(agents._read_int("JUDGE_NUM_CTX", 8192), 16384)


class DeterministicShortcutTests(unittest.TestCase):
    def test_an_abstention_is_graded_without_calling_the_model(self):
        llm = mock.Mock()
        result = agents.judge("What GPU is needed?", "I do not know; the context does not say.", CONTEXT, llm=llm)
        llm.invoke.assert_not_called()
        self.assertEqual((result["faithfulness"], result["relevance"]), ("pass", "fail"))
        self.assertTrue(result["checks"]["abstention"])

    def test_a_number_found_nowhere_in_the_context_fails_faithfulness_without_the_faithfulness_stages(self):
        prompts = []

        class Recorder:
            def __init__(self, replies):
                self.replies = iter(replies)

            def invoke(self, prompt):
                prompts.append(prompt)
                return mock.Mock(content=next(self.replies))

        result = agents.judge("How long is the warranty?", "The warranty lasts 36 months.", CONTEXT, llm=Recorder([relevance_reply()]))
        self.assertEqual(result["faithfulness"], "fail")
        self.assertIn("36", result["faithfulness_reasoning"])
        self.assertIn("without the model", result["faithfulness_reasoning"])
        self.assertEqual(result["checks"]["unverified_numbers"], ["36"])
        self.assertEqual(len(prompts), 1)  # only the relevance stage ran
        self.assertIn("Question:", prompts[0])
        self.assertEqual(result["relevance"], "pass")

    def test_unverified_names_reach_the_claim_check_prompt_as_a_hint(self):
        prompts = []

        class Recorder:
            def __init__(self, replies):
                self.replies = iter(replies)

            def invoke(self, prompt):
                prompts.append(prompt)
                return mock.Mock(content=next(self.replies))

        llm = Recorder([claims_reply("The plant in Boston honours the warranty."),
                        support_reply(("unsupported", "", "Boston is not mentioned.")), relevance_reply()])
        result = agents.judge("Where?", "The plant in Boston honours the warranty.", CONTEXT, llm=llm)
        self.assertEqual(result["checks"]["unverified_names"], ["Boston"])
        self.assertIn("these names appear in the answer but nowhere in the context", prompts[1])
        self.assertEqual(result["faithfulness"], "fail")

    def test_the_no_verified_citation_flag_is_informational_only(self):
        llm = scripted(claims_reply("The warranty lasts 24 months."), support_reply(("supported", EVIDENCE, "ok")), relevance_reply())
        result = agents.judge("q", "The warranty is 24 months.", CONTEXT, llm=llm, citations=[])
        self.assertTrue(result["checks"]["no_verified_citation"])
        self.assertEqual(result["faithfulness"], "pass")


class PromptTests(unittest.TestCase):
    def test_every_stage_prompt_carries_worked_examples(self):
        import judging
        self.assertEqual(judging.CLAIMS_PROMPT.count("Example "), 3)
        self.assertEqual(judging.SUPPORT_PROMPT.count("Example "), 3)
        self.assertEqual(judging.RELEVANCE_PROMPT.count("Example "), 3)

    def test_the_examples_are_generic_and_the_slots_are_filled(self):
        import judging
        built = judging.build_support_prompt("CTX-TEXT", ["claim one", "claim two"])
        self.assertIn("CTX-TEXT", built)
        self.assertIn("1. claim one\n2. claim two", built)
        self.assertNotIn("<<", built)
        for prompt in (judging.CLAIMS_PROMPT, judging.SUPPORT_PROMPT, judging.RELEVANCE_PROMPT):
            for leaked in ("warranty", "PTO", "revenue"):
                self.assertNotIn(leaked, prompt)


class JudgeModelTests(unittest.TestCase):
    def test_default_judge_builds_one_model_per_stage_with_its_schema_and_deterministic_settings(self):
        import judging
        built = []

        def fake_chat(**kwargs):
            built.append(kwargs)
            return scripted(claims_reply("The warranty lasts 24 months."), support_reply(("supported", EVIDENCE, "ok")), relevance_reply())

        with mock.patch.object(agents, "ChatOllama", side_effect=fake_chat):
            agents.judge("q", "a", CONTEXT)
        self.assertEqual([k["format"] for k in built], [judging.CLAIMS_SCHEMA, judging.SUPPORT_SCHEMA, judging.RELEVANCE_SCHEMA])
        for kwargs in built:
            self.assertEqual(kwargs["model"], agents.JUDGE_MODEL)
            self.assertEqual(kwargs["temperature"], 0.0)
            self.assertEqual(kwargs["seed"], agents.MODEL_SEED)
            self.assertEqual(kwargs["num_ctx"], agents.JUDGE_NUM_CTX)

    def test_the_judge_window_is_larger_than_ollamas_default(self):
        self.assertGreater(agents.JUDGE_NUM_CTX, 4096)

    def test_unpulled_judge_model_raises_a_friendly_error(self):
        llm = mock.Mock()
        llm.invoke.side_effect = ollama.ResponseError("model not found", 404)
        with self.assertRaises(agents.JudgeModelNotFound) as caught:
            agents.judge("q", "a", CONTEXT, llm=llm)
        self.assertIn("ollama pull", str(caught.exception))

    def test_other_ollama_errors_are_not_swallowed(self):
        llm = mock.Mock()
        llm.invoke.side_effect = ollama.ResponseError("boom", 500)
        with self.assertRaises(ollama.ResponseError):
            agents.judge("q", "a", CONTEXT, llm=llm)

    def test_verdict_to_score_mapping(self):
        self.assertEqual([agents.verdict_to_score(v) for v in ("pass", "fail", "error", "")], [1, 0, None, None])

    def test_warning_only_when_judge_equals_answerer(self):
        with mock.patch.object(agents, "JUDGE_MODEL", agents.ANSWER_MODEL):
            self.assertIn("self-grading", agents.get_judge_warning())
        with mock.patch.object(agents, "JUDGE_MODEL", "some-other-model"):
            self.assertIsNone(agents.get_judge_warning())

    def test_equal_models_are_detected_despite_case_and_latest_tag(self):
        for variant in ("llama3:latest", "Llama3", " llama3 "):
            with mock.patch.object(agents, "JUDGE_MODEL", variant), mock.patch.object(agents, "ANSWER_MODEL", "llama3"):
                self.assertIsNotNone(agents.get_judge_warning(), variant)

    def test_judge_and_answer_models_differ_by_default(self):
        self.assertNotEqual(agents.JUDGE_MODEL, agents.ANSWER_MODEL)


if __name__ == "__main__":
    unittest.main()
