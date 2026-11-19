import json
import unittest

import dev_insight
import judging


class NumberTests(unittest.TestCase):
    def test_digits_commas_decimals_percent_and_money_are_normalised(self):
        self.assertEqual(judging.extract_numbers("It costs $1,200.50 or 4.50 with 12% off, 24 units, 007."),
                         {"1200.5", "4.5", "12", "24", "7"})

    def test_spelled_out_numbers_match_their_digits(self):
        self.assertEqual(judging.extract_numbers("twenty four months"), {"24"})
        self.assertEqual(judging.extract_numbers("twenty-four months"), {"24"})
        self.assertEqual(judging.extract_numbers("two hundred and five"), {"205"})
        self.assertEqual(judging.extract_numbers("three thousand five hundred"), {"3500"})
        self.assertEqual(judging.extract_numbers("seven to fourteen days"), {"7", "14"})

    def test_a_lone_one_is_not_a_quantity(self):
        self.assertEqual(judging.extract_numbers("One of the reasons is that no one knew."), set())
        self.assertEqual(judging.extract_numbers("twenty one items"), {"21"})

    def test_decimals_are_not_split_into_two_numbers(self):
        self.assertEqual(judging.extract_numbers("Section 4.2 applies"), {"4.2"})


class NameTests(unittest.TestCase):
    def test_capitalised_words_mid_sentence_are_names_but_the_first_word_is_not(self):
        self.assertEqual(judging.extract_names("Yes, the plant opened in Austin. It employs Maria."), ["Austin", "Maria"])

    def test_acronyms_count_and_duplicates_are_listed_once(self):
        self.assertEqual(judging.extract_names("The API returns HTTP errors; the API is rate limited."), ["API", "HTTP"])

    def test_plain_lowercase_text_has_none(self):
        self.assertEqual(judging.extract_names("the plant opened last year."), [])


class AbstentionTests(unittest.TestCase):
    def test_short_declines_are_abstentions(self):
        for text in ("I do not know.", "I don't know the answer.", "The context does not say.",
                     "I can't verify an answer to this question from the document.", "That is not mentioned in the document."):
            self.assertTrue(judging.is_abstention(text), text)

    def test_a_real_answer_is_not(self):
        self.assertFalse(judging.is_abstention("The warranty lasts 24 months."))

    def test_a_long_answer_that_mentions_a_gap_goes_to_the_model(self):
        long_answer = ("The warranty covers manufacturing defects, repairs and replacement parts, although the "
                       "document does not specify who pays for shipping or how long a repair takes in practice.")
        self.assertFalse(judging.is_abstention(long_answer))

    def test_a_claim_followed_by_a_decline_is_not_an_abstention(self):
        for text in ("The warranty lasts 36 months, but the document does not say who pays for shipping.",
                     "The warranty is long but the document does not say how long.",
                     "Austin is the capital, but I do not know the population."):
            self.assertFalse(judging.is_abstention(text), text)

    def test_a_decline_that_states_a_figure_goes_to_the_model(self):
        self.assertFalse(judging.is_abstention("I do not know, but it is probably 24 months."))

    def test_a_polite_lead_in_is_still_an_abstention(self):
        self.assertTrue(judging.is_abstention("Sorry, I do not know."))


class AbstentionWithAttachedClaimTests(unittest.TestCase):
    """A decline with a claim attached must go to the model: an abstention skips it entirely."""

    def test_a_decline_with_an_invented_claim_is_not_an_abstention(self):
        for text in ("The document does not say, but the CEO is Bob Smith.",
                     "Sorry, I do not know, but the CEO was fired for fraud.",
                     "Yes. The document does not mention it, and Alice stole the funds.",
                     "Alice stole it. I do not know why.",
                     "I do not know, but it is thirty six months.",
                     "I do not know. Boston hosts it.",
                     "I do not know; the plant is in Boston"):
            self.assertFalse(judging.is_abstention(text), text)

    def test_spelled_out_figures_count_as_numbers(self):
        self.assertFalse(judging.is_abstention("I do not know whether it is thirty six months"))

    def test_plain_declines_still_skip_the_model(self):
        for text in ("I do not know.", "The context does not say.", "Sorry, I do not know.",
                     "I can't verify an answer to this question from the document.", "That is not mentioned in the document."):
            self.assertTrue(judging.is_abstention(text), text)


class HardenedCheckTests(unittest.TestCase):
    def test_context_headers_are_stripped(self):
        context = "[c1] (page 12)\nThe plant opened in spring.\n\n---\n\n[c2] (page 3)\nIt employs staff."
        stripped = judging.strip_context_headers(context)
        self.assertNotIn("page", stripped)
        self.assertNotIn("[c1]", stripped)
        self.assertNotIn("---", stripped)
        self.assertIn("The plant opened in spring.", stripped)

    def test_a_page_number_is_not_a_number_the_document_states(self):
        context = "[c1] (page 12)\nThe plant opened in spring."
        self.assertEqual(judging.precheck("q", "The plant employs 12 people.", context)["unverified_numbers"], ["12"])

    def test_list_markers_are_not_quantities(self):
        self.assertEqual(judging.extract_numbers("1. The shop opens.\n2. The shop closes."), set())

    def test_an_invented_name_that_opens_a_sentence_is_checked(self):
        self.assertEqual(judging.precheck("q", "Boston hosts it.", "The plant opened in spring.")["unverified_names"], ["Boston"])

    def test_ordinary_sentence_starters_and_the_pronoun_i_are_not_names(self):
        self.assertEqual(judging.extract_names("The plant opened. According to staff, I think so. Yes it did."), [])

    def test_names_are_matched_as_words_not_substrings(self):
        context = "The customer and the party arrived."
        self.assertEqual(judging.precheck("q", "Tom and Art came.", context)["unverified_names"], ["Tom", "Art"])

    def test_evidence_helpers(self):
        self.assertFalse(judging.evidence_in_context("", "The shop opens."))
        self.assertFalse(judging.evidence_in_context("...", "The shop opens."))
        self.assertTrue(judging.evidence_in_context("the SHOP opens", "The shop opens at nine."))
        self.assertTrue(judging.evidence_relates_to_claim("The shop opens at 9", "Hours: opens 9 to 5"))
        self.assertFalse(judging.evidence_relates_to_claim("The shop opens", "Water damage is excluded."))

    def test_fail_outranks_error_and_error_outranks_pass(self):
        supported = {"claim": "a", "verdict": "supported", "evidence": "e", "reason": ""}
        unsupported = {"claim": "b", "verdict": "unsupported", "evidence": "", "reason": "no"}
        unchecked = {"claim": "c", "verdict": "unchecked", "evidence": "", "reason": "skipped"}
        self.assertEqual(judging.faithfulness_verdict([supported, unsupported, unchecked])[0], "fail")
        self.assertEqual(judging.faithfulness_verdict([supported, unchecked])[0], "error")
        self.assertEqual(judging.faithfulness_verdict([supported])[0], "pass")

    def test_claim_limit_is_enforced_but_exactly_the_limit_is_fine(self):
        ok = json.dumps({"claims": [f"claim {i}" for i in range(judging.CLAIM_LIMIT)]})
        self.assertEqual(len(judging.parse_claims(ok, "a")), judging.CLAIM_LIMIT)
        with self.assertRaises(judging.JudgeOutputError):
            judging.parse_claims(json.dumps({"claims": [f"claim {i}" for i in range(judging.CLAIM_LIMIT + 1)]}), "a")

    def test_text_that_looks_like_a_slot_is_left_alone(self):
        built = judging.build_support_prompt("The doc says <<CLAIMS>> literally.", ["claim one"])
        self.assertIn("The doc says <<CLAIMS>> literally.", built)
        self.assertEqual(built.count("claim one"), 1)
        self.assertIn("<<CLAIMS>>", built)
        injected = judging.build_claims_prompt("An answer with <<ANSWER>> inside")
        self.assertIn("An answer with <<ANSWER>> inside", injected)


class PrecheckTests(unittest.TestCase):
    CONTEXT = "The warranty covers defects for 24 months from purchase in Austin."

    def check(self, answer, question="How long is the warranty?", citations=None):
        return judging.precheck(question, answer, self.CONTEXT, citations)

    def test_a_number_missing_from_the_context_is_flagged(self):
        self.assertEqual(self.check("The warranty lasts 36 months.")["unverified_numbers"], ["36"])

    def test_numbers_in_the_context_or_the_question_are_not_flagged(self):
        self.assertEqual(self.check("It lasts 24 months.")["unverified_numbers"], [])
        self.assertEqual(self.check("Yes, 18 months is wrong.", question="Is it 18 months?")["unverified_numbers"], [])
        self.assertEqual(self.check("It lasts twenty four months.")["unverified_numbers"], [])

    def test_a_derived_number_is_flagged_too(self):
        self.assertEqual(self.check("That is 2 years.")["unverified_numbers"], ["2"])  # not stated, even if it could be computed

    def test_names_missing_from_the_context_are_flagged(self):
        self.assertEqual(self.check("The plant in Boston honours it.")["unverified_names"], ["Boston"])
        self.assertEqual(self.check("The plant in Austin honours it.")["unverified_names"], [])

    def test_the_no_verified_citation_flag(self):
        self.assertFalse(self.check("It lasts 24 months.")["no_verified_citation"])  # citations not provided
        self.assertTrue(self.check("It lasts 24 months.", citations=[])["no_verified_citation"])
        self.assertFalse(self.check("It lasts 24 months.", citations=[{"chunk_id": "c1"}])["no_verified_citation"])
        self.assertFalse(self.check("I do not know.", citations=[])["no_verified_citation"])

    def test_no_findings_means_no_hint_text(self):
        self.assertEqual(judging.hint_text(self.check("It lasts 24 months.")), "")

    def test_hints_name_the_unverified_names_only(self):
        hint = judging.hint_text(self.check("The plant in Boston honours it, for 36 months."))
        self.assertIn("Boston", hint)
        self.assertNotIn("36", hint)  # numbers are handled by a hard check, not a hint


class ParsingTests(unittest.TestCase):
    CONTEXT = "The shop opens at 9 am.\nIt closes at 5 pm."

    def test_claims_are_squashed_and_capped(self):
        raw = json.dumps({"claims": [f"claim  {i}" for i in range(10)]})
        claims = judging.parse_claims(raw, "answer")
        self.assertEqual(len(claims), judging.CLAIM_LIMIT)
        self.assertEqual(claims[0], "claim 0")

    def test_bad_claim_replies_fall_back_or_raise(self):
        self.assertEqual(judging.parse_claims(json.dumps({"claims": ["", 5]}), "the whole answer"), ["the whole answer"])
        for raw in ("nope", "[1, 2]", None):
            with self.assertRaises(judging.JudgeOutputError):
                judging.parse_claims(raw, "a")

    def test_support_results_follow_claim_order_and_keep_verified_evidence(self):
        raw = json.dumps({"results": [
            {"claim": 2, "verdict": "unsupported", "evidence": "", "reason": "nope"},
            {"claim": 1, "verdict": "supported", "evidence": "The shop opens\nat 9 am.", "reason": "ok"}]})
        results = judging.parse_support(raw, ["opens at 9", "has a cafe"], self.CONTEXT)
        self.assertEqual([r["verdict"] for r in results], ["supported", "unsupported"])
        self.assertEqual(results[0]["evidence"], "The shop opens at 9 am.")

    def test_an_unknown_verdict_counts_as_unsupported(self):
        raw = json.dumps({"results": [{"claim": 1, "verdict": "kind of", "evidence": "The shop opens at 9 am.", "reason": ""}]})
        self.assertEqual(judging.parse_support(raw, ["c"], self.CONTEXT)[0]["verdict"], "unsupported")

    def test_missing_or_malformed_results_raise(self):
        for raw in (json.dumps({"results": "x"}), json.dumps({}), "{bad"):
            with self.assertRaises(judging.JudgeOutputError):
                judging.parse_support(raw, ["c"], self.CONTEXT)

    def test_supported_without_evidence_is_downgraded(self):
        raw = json.dumps({"results": [{"claim": 1, "verdict": "supported", "evidence": "", "reason": "trust me"}]})
        self.assertEqual(judging.parse_support(raw, ["c"], self.CONTEXT)[0]["verdict"], "unsupported")

    def test_faithfulness_verdicts(self):
        ok = [{"claim": "a", "verdict": "supported", "evidence": "e", "reason": ""}]
        bad = ok + [{"claim": "b", "verdict": "unsupported", "evidence": "", "reason": "not stated"}]
        self.assertEqual(judging.faithfulness_verdict(ok)[0], "pass")
        verdict, why = judging.faithfulness_verdict(bad)
        self.assertEqual(verdict, "fail")
        self.assertIn("1 of 2 claims supported", why)
        self.assertIn("b (not stated)", why)

    def test_relevance_parsing(self):
        self.assertEqual(judging.parse_relevance(json.dumps({"asked": "x", "addresses": "YES", "reason": "r"}))[0], "pass")
        for bad in ("{bad", json.dumps({"addresses": "sort of"}), json.dumps([])):
            with self.assertRaises(judging.JudgeOutputError):
                judging.parse_relevance(bad)


class JudgeReportTests(unittest.TestCase):
    def test_report_shows_checks_claims_evidence_and_relevance(self):
        lines = []
        result = {
            "faithfulness": "fail", "relevance": "pass", "faithfulness_reasoning": "x", "relevance_reasoning": "asked: q | yes: ok",
            "claims": [{"claim": "It lasts 24 months.", "verdict": "supported", "evidence": "for 24 months", "reason": ""},
                       {"claim": "It covers water.", "verdict": "unsupported", "evidence": "", "reason": "Water is excluded."}],
            "checks": {"abstention": False, "unverified_numbers": ["36"], "unverified_names": [], "no_verified_citation": True},
        }
        dev_insight.print_judge_report(result, out=lines.append)
        text = "\n".join(lines)
        for expected in ("JUDGE", "numbers in the answer found nowhere", "36", "no verified citation",
                         "ok 1. It lasts 24 months.", "evidence: for 24 months", "NO 2. It covers water.", "why: Water is excluded.", "relevance: pass"):
            self.assertIn(expected, text)

    def test_report_for_an_abstention(self):
        lines = []
        dev_insight.print_judge_report({"faithfulness": "pass", "relevance": "fail", "faithfulness_reasoning": "no claims",
                                        "relevance_reasoning": "declines", "claims": [], "checks": {"abstention": True}}, out=lines.append)
        self.assertIn("no model was needed", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
