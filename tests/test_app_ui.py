"""Renders the real app.py headlessly with a fake workflow (no Ollama, no PDF)."""
import os
import unittest
from unittest import mock

from streamlit.testing.v1 import AppTest

import session_logic

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


class FakeWorkflow:
    def __init__(self, result):
        self.result = result

    def invoke(self, state, config=None):
        return self.result


def ask(result):
    # Keep the app from treating "no uploaded file" as a reset of the fake workflow
    with mock.patch.object(session_logic, "upload_action", return_value="none"):
        at = AppTest.from_file(APP, default_timeout=30)
        at.session_state["workflow"] = FakeWorkflow(result)
        at.run()
        at.chat_input[0].set_value("What colour is the sky?").run()
    return at


CHUNKS = [{"chunk_id": "c1", "page_no": 4, "type": "text", "content": "The sky is blue."}]
BASE = {"answer": "Blue.", "faithfulness": "pass", "relevance": "pass", "retrieved_chunks": CHUNKS}


class AppRenderTests(unittest.TestCase):
    def test_verified_citations_render_as_source_cards_with_plain_text_quotes(self):
        at = ask({**BASE, "citations": [{"chunk_id": "c1", "page_no": 4, "quote": "The sky is *blue*."}]})
        self.assertEqual(len(at.exception), 0)
        self.assertIn("**Page 4 · c1**", [m.value for m in at.markdown])
        self.assertIn("The sky is *blue*.", [t.value for t in at.text])  # shown as plain text, not rendered markdown

    def test_no_citations_shows_the_no_verified_sources_note(self):
        at = ask({**BASE, "citations": []})
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([c.value for c in at.caption], [session_logic.NO_SOURCES_NOTE])

    def test_verdicts_show_as_labels_with_a_warning_and_never_a_percentage(self):
        at = ask({"answer": "Blue.", "citations": [], "faithfulness": "fail", "relevance": "pass"})
        self.assertEqual([m.value for m in at.metric], ["❌ Fail", "✅ Pass"])
        self.assertEqual(len(at.warning), 1)

    def test_an_abstained_answer_shows_its_own_warning_and_no_sources_note(self):
        at = ask({"answer": "I can't verify an answer to this question from the document.", "citations": [],
                  "faithfulness": "fail", "relevance": "", "abstained": True, "retrieved_chunks": CHUNKS})
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([c.value for c in at.caption], [])
        self.assertEqual([w.value for w in at.warning], [session_logic.ABSTAINED_WARNING])
        self.assertEqual([m.value for m in at.metric], ["❌ Fail", "– Not graded"])

    def test_an_abstained_answer_keeps_its_warning_and_no_note_after_a_rerun(self):
        at = ask({"answer": "I can't verify an answer to this question from the document.", "citations": [],
                  "faithfulness": "fail", "relevance": "", "abstained": True, "retrieved_chunks": CHUNKS})
        with mock.patch.object(session_logic, "upload_action", return_value="none"):
            at.run()
        self.assertEqual([c.value for c in at.caption], [])
        self.assertEqual([w.value for w in at.warning], [session_logic.ABSTAINED_WARNING])

    def test_the_retrieved_chunks_panel_lists_chunks_in_order_and_survives_a_rerun(self):
        chunks = [
            {"chunk_id": "c4", "page_no": 2, "type": "text", "content": "First retrieved passage."},
            {"chunk_id": "c9", "page_no": 5, "type": "table", "content": "Second retrieved passage."},
        ]
        at = ask({**BASE, "citations": [], "retrieved_chunks": chunks})
        with mock.patch.object(session_logic, "upload_action", return_value="none"):
            at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([e.label for e in at.expander], ["🔎 Retrieved chunks (2)"])
        md = [m.value for m in at.markdown]
        self.assertLess(md.index("**#1 · c4 · page 2 · text**"), md.index("**#2 · c9 · page 5 · table**"))
        self.assertIn("First retrieved passage.", [t.value for t in at.text])

    def test_an_abstained_answer_still_shows_the_retrieved_chunks_panel(self):
        at = ask({"answer": "I can't verify an answer to this question from the document.", "citations": [],
                  "faithfulness": "fail", "relevance": "", "abstained": True, "retrieved_chunks": CHUNKS})
        self.assertEqual([e.label for e in at.expander], ["🔎 Retrieved chunks (1)"])

    def test_an_empty_retrieval_says_so(self):
        at = ask({**BASE, "citations": [], "retrieved_chunks": []})
        self.assertIn("Nothing was retrieved for this question.", [c.value for c in at.caption])

    def test_citations_survive_a_rerun_in_the_history(self):
        at = ask({**BASE, "citations": [{"chunk_id": "c2", "page_no": 7, "quote": "Grass is green."}]})
        with mock.patch.object(session_logic, "upload_action", return_value="none"):
            at.run()
        self.assertIn("**Page 7 · c2**", [m.value for m in at.markdown])


if __name__ == "__main__":
    unittest.main()
