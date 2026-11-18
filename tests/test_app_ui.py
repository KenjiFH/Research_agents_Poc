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


BASE = {"answer": "Blue.", "faithfulness": "pass", "relevance": "pass"}


class AppRenderTests(unittest.TestCase):
    def test_verified_citations_render_as_source_cards_with_plain_text_quotes(self):
        at = ask({**BASE, "citations": [{"chunk_id": "c1", "page_no": 4, "quote": "The sky is *blue*."}]})
        self.assertEqual(len(at.exception), 0)
        self.assertIn("**Page 4 · c1**", [m.value for m in at.markdown])
        self.assertEqual([t.value for t in at.text], ["The sky is *blue*."])

    def test_no_citations_shows_the_no_verified_sources_note(self):
        at = ask({**BASE, "citations": []})
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([c.value for c in at.caption], [session_logic.NO_SOURCES_NOTE])

    def test_verdicts_show_as_labels_with_a_warning_and_never_a_percentage(self):
        at = ask({"answer": "Blue.", "citations": [], "faithfulness": "fail", "relevance": "pass"})
        self.assertEqual([m.value for m in at.metric], ["❌ Fail", "✅ Pass"])
        self.assertEqual(len(at.warning), 1)

    def test_citations_survive_a_rerun_in_the_history(self):
        at = ask({**BASE, "citations": [{"chunk_id": "c2", "page_no": 7, "quote": "Grass is green."}]})
        with mock.patch.object(session_logic, "upload_action", return_value="none"):
            at.run()
        self.assertIn("**Page 7 · c2**", [m.value for m in at.markdown])


if __name__ == "__main__":
    unittest.main()
