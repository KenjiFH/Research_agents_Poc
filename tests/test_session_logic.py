import unittest

from session_logic import NO_SOURCES_NOTE, chunk_heading, chunk_preview, citation_title, sources_note, upload_action, verdict_label, verdict_warning


class UploadActionTests(unittest.TestCase):
    def test_first_upload_builds(self):
        self.assertEqual(upload_action("f1", None, False, False), "build")

    def test_same_file_is_left_alone(self):
        self.assertEqual(upload_action("f1", "f1", True, True), "none")

    def test_different_file_replaces_and_deletes_the_old_collection(self):
        self.assertEqual(upload_action("f2", "f1", True, True), "replace")

    def test_failed_build_with_a_stored_collection_cleans_up_before_retry(self):
        self.assertEqual(upload_action("f1", "f1", False, True), "replace")

    def test_removing_the_file_resets(self):
        self.assertEqual(upload_action(None, "f1", True, True), "reset")

    def test_removed_file_with_only_a_store_still_resets(self):
        self.assertEqual(upload_action(None, "f1", False, True), "reset")

    def test_nothing_uploaded_and_nothing_built_is_a_noop(self):
        self.assertEqual(upload_action(None, None, False, False), "none")


class VerdictDisplayTests(unittest.TestCase):
    def test_labels_never_show_a_percentage(self):
        self.assertEqual(verdict_label("pass"), "✅ Pass")
        self.assertEqual(verdict_label("fail"), "❌ Fail")
        self.assertEqual(verdict_label("error"), "⚠️ Error")
        self.assertNotIn("%", verdict_label("pass"))

    def test_unknown_verdict_shows_not_graded(self):
        self.assertEqual(verdict_label(""), "– Not graded")

    def test_warns_on_faithfulness_fail_error_and_relevance_fail(self):
        self.assertIn("not be supported", verdict_warning("fail", "pass"))
        self.assertIn("could not be checked", verdict_warning("error", "pass"))
        self.assertIn("may not address", verdict_warning("pass", "fail"))

    def test_warns_when_relevance_could_not_be_checked(self):
        self.assertIn("Relevance could not be checked", verdict_warning("pass", "error"))

    def test_no_warning_when_both_pass(self):
        self.assertIsNone(verdict_warning("pass", "pass"))


class SourceCardTests(unittest.TestCase):
    def test_title_shows_page_and_chunk(self):
        self.assertEqual(citation_title({"page_no": 4, "chunk_id": "c3", "quote": "q"}), "Page 4 · c3")

    def test_title_survives_a_missing_page(self):
        self.assertEqual(citation_title({"chunk_id": "c3", "quote": "q"}), "Page ? · c3")

    def test_note_appears_only_when_there_are_no_verified_citations(self):
        self.assertEqual(sources_note([]), NO_SOURCES_NOTE)
        self.assertIsNone(sources_note([{"page_no": 1, "chunk_id": "c1", "quote": "q"}]))


class RetrievedPanelTests(unittest.TestCase):
    def test_heading_shows_rank_id_page_and_type(self):
        chunk = {"chunk_id": "c9", "page_no": 3, "type": "table", "content": "x"}
        self.assertEqual(chunk_heading(chunk, 2), "#2 · c9 · page 3 · table")

    def test_heading_survives_missing_fields(self):
        self.assertEqual(chunk_heading({}, 1), "#1 · ? · page ? · ?")

    def test_preview_squashes_whitespace_and_cuts_long_text(self):
        self.assertEqual(chunk_preview({"content": "a\n\n  b"}), "a b")
        long = chunk_preview({"content": "word " * 200}, limit=20)
        self.assertTrue(long.endswith("…"))
        self.assertLessEqual(len(long), 21)


if __name__ == "__main__":
    unittest.main()
