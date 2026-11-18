import unittest

from session_logic import upload_action, verdict_label, verdict_warning


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


if __name__ == "__main__":
    unittest.main()
