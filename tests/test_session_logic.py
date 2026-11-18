import unittest

from session_logic import upload_action


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


if __name__ == "__main__":
    unittest.main()
