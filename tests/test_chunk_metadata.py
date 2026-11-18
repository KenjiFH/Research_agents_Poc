import unittest
from types import SimpleNamespace

from langchain_core.documents import Document

from document_processor import get_page_no, tag_chunks


class GetPageNoTests(unittest.TestCase):
    def test_reads_first_provenance_page(self):
        item = SimpleNamespace(prov=[SimpleNamespace(page_no=3), SimpleNamespace(page_no=4)])
        self.assertEqual(get_page_no(item), 3)

    def test_missing_provenance_returns_default(self):
        self.assertIsNone(get_page_no(SimpleNamespace(prov=[])))
        self.assertIsNone(get_page_no(SimpleNamespace()))
        self.assertEqual(get_page_no(SimpleNamespace(prov=None), default=1), 1)

    def test_non_int_page_is_ignored(self):
        item = SimpleNamespace(prov=[SimpleNamespace(page_no="2")])
        self.assertIsNone(get_page_no(item))


class TagChunksTests(unittest.TestCase):
    def test_assigns_ordered_ids_for_all_chunk_types(self):
        chunks = [
            Document(page_content="t", metadata={"type": "text", "page_no": 1}),
            Document(page_content="tb", metadata={"type": "table", "page_no": 2}),
            Document(page_content="im", metadata={"type": "image"}),
        ]
        tag_chunks(chunks)
        self.assertEqual([c.metadata["chunk_id"] for c in chunks], ["c1", "c2", "c3"])
        self.assertEqual(chunks[0].metadata["page_no"], 1)
        self.assertEqual(chunks[2].metadata["page_no"], 1)

    def test_non_int_page_defaults_to_one(self):
        chunk = Document(page_content="x", metadata={"type": "text", "page_no": None})
        tag_chunks([chunk])
        self.assertEqual(chunk.metadata["page_no"], 1)


if __name__ == "__main__":
    unittest.main()
