import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from document_processor import build_vector_store, delete_vector_store, new_collection_name


def _chunks(prefix, n=3):
    return [
        Document(page_content=f"{prefix} passage number {i}", metadata={"type": "text", "page_no": 1, "chunk_id": f"c{i + 1}"})
        for i in range(n)
    ]


class CollectionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = DeterministicFakeEmbedding(size=16)
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            try:
                delete_vector_store(store)
            except Exception:
                pass

    def _store(self, prefix):
        store = build_vector_store(_chunks(prefix), self.embeddings)
        self.stores.append(store)
        return store

    def test_collection_names_are_unique(self):
        self.assertNotEqual(new_collection_name(), new_collection_name())

    def test_second_upload_never_returns_first_documents_chunks(self):
        self._store("ALPHA")
        second = self._store("BETA")
        docs = second.as_retriever(search_kwargs={"k": 10}).invoke("ALPHA passage number 0")
        self.assertTrue(docs)
        self.assertTrue(all(d.page_content.startswith("BETA") for d in docs))

    def test_delete_removes_the_collection(self):
        store = self._store("GAMMA")
        name = store._collection.name
        client = store._client

        def names():
            return [c if isinstance(c, str) else c.name for c in client.list_collections()]

        self.assertIn(name, names())
        delete_vector_store(store)
        self.stores.remove(store)
        self.assertNotIn(name, names())

    def test_delete_none_is_a_noop(self):
        delete_vector_store(None)


if __name__ == "__main__":
    unittest.main()
