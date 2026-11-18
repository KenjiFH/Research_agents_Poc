import unittest
from types import SimpleNamespace

from document_processor import RETRIEVER_K, build_vector_store, delete_vector_store, items_to_raw_documents, make_retriever
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding


def text(content, page, label="text"):
    return (SimpleNamespace(label=label, text=content, prov=[SimpleNamespace(page_no=page)]), 0)


def table(page, caption=None):
    class Frame:
        def to_markdown(self, index=False):
            return "| a |\n|---|\n| 1 |"

    item = SimpleNamespace(
        label="table",
        prov=[SimpleNamespace(page_no=page)],
        captions=[SimpleNamespace(text=caption)] if caption else [],
        export_to_dataframe=lambda: Frame(),
    )
    return (item, 0)


class ItemsToRawDocumentsTests(unittest.TestCase):
    def test_text_never_spans_pages(self):
        docs = items_to_raw_documents([
            text("First paragraph on page one is long enough.", 1),
            text("Second paragraph on page two is long enough.", 2),
        ])
        self.assertEqual([d.metadata["page_no"] for d in docs], [1, 2])
        self.assertNotIn("Second", docs[0].page_content)

    def test_same_page_text_stays_together(self):
        docs = items_to_raw_documents([text("Alpha paragraph one.", 3), text("Beta paragraph two.", 3)])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["page_no"], 3)

    def test_page_less_item_inherits_the_last_page(self):
        pageless = (SimpleNamespace(label="text", text="No page info here at all.", prov=[]), 0)
        docs = items_to_raw_documents([text("On page four, a paragraph.", 4), pageless])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["page_no"], 4)

    def test_table_gets_its_own_page_and_flushes_preceding_text(self):
        docs = items_to_raw_documents([text("Intro text before the table.", 2), table(2, "Revenue")])
        self.assertEqual([d.metadata["type"] for d in docs], ["text", "table"])
        self.assertEqual([d.metadata["page_no"] for d in docs], [2, 2])
        self.assertEqual(docs[1].metadata["title"], "Revenue")


def picture(page, caption=None):
    item = SimpleNamespace(
        label="picture", prov=[SimpleNamespace(page_no=page)],
        captions=[SimpleNamespace(text=caption)] if caption else [], text="OCR words", annotations=None,
    )
    return (item, 0)


class MoreItemsTests(unittest.TestCase):
    def test_picture_flushes_text_and_carries_its_own_page(self):
        docs = items_to_raw_documents([text("A paragraph before the figure.", 1), picture(2, "Figure 1")])
        self.assertEqual([d.metadata["type"] for d in docs], ["text", "image"])
        self.assertEqual([d.metadata["page_no"] for d in docs], [1, 2])
        self.assertEqual(docs[1].metadata["source_element"], "picture")

    def test_table_on_a_later_page_keeps_text_on_the_earlier_page(self):
        docs = items_to_raw_documents([text("Page one text before a table.", 1), table(3)])
        self.assertEqual([d.metadata["page_no"] for d in docs], [1, 3])

    def test_malformed_table_is_skipped_after_the_text_was_flushed(self):
        broken = (SimpleNamespace(label="table", prov=[SimpleNamespace(page_no=1)], captions=[],
                                  export_to_dataframe=lambda: (_ for _ in ()).throw(ValueError("bad"))), 0)
        docs = items_to_raw_documents([text("Text before a broken table.", 1), broken, text("Text after it.", 1)])
        self.assertEqual([d.metadata["type"] for d in docs], ["text", "text"])


class RetrieverBindingTests(unittest.TestCase):
    def test_default_retriever_uses_k_and_only_its_own_collection(self):
        emb = DeterministicFakeEmbedding(size=16)
        a = build_vector_store([Document(page_content=f"ALPHA {i}", metadata={"type": "text", "page_no": 1}) for i in range(6)], emb)
        b = build_vector_store([Document(page_content=f"BETA {i}", metadata={"type": "text", "page_no": 1}) for i in range(6)], emb)
        try:
            retriever = make_retriever(b)
            self.assertEqual(retriever.search_kwargs["k"], RETRIEVER_K)
            docs = retriever.invoke("ALPHA 1")
            self.assertEqual(len(docs), RETRIEVER_K)
            self.assertTrue(all(d.page_content.startswith("BETA") for d in docs))
        finally:
            delete_vector_store(a)
            delete_vector_store(b)


if __name__ == "__main__":
    unittest.main()
