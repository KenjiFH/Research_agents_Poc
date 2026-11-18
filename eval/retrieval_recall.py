"""Compare retrieval setups on a synthetic noisy document (needs a local Ollama with nomic-embed-text).

Run from the project root:  python3 eval/retrieval_recall.py

The document is 60 boilerplate/chatter chunks plus six fact chunks. For each question it reports
whether the chunk holding the answer was retrieved by: the old top-4 semantic retriever, the
current semantic retriever (larger candidate set + de-duplication), and the hybrid (+ BM25).
It is a small, synthetic check of direction, not a benchmark; run it on your own PDF's chunks for
a real signal.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import document_processor as dp
# a longer, noisier "document": 60 chunks, many near-duplicate boilerplate/overlap chunks, answers buried
filler = [
 "Page header: Acme Corp Annual Report 2024. Confidential. Page footer: all rights reserved. Acme Corp.",
 "Acme Corp is committed to quality, customer satisfaction, and long-term shareholder value in all its operations.",
 "The following sections describe our policies, procedures and general information about products and services.",
]
facts = {
 "q3": ("Revenue for Q3 2024 was 4.2 million dollars, an increase of 12 percent driven by subscription growth.", "What was the Q3 2024 revenue?"),
 "429": ("The API rate limit is 100 requests per minute per key. Exceeding it returns HTTP 429 Too Many Requests.", "What does error 429 mean?"),
 "sec42": ("Section 4.2 The contractor shall indemnify the client against third party claims arising from negligence.", "What does section 4.2 say?"),
 "pto": ("Employees accrue 1.5 days of paid leave per month and may carry over up to 10 unused days.", "How many vacation days carry over?"),
 "quorum": ("The board meets quarterly and a quorum requires at least five of the nine members present.", "How many board members make a quorum?"),
 "warranty": ("The warranty covers manufacturing defects for 24 months from purchase. Water damage is excluded.", "How long is the warranty?"),
}
chunks=[]; n=0
def add(text, page):
    global n; n+=1
    chunks.append(Document(page_content=text, metadata={"chunk_id":f"c{n}","page_no":page,"type":"text"}))
gold={}
for page in range(1,11):
    for f in filler:                       # repeated boilerplate on every page (near-duplicates)
        add(f+f" Page {page}.", page)
    for i in range(3):                     # general chatter that sounds topical
        add(f"Our company overview section {page}.{i} discusses revenue, policies, limits, leave, board governance and warranty topics in general terms without specific figures.", page)
for key,(text,_) in facts.items():
    add(text, 7); gold[key]=f"c{n}"
emb=OllamaEmbeddings(model="nomic-embed-text")
store=dp.build_vector_store(chunks, emb)
def ids(docs): return [d.metadata["chunk_id"] for d in docs]
old = lambda q: ids(store.as_retriever(search_kwargs={"k":4}).invoke(q))
sem_only = dp.make_retriever(store)            # larger candidate set + dedup + final 6
hybrid = dp.make_retriever(store, chunks)      # + BM25 keyword search
res={"old k=4 semantic":0,"larger k + dedup":0,"+ BM25":0}
for key,(_,q) in facts.items():
    g=gold[key]; a=g in old(q); b=g in ids(sem_only.invoke(q)); c=g in ids(hybrid.invoke(q))
    res["old k=4 semantic"]+=a; res["larger k + dedup"]+=b; res["+ BM25"]+=c
    print(f"{q[:44]:44} old:{'hit ' if a else 'MISS'} larger-k+dedup:{'hit ' if b else 'MISS'} +BM25:{'hit ' if c else 'MISS'}")
print(f"\nanswer chunk retrieved (of {len(facts)}):", res)
dp.delete_vector_store(store)
