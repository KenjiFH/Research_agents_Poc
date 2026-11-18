"""Compare answer styles on paraphrased questions (needs a local Ollama with nomic-embed-text).

Run from the project root:  python3 eval/answer_compare.py [model ...]   (default: llama3)

For each model it answers 12 paraphrased questions two ways, with the right passage among four
retrieved ones: the OLD setup (loose JSON mode, terse prompt) and the CURRENT one (the schema and
prompt the app ships, agents.SYNTHESIS_SCHEMA and agents.SYNTHESIS_PROMPT). It reports how many
answers contain the expected fact, the average answer length, how many citations verified, and how
often an answer echoes the prompt's example sentence. A small, synthetic check, not a benchmark.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

import agents

DOCS = [
    "The warranty covers manufacturing defects for a period of 24 months from the date of purchase. Water damage is excluded.",
    "Employees accrue 1.5 days of paid leave per month. Unused leave up to 10 days may be carried over to the next year.",
    "Revenue for Q3 2024 was $4.2 million, an increase of 12% over the prior quarter, driven by subscription growth.",
    "To reset your password, open Settings, choose Security, and select Reset Password. A link is emailed within five minutes.",
    "The data retention policy requires customer records to be deleted 7 years after the account is closed.",
    "Section 4.2: The contractor shall indemnify the client against third-party claims arising from negligence.",
    "The API rate limit is 100 requests per minute per key. Exceeding it returns HTTP 429.",
    "The company was founded in 1998 in Austin, Texas by two former engineers.",
    "Shipping to international destinations takes 7 to 14 business days. Customs fees are the buyer's responsibility.",
    "Battery life is rated at 18 hours of continuous video playback under standard test conditions.",
    "The board of directors meets quarterly. A quorum requires at least five of the nine members.",
    "Refunds are issued to the original payment method within 10 business days after the returned item is inspected.",
]
QUESTIONS = [
    ("What happens if I get the device wet?", "water damage is excluded"),
    ("How many vacation days do I earn?", "1.5"),
    ("Can I roll over unused PTO?", "10 days"),
    ("How much did sales grow last quarter?", "12%"),
    ("I forgot my login credentials, what do I do?", "reset password"),
    ("How long do you keep customer data?", "7 years"),
    ("Who is liable if a third party sues?", "contractor"),
    ("What error do I get for too many requests?", "429"),
    ("Where was the company started?", "austin"),
    ("How long does overseas delivery take?", "7 to 14"),
    ("When will I get my money back?", "10 business days"),
    ("How many members needed for a quorum?", "five"),
]
OLD_PROMPT = (
    "You are a helpful and precise assistant. Answer the user's question based ONLY on the provided context. "
    "If the answer is not in the context, state that you do not know and return no citations.\n\n"
    "Each context passage starts with its id in square brackets, like [c3].\n"
    "Reply with JSON only, in exactly this shape:\n"
    '{{"answer": "<your answer>", "citations": [{{"chunk_id": "<id of a passage you used>", "quote": "<words copied exactly from that passage>"}}]}}\n'
    "Copy each quote verbatim from the passage; never paraphrase it.\n\nContext:\n{context}\n\nQuestion: {question}\n\nJSON:"
)


def run(model, embeddings, vectors):
    setups = {
        "old (loose JSON)": (ChatOllama(model=model, temperature=0.2, format="json"), ChatPromptTemplate.from_template(OLD_PROMPT)),
        "current (schema)": (ChatOllama(model=model, temperature=0.2, format=agents.SYNTHESIS_SCHEMA), ChatPromptTemplate.from_template(agents.SYNTHESIS_PROMPT)),
    }
    totals = {name: dict(correct=0, words=0, cites=0, echo=0, n=0) for name in setups}
    for question, key in QUESTIONS:
        query = np.array(embeddings.embed_query(question))
        top = np.argsort(((vectors - query) ** 2).sum(1))[:4]
        retrieved = [Document(page_content=DOCS[i], metadata={"chunk_id": f"c{i + 1}", "page_no": 1}) for i in top]
        context, chunks = agents.format_context(retrieved)
        for name, (llm, prompt) in setups.items():
            raw = llm.invoke(prompt.format_messages(context=context, question=question)).content
            answer, citations = agents.parse_synthesis(raw, chunks)
            t = totals[name]
            t["correct"] += key in answer.lower()
            t["words"] += len(answer.split())
            t["cites"] += len(agents.verify_citations(citations, chunks))
            t["echo"] += "renewed" in answer.lower()  # the current prompt's example sentence
            t["n"] += 1
    for name, t in totals.items():
        print(f"  {name}: correct {t['correct']}/{t['n']}  avg words {t['words'] / t['n']:.1f}  "
              f"answers with a verified citation {t['cites']}  echoed the example {t['echo']}")


if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectors = np.array(embeddings.embed_documents(DOCS))
    for model in sys.argv[1:] or ["llama3"]:
        print(model)
        run(model, embeddings, vectors)
