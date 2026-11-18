# New imports for LangSmith client
from langchain_core.runnables.config import RunnableConfig
from langsmith import Client

# Initialize the LangSmith client
ls_client = Client()


import json
import os
import warnings
import ollama
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

#this import was gutted in recent update
#from langchain.evaluation import load_evaluator 


#FOR FUTURE REFERENCE: unresolved imports may just be in langchain_classic.[pkgname]

# so we have to use this
from langchain_classic.evaluation import load_evaluator

LLM_MODEL = "llama3"

# The judge grades answers and must not be the model that wrote them (AD-4).
# Override with the JUDGE_MODEL environment variable; pull the model with Ollama first.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "").strip() or "qwen2.5:14b"


def _same_model(a, b):
    """Ollama treats 'llama3' and 'llama3:latest' as one model; compare case-insensitively."""
    def normalize(name):
        name = name.strip().lower()
        return name[: -len(":latest")] if name.endswith(":latest") else name
    return normalize(a) == normalize(b)


class JudgeModelNotFound(RuntimeError):
    """The configured judge model is not available in Ollama."""


def get_judge_warning():
    """Return a warning when the judge and the answering model are the same, else None."""
    if _same_model(JUDGE_MODEL, LLM_MODEL):
        return (
            f"Judge model and answering model are both '{LLM_MODEL}': "
            "self-grading is biased. Set JUDGE_MODEL to a different model."
        )
    return None


def judge(question, answer, context, llm=None):
    """Grade one answer for faithfulness and relevance. The only judge implementation (AD-7).

    Returns {"faithfulness_score", "relevance_score", "faithfulness_reasoning", "relevance_reasoning"}
    with scores on the existing 0-100 scale (None when the evaluator gave no score).
    """
    if llm is None:
        llm = ChatOllama(model=JUDGE_MODEL, temperature=0.0)

    faithfulness_evaluator = load_evaluator(
        "labeled_criteria",
        criteria={
            "faithfulness": "Is the submission strictly based on the provided reference context, without introducing any outside information?"
        },
        llm=llm,
    )
    relevance_evaluator = load_evaluator(
        "criteria",
        criteria={
            "relevance": "Does the submission directly and completely answer the user's input question?"
        },
        llm=llm,
    )

    try:
        f_eval = faithfulness_evaluator.evaluate_strings(
            prediction=answer, input=question, reference=context
        )
        r_eval = relevance_evaluator.evaluate_strings(prediction=answer, input=question)
    except ollama.ResponseError as error:
        if getattr(error, "status_code", None) == 404:
            raise JudgeModelNotFound(
                f"Judge model '{JUDGE_MODEL}' is not available in Ollama. "
                f"Run: ollama pull {JUDGE_MODEL}"
            ) from error
        raise

    def to_percent(raw):
        return None if raw is None else raw * 100

    return {
        "faithfulness_score": to_percent(f_eval.get("score")),
        "relevance_score": to_percent(r_eval.get("score")),
        "faithfulness_reasoning": f_eval.get("reasoning"),
        "relevance_reasoning": r_eval.get("reasoning"),
    }

def format_context(docs):
    """Number retrieved chunks by chunk_id so the synthesizer can cite them (AD-5).

    Returns (context_text, retrieved_chunks) where retrieved_chunks is a list of
    {"chunk_id", "page_no", "content"} used later to resolve and verify citations.
    """
    retrieved = []
    blocks = []
    for position, doc in enumerate(docs, start=1):
        # Fallback ids use an "r" prefix so they can never collide with real "c<n>" ids
        chunk_id = doc.metadata.get("chunk_id") or f"r{position}"
        page_no = doc.metadata.get("page_no")
        retrieved.append({"chunk_id": chunk_id, "page_no": page_no, "content": doc.page_content})
        page_label = f" (page {page_no})" if page_no is not None else ""
        blocks.append(f"[{chunk_id}]{page_label}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks), retrieved


def parse_synthesis(raw_text, retrieved_chunks):
    """Split the synthesizer's JSON reply into (answer, citations).

    Each citation is {"chunk_id", "page_no", "quote"}; the page comes from the retrieved chunk,
    never from the model. Citations naming an unknown chunk are dropped. A reply that is not the
    expected JSON is kept as a plain answer with no citations.
    """
    pages = {chunk["chunk_id"]: chunk["page_no"] for chunk in retrieved_chunks}
    try:
        data = json.loads(raw_text)
    except (TypeError, ValueError):
        data = None
    if not isinstance(data, dict):
        return str(raw_text).strip(), []
    if not str(data.get("answer") or "").strip():
        # Valid JSON but no answer text: never show raw JSON to the user
        return "I do not know.", []

    citations = []
    for entry in data.get("citations") or []:
        if not isinstance(entry, dict):
            continue
        chunk_id = str(entry.get("chunk_id") or "").strip()
        quote = str(entry.get("quote") or "").strip()
        if chunk_id in pages and quote:
            citations.append({"chunk_id": chunk_id, "page_no": pages[chunk_id], "quote": quote})
    return str(data["answer"]).strip(), citations


# 1. Updated Graph State
class RAGState(TypedDict):
    question: str
    context: str
    retrieved_chunks: list   # [{chunk_id, page_no, content}] behind the numbered context
    answer: str
    citations: list          # [{chunk_id, page_no, quote}]
    faithfulness_score: int  # New: Did it use only the context?
    relevance_score: int     # New: Did it answer the prompt?

def create_multi_agent_workflow(retriever):
    # We use format="json" for the evaluator to ensure we get a parseable response
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    llm_json = ChatOllama(model=LLM_MODEL, temperature=0.0, format="json")
    # The synthesizer replies in JSON so it can return citations with its answer
    synth_llm = ChatOllama(model=LLM_MODEL, temperature=0.2, format="json")

    # ==========================================
    # Agent 1: The Researcher
    # ==========================================
    def researcher_node(state: RAGState):
        print("--- RESEARCHER: Searching Vector DB ---")
        docs = retriever.invoke(state["question"])
        formatted_context, retrieved_chunks = format_context(docs)
        return {"context": formatted_context, "retrieved_chunks": retrieved_chunks}

    # ==========================================
    # Agent 2: The Synthesizer
    # ==========================================
    def synthesizer_node(state: RAGState):
        print("--- SYNTHESIZER: Drafting Response ---")
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful and precise assistant. Answer the user's question "
            "based ONLY on the provided context. If the answer is not in the context, "
            "state that you do not know and return no citations.\n\n"
            "Each context passage starts with its id in square brackets, like [c3].\n"
            "Reply with JSON only, in exactly this shape:\n"
            '{{"answer": "<your answer>", "citations": '
            '[{{"chunk_id": "<id of a passage you used>", "quote": "<words copied exactly from that passage>"}}]}}\n'
            "Copy each quote verbatim from the passage; never paraphrase it.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "JSON:"
        )
        chain = prompt | synth_llm
        response = chain.invoke({"context": state["context"], "question": state["question"]})
        answer, citations = parse_synthesis(response.content, state.get("retrieved_chunks", []))
        return {"answer": answer, "citations": citations}

    # ==========================================
    # Agent 3: The Evaluator (NEW)
    # ==========================================
    # ==========================================
    # Agent 3: The Evaluator (LangChain Native)
    # ==========================================
    def evaluator_node(state: RAGState, config: RunnableConfig):
        print(f"--- EVALUATOR: Grading Response with judge model '{JUDGE_MODEL}' ---")

        warning = get_judge_warning()
        if warning:
            warnings.warn(warning)

        result = judge(state["question"], state["answer"], state["context"])

        print(f"\n[LANGCHAIN EVALUATOR REASONING]")
        print(f"Faithfulness: {result['faithfulness_reasoning']}")
        print(f"Relevance: {result['relevance_reasoning']}\n")

        # Existing behavior kept: a missing score becomes 0 (story 4.1 changes this)
        f_score = result["faithfulness_score"] if result["faithfulness_score"] is not None else 0
        r_score = result["relevance_score"] if result["relevance_score"] is not None else 0

        # --- THE LANGSMITH OBSERVABILITY PUSH ---
        # 1. Grab the unique ID for this specific execution trace
        run_id = config.get("run_id")
        
        if run_id:
            try:
                # 2. Push Faithfulness to the dashboard (LangSmith prefers 0.0 to 1.0 floats)
                ls_client.create_feedback(
                    run_id,
                    key="faithfulness",
                    score=f_score, 
                )
                # 3. Push Relevance to the dashboard
                ls_client.create_feedback(
                    run_id,
                    key="relevance",
                    score=r_score,
                )
                print("Successfully pushed QA metrics to LangSmith Dashboard!")
            except Exception as e:
                print(f"Failed to push metrics to LangSmith: {e}")

        return {"faithfulness_score": int(f_score), "relevance_score": int(r_score)}

    # ==========================================
    # Build the Graph
    # ==========================================

    # we use Stategraph for Deterministic Routing (LLM doesnt get to decide which agent to use), as well as for extensibility -
    #if we want a new feature we just add a new node (eg., "translator")
    workflow = StateGraph(RAGState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("evaluator", evaluator_node)

    # Updated Flow: Researcher -> Synthesizer -> Evaluator -> END
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "synthesizer")
    workflow.add_edge("synthesizer", "evaluator")
    workflow.add_edge("evaluator", END)
    
    return workflow.compile()
