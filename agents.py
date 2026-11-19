# New imports for LangSmith client
from langchain_core.runnables.config import RunnableConfig
from langsmith import Client

# Initialize the LangSmith client
ls_client = Client()


import json
import os
import warnings
import ollama
import dev_insight
import judging
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# The model that writes the answer. It is a separate setting from JUDGE_MODEL (AD-4); set
# ANSWER_MODEL in the environment to change it.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "").strip() or "llama3"

# The judge grades answers and must not be the model that wrote them (AD-4).
# Override with the JUDGE_MODEL environment variable; pull the model with Ollama first.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "").strip() or "qwen2.5:14b"
# The staged judge prompts carry few-shot examples plus the whole context, which can outgrow
# Ollama's default 4096-token window, so the judge gets a larger one. Override with JUDGE_NUM_CTX.
def _read_int(name, default):
    """An integer setting from the environment; anything that is not a whole number falls back."""
    try:
        return int(os.getenv(name, "").strip())
    except ValueError:
        return default


JUDGE_NUM_CTX = _read_int("JUDGE_NUM_CTX", 8192)
JUDGE_PROMPT_LIMIT = 0.85  # share of the window a prompt may fill before the judge refuses rather than truncate


# The synthesizer's reply is constrained to this schema, so structure comes from the model server
# instead of loose JSON mode, and the description asks for a complete-sentence answer.
SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "A complete sentence that restates what was asked and gives the specifics, drawn only from the context.",
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["chunk_id", "quote"],
            },
        },
    },
    "required": ["answer", "citations"],
}


# The synthesizer's instructions. The example is deliberately generic so a small model has nothing
# domain-specific to echo; citations are required whenever the answer comes from the context.
SYNTHESIS_PROMPT = (
    "You are a helpful and precise assistant. Answer the user's question "
    "based ONLY on the provided context. If the answer is not in the context, "
    "say you do not know and return no citations.\n\n"
    "Write the answer as one or two complete sentences that restate what was asked and give "
    "the specifics, for example 'Yes, the contract can be renewed for up to two further terms.' "
    "Never answer with a single word.\n"
    "Each context passage starts with its id in square brackets, like [c3]. Whenever the answer "
    "comes from the context, cite at least one passage, and cite every passage you used; "
    "copy each quote verbatim from that passage, never paraphrase it.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def _read_seed(default=7):
    """MODEL_SEED from the environment; anything that is not a whole number falls back to the default."""
    try:
        return int(os.getenv("MODEL_SEED", "").strip())
    except ValueError:
        return default


# Random sampling is switched off for every model call: temperature 0 picks the most likely token
# each time, and a fixed seed pins whatever randomness remains. The same prompt then gives the same
# answer and the same grade (for the same model, hardware and Ollama version). Change MODEL_SEED to
# get a different but still repeatable result.
MODEL_SEED = _read_seed()


def model_options():
    """Sampling settings shared by the answering model and the judge."""
    return {"temperature": 0.0, "seed": MODEL_SEED}


def _same_model(a, b):
    """Ollama treats 'llama3' and 'llama3:latest' as one model; compare case-insensitively."""
    def normalize(name):
        name = name.strip().lower()
        return name[: -len(":latest")] if name.endswith(":latest") else name
    return normalize(a) == normalize(b)


def tracing_enabled():
    """True when LangSmith tracing is switched on in the environment (as the README's .env does)."""
    for name in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        if os.getenv(name, "").strip().lower() in ("true", "1", "yes"):
            return True
    return False


# Abstention shown instead of an answer that failed the faithfulness check twice (AD-8).
ABSTENTION_MESSAGE = "I can't verify an answer to this question from the document."


def gate_enabled():
    """The faithfulness gate is off until GATE_ENABLED is set; switch it on once the judge is
    validated against human labels (story 3.3)."""
    return os.getenv("GATE_ENABLED", "").strip().lower() in ("true", "1", "yes")


def gate_decision(faithfulness, retry_count):
    """What happens after grading (AD-1): "retry" once on a fail, then "abstain", else "end".

    An "error" verdict never retries or abstains (AD-3), and nothing happens while the gate is off.
    """
    if not gate_enabled() or faithfulness != "fail":
        return "end"
    return "retry" if retry_count < 1 else "abstain"


def verdict_to_score(verdict):
    """LangSmith feedback score for a verdict: pass 1, fail 0, error None (nothing is sent)."""
    return {"pass": 1, "fail": 0}.get(verdict)


class JudgeModelNotFound(RuntimeError):
    """The configured judge model is not available in Ollama."""


def get_judge_warning():
    """Return a warning when the judge and the answering model are the same, else None."""
    if _same_model(JUDGE_MODEL, ANSWER_MODEL):
        return (
            f"Judge model and answering model are both '{ANSWER_MODEL}': "
            "self-grading is biased. Set JUDGE_MODEL or ANSWER_MODEL so they differ."
        )
    return None


def _ask(llm, prompt):
    """One model call, with a missing judge model turned into a message that says how to fix it."""
    try:
        return llm.invoke(prompt).content
    except ollama.ResponseError as error:
        if getattr(error, "status_code", None) == 404:
            raise JudgeModelNotFound(
                f"Judge model '{JUDGE_MODEL}' is not available in Ollama. "
                f"Run: ollama pull {JUDGE_MODEL}"
            ) from error
        raise


def _require_fits_window(prompt):
    """Refuse a prompt that would fill the judge's window: Ollama would silently drop the start of it,
    and claims about the lost text would then be wrongly marked unsupported."""
    estimated = len(prompt) // 4  # rough characters per token
    if estimated > JUDGE_NUM_CTX * JUDGE_PROMPT_LIMIT:
        raise judging.JudgeOutputError(
            f"the judge prompt is about {estimated} tokens, too large for the {JUDGE_NUM_CTX}-token judge window; "
            "raise JUDGE_NUM_CTX or retrieve less text")


def _judge_models(llm):
    """The model for each stage: the one supplied (tests), else the judge model with that stage's schema."""
    if llm is not None:
        return llm, llm, llm
    return tuple(
        ChatOllama(model=JUDGE_MODEL, format=schema, num_ctx=JUDGE_NUM_CTX, **model_options())
        for schema in (judging.CLAIMS_SCHEMA, judging.SUPPORT_SCHEMA, judging.RELEVANCE_SCHEMA)
    )


def judge(question, answer, context, llm=None, citations=None):
    """Grade one answer for faithfulness and relevance, in stages. The only judge implementation (AD-7).

    0. Quick deterministic checks (judging.precheck): an answer that only declines to answer needs no
       model; an answer that states a number found nowhere in the context or the question fails
       faithfulness without the model; names found nowhere in the context become hints.
    1. The answer is split into separate claims, 2. each claim is checked against the context with a
       quoted piece of evidence, 3. code verifies each quote is really in the context. The answer is
       faithful only if every claim is supported by verified evidence.
    Relevance is a separate short call: say what was asked, then decide whether the answer addresses it.
    Each stage prompt carries worked examples.

    Returns {"faithfulness", "relevance", "faithfulness_reasoning", "relevance_reasoning", "claims",
    "checks"}: each metric is "pass", "fail" or "error" (the judge's reply could not be used), "claims"
    lists each claim with its verdict and evidence, and "checks" holds the deterministic findings.
    A missing judge model raises JudgeModelNotFound; other model failures propagate to the caller.
    """
    checks = judging.precheck(question, answer, context, citations)
    if checks["abstention"]:
        return {
            "faithfulness": "pass", "relevance": "fail",
            "faithfulness_reasoning": "The answer declines to answer, so it makes no claims to check.",
            "relevance_reasoning": "The answer declines to answer the question.",
            "claims": [], "checks": checks,
        }

    claims_llm, support_llm, relevance_llm = _judge_models(llm)

    claims = []
    if checks["unverified_numbers"]:
        # A figure the document never states: faithfulness fails without asking the model.
        faithfulness = "fail"
        faithfulness_reasoning = judging.numbers_reasoning(checks["unverified_numbers"])
    else:
        try:
            listed = judging.parse_claims(_ask(claims_llm, judging.build_claims_prompt(answer)), answer)
            support_prompt = judging.build_support_prompt(context, listed, checks)
            _require_fits_window(support_prompt)
            claims = judging.parse_support(_ask(support_llm, support_prompt), listed, context)
            faithfulness, faithfulness_reasoning = judging.faithfulness_verdict(claims)
        except judging.JudgeOutputError as error:
            faithfulness, faithfulness_reasoning = "error", str(error)

    try:
        relevance, relevance_reasoning = judging.parse_relevance(
            _ask(relevance_llm, judging.build_relevance_prompt(question, answer)))
    except judging.JudgeOutputError as error:
        relevance, relevance_reasoning = "error", str(error)

    return {
        "faithfulness": faithfulness, "relevance": relevance,
        "faithfulness_reasoning": faithfulness_reasoning, "relevance_reasoning": relevance_reasoning,
        "claims": claims, "checks": checks,
    }


def format_context(docs):
    """Number retrieved chunks by chunk_id so the synthesizer can cite them (AD-5).

    Returns (context_text, retrieved_chunks) where retrieved_chunks is a list of
    {"chunk_id", "page_no", "type", "content"} used later to resolve and verify citations.
    """
    retrieved = []
    blocks = []
    for position, doc in enumerate(docs, start=1):
        # Fallback ids use an "r" prefix so they can never collide with real "c<n>" ids
        chunk_id = doc.metadata.get("chunk_id") or f"r{position}"
        page_no = doc.metadata.get("page_no")
        retrieved.append({
            "chunk_id": chunk_id, "page_no": page_no, "type": doc.metadata.get("type"), "content": doc.page_content,
        })
        page_label = f" (page {page_no})" if page_no is not None else ""
        blocks.append(f"[{chunk_id}]{page_label}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks), retrieved


def _squash_whitespace(text):
    return " ".join(str(text).split())


def verify_citations(citations, retrieved_chunks):
    """Keep only citations whose quote is a verbatim substring of the chunk they name (AD-5).

    Whitespace runs (line breaks, repeated spaces) are treated as one space on both sides, so a
    quote copied across a line break still verifies; words and case must match exactly.
    """
    content = {chunk["chunk_id"]: _squash_whitespace(chunk["content"]) for chunk in retrieved_chunks}
    verified = []
    for citation in citations:
        quote = _squash_whitespace(citation.get("quote", ""))
        if quote and quote in content.get(citation.get("chunk_id"), ""):
            verified.append(citation)
    return verified


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
    retrieved_chunks: list   # [{chunk_id, page_no, type, content}] behind the numbered context
    answer: str
    citations: list          # [{chunk_id, page_no, quote}]
    faithfulness: str        # "pass" | "fail" | "error": did it use only the context?
    relevance: str           # "pass" | "fail" | "error": did it answer the prompt?
    retry_count: int         # synthesis retries used (capped at one, AD-1)
    abstained: bool          # True when the answer was withheld (AD-8)
    gate: str                # "retry" | "abstain" | "end": decided once, by the evaluator (AD-1)

def create_multi_agent_workflow(retriever):
    # The synthesizer replies in a schema-constrained structure so it can return citations with its answer
    synth_llm = ChatOllama(model=ANSWER_MODEL, format=SYNTHESIS_SCHEMA, **model_options())

    # ==========================================
    # Agent 1: The Researcher
    # ==========================================
    def researcher_node(state: RAGState):
        print("--- RESEARCHER: Searching Vector DB ---")
        docs = retriever.invoke(state["question"])
        formatted_context, retrieved_chunks = format_context(docs)
        if dev_insight.debug_enabled():
            dev_insight.print_context_report(formatted_context, retrieved_chunks)
        return {"context": formatted_context, "retrieved_chunks": retrieved_chunks}

    # ==========================================
    # Agent 2: The Synthesizer
    # ==========================================
    def synthesizer_node(state: RAGState):
        print("--- SYNTHESIZER: Drafting Response ---")
        prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
        chain = prompt | synth_llm
        response = chain.invoke({"context": state["context"], "question": state["question"]})
        retrieved = state.get("retrieved_chunks", [])
        answer, citations = parse_synthesis(response.content, retrieved)
        return {"answer": answer, "citations": verify_citations(citations, retrieved)}

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

        try:
            result = judge(state["question"], state["answer"], state["context"],
                           citations=state.get("citations"))
        except JudgeModelNotFound:
            raise  # a setup problem the user must fix; the app shows the message
        except Exception as error:
            # The judge could not grade: an error verdict, never a silent fail (AD-3)
            print(f"Evaluator failed: {error}")
            result = {
                "faithfulness": "error", "relevance": "error",
                "faithfulness_reasoning": None, "relevance_reasoning": None,
            }

        print(f"\n[LANGCHAIN EVALUATOR REASONING]")
        print(f"Faithfulness ({result['faithfulness']}): {result['faithfulness_reasoning']}")
        print(f"Relevance ({result['relevance']}): {result['relevance_reasoning']}\n")
        if dev_insight.debug_enabled():
            dev_insight.print_judge_report(result)

        # --- THE LANGSMITH OBSERVABILITY PUSH ---
        # The same verdict the UI shows: pass 1, fail 0, error sends nothing.
        # LangGraph does not pass the root run id to nodes, so the caller supplies it as
        # configurable["ls_run_id"], the id the app gave the invocation. Confirm against a real
        # trace that feedback lands on the root run (story 4.2).
        # A graded answer that will be retried is not the final verdict, so it sends nothing.
        run_id = None
        decision = gate_decision(result["faithfulness"], state.get("retry_count", 0))
        if tracing_enabled() and decision != "retry":
            run_id = (config.get("configurable") or {}).get("ls_run_id") or config.get("run_id")

        if run_id:
            try:
                # A withheld answer has no meaningful relevance, so only faithfulness is sent
                keys = ("faithfulness",) if decision == "abstain" else ("faithfulness", "relevance")
                for key in keys:
                    score = verdict_to_score(result[key])
                    if score is not None:
                        ls_client.create_feedback(run_id, key=key, score=score)
                print("Pushed QA verdicts to the LangSmith dashboard.")
            except Exception as e:
                print(f"Failed to push metrics to LangSmith: {e}")

        return {"faithfulness": result["faithfulness"], "relevance": result["relevance"], "gate": decision}

    # ==========================================
    # Retry and Abstain (the faithfulness gate)
    # ==========================================
    def retry_node(state: RAGState):
        print("--- GATE: faithfulness failed, retrying synthesis once ---")
        return {"retry_count": state.get("retry_count", 0) + 1}

    def abstain_node(state: RAGState):
        print("--- GATE: faithfulness failed again, abstaining ---")
        # The relevance grade belonged to the withheld answer, so it no longer applies
        return {"answer": ABSTENTION_MESSAGE, "citations": [], "abstained": True, "relevance": ""}

    def route_after_evaluation(state: RAGState):
        return {"retry": "retry", "abstain": "abstain", "end": END}[state["gate"]]

    # ==========================================
    # Build the Graph
    # ==========================================

    # we use Stategraph for Deterministic Routing (LLM doesnt get to decide which agent to use), as well as for extensibility -
    #if we want a new feature we just add a new node (eg., "translator")
    workflow = StateGraph(RAGState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("abstain", abstain_node)

    # Flow: Researcher -> Synthesizer -> Evaluator -> (gate) END | retry -> Synthesizer | abstain -> END
    # The only conditional edge branches on the faithfulness verdict; the model never picks a node (AD-1).
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "synthesizer")
    workflow.add_edge("synthesizer", "evaluator")
    workflow.add_conditional_edges(
        "evaluator", route_after_evaluation, {"retry": "retry", "abstain": "abstain", END: END}
    )
    workflow.add_edge("retry", "synthesizer")
    workflow.add_edge("abstain", END)
    
    return workflow.compile()
