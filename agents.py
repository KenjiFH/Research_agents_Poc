# New imports for LangSmith client
from langchain_core.runnables.config import RunnableConfig
from langsmith import Client

# Initialize the LangSmith client
ls_client = Client()


import json
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

# 1. Updated Graph State
class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    faithfulness_score: int  # New: Did it use only the context?
    relevance_score: int     # New: Did it answer the prompt?

def create_multi_agent_workflow(retriever):
    # We use format="json" for the evaluator to ensure we get a parseable response
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    llm_json = ChatOllama(model=LLM_MODEL, temperature=0.0, format="json")

    # ==========================================
    # Agent 1: The Researcher
    # ==========================================
    def researcher_node(state: RAGState):
        print("--- RESEARCHER: Searching Vector DB ---")
        docs = retriever.invoke(state["question"])
        formatted_context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return {"context": formatted_context}

    # ==========================================
    # Agent 2: The Synthesizer
    # ==========================================
    def synthesizer_node(state: RAGState):
        print("--- SYNTHESIZER: Drafting Response ---")
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful and precise assistant. Answer the user's question "
            "based ONLY on the provided context. If the answer is not in the context, "
            "state that you do not know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        chain = prompt | llm
        response = chain.invoke({"context": state["context"], "question": state["question"]})
        return {"answer": response.content}

    # ==========================================
    # Agent 3: The Evaluator (NEW)
    # ==========================================
    # ==========================================
    # Agent 3: The Evaluator (LangChain Native)
    # ==========================================
    def evaluator_node(state: RAGState, config: RunnableConfig):
        print("--- EVALUATOR: Grading Response using LangChain Native Evaluators ---")
        
        # WHY WE CHANGED THIS: We no longer need `llm_json`. 
        # LangChain's CriteriaEvalChain uses advanced prompt engineering (Chain-of-Thought) 
        # under the hood and handles the string-to-boolean parsing natively.
        llm = ChatOllama(model=LLM_MODEL, temperature=0.0)

        # 1. Define custom criteria for Faithfulness (Hallucination Check)
        faithfulness_criteria = {
            "faithfulness": "Is the submission strictly based on the provided reference context, without introducing any outside information?"
        }
        
        # 2. Load the "labeled_criteria" evaluator
        # WHY WE CHANGED THIS: The "labeled_criteria" type explicitly requires a 'reference' 
        # parameter. This allows us to pass our retrieved ChromaDB chunks as the absolute truth.
        faithfulness_evaluator = load_evaluator(
            "labeled_criteria", 
            criteria=faithfulness_criteria, 
            llm=llm
        )

        # 3. Define custom criteria for Relevance (Helpfulness Check)
        relevance_criteria = {
            "relevance": "Does the submission directly and completely answer the user's input question?"
        }
        
        # 4. Load the standard "criteria" evaluator
        # WHY WE CHANGED THIS: Relevance only cares if the answer matches the question. 
        # It doesn't need to look at the database context, so we use the standard "criteria" type.
        relevance_evaluator = load_evaluator(
            "criteria", 
            criteria=relevance_criteria, 
            llm=llm
        )

        # --- Execution ---
        print("Evaluating Faithfulness...")
        f_eval = faithfulness_evaluator.evaluate_strings(
            prediction=state["answer"],
            input=state["question"],
            reference=state["context"] # The "labeled_criteria" uses this as the source of truth
        )

        print("Evaluating Relevance...")
        r_eval = relevance_evaluator.evaluate_strings(
            prediction=state["answer"],
            input=state["question"]
        )

        # --- Parsing the Output ---
        # The "Defensive Programming" Fix
        # LangChain evaluators output a dictionary containing: 
        # 'reasoning' (the Chain of Thought), 'value' ('Y'/'N'), and 'score' (1 or 0)
        
        # --- Parsing the Output (Defensive) ---
        print(f"\n[LANGCHAIN EVALUATOR REASONING]")
        print(f"Faithfulness: {f_eval.get('reasoning')}")
        print(f"Relevance: {r_eval.get('reasoning')}\n")

            # Extract the raw score safely (your defensive fix!)
        raw_f = f_eval.get("score")
        raw_r = r_eval.get("score")

        f_score = (raw_f if raw_f is not None else 0) * 100
        r_score = (raw_r if raw_r is not None else 0) * 100

        # --- THE LANGSMITH OBSERVABILITY PUSH ---
        # 1. Grab the unique ID for this specific execution trace
        run_id = config.get("run_id")
        
        if run_id:
            try:
                # 2. Push Faithfulness to the dashboard (LangSmith prefers 0.0 to 1.0 floats)
                ls_client.create_feedback(
                    run_id,
                    key="faithfulness",
                    score=f_score / 100.0, 
                )
                # 3. Push Relevance to the dashboard
                ls_client.create_feedback(
                    run_id,
                    key="relevance",
                    score=r_score / 100.0,
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
