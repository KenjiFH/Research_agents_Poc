import os
import uuid
from dotenv import load_dotenv

# Load the LangSmith keys before the agents boot up
load_dotenv()
import streamlit as st
from document_processor import process_document_and_create_vdb, delete_vector_store
from session_logic import upload_action, verdict_label, verdict_warning, citation_title, sources_note, ABSTAINED_WARNING
from agents import create_multi_agent_workflow, get_judge_warning, JudgeModelNotFound

st.set_page_config(page_title="Multi-Agent RAG POC", page_icon="🤖", layout="centered")
st.title("🤖 Multi-Agent RAG Assistant")

# AD-4: the judge must not be the answering model; make a misconfiguration visible
_judge_warning = get_judge_warning()
if _judge_warning:
    st.warning(_judge_warning)

# Initialize Session State
if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# NEW: State to hold the raw Markdown chunks for debugging
if "raw_chunks" not in st.session_state:
    st.session_state.raw_chunks = []
# This upload's vector store and file id, so a reset or a new file deletes the old collection
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_id" not in st.session_state:
    st.session_state.file_id = None


def reset_session():
    delete_vector_store(st.session_state.vector_store)
    st.session_state.vector_store = None
    st.session_state.file_id = None
    st.session_state.workflow = None
    st.session_state.messages = []
    st.session_state.raw_chunks = []

# Sidebar Logic
with st.sidebar:
    st.header("1. Upload Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    action = upload_action(
        uploaded_file.file_id if uploaded_file else None,
        st.session_state.file_id,
        st.session_state.workflow is not None,
        st.session_state.vector_store is not None,
    )

    # Build agents for a new upload; a different file replaces the previous one
    if action in ("build", "replace"):
        if action == "replace":
            reset_session()  # delete the previous document's collection first
        with st.spinner("Processing document with Docling..."):
            # Catch the retriever, the chunks and the per-upload vector store
            retriever, chunks, vector_store = process_document_and_create_vdb(uploaded_file)
            # Keep the store before anything else can fail, so a reset can always delete it
            st.session_state.vector_store = vector_store
            st.session_state.file_id = uploaded_file.file_id
            st.session_state.workflow = create_multi_agent_workflow(retriever)
            st.session_state.raw_chunks = chunks  # Save chunks to memory
        st.success("Document processed! Agents are ready.")

    # File removed: hard reset
    elif action == "reset":
        reset_session()
        st.rerun()

    # NEW: The Debug Expander
    # NEW: The Upgraded Debug Expander
    if st.session_state.raw_chunks:
        with st.expander("🛠️ Debug: View Extracted Chunks & Metadata"):
            st.caption(f"Total Chunks Generated: {len(st.session_state.raw_chunks)}")
            
            # Loop through the first 10 chunks to avoid UI lag
            for i, chunk in enumerate(st.session_state.raw_chunks[:10]):
                st.markdown(f"### Chunk {i+1}")
                
                # Display the raw Markdown text
                st.code(chunk.page_content, language="markdown")
                
                # Display the Metadata cleanly using st.json
                st.markdown("**🔍 Chunk Metadata:**")
                st.json(chunk.metadata)
                
                st.divider()

    if st.button("Clear Chat & Reset"):
        reset_session()
        st.rerun()

def render_metrics(faithfulness, relevance, abstained=False):
    """Show each check as Pass / Fail / Error, with a warning when the answer should not be trusted."""
    col1, col2 = st.columns(2)
    col1.metric("🔍 Faithfulness (Hallucination Check)", verdict_label(faithfulness))
    col2.metric("🎯 Relevance (Accuracy Check)", verdict_label(relevance))
    # An abstained answer has its own warning and no sources (AD-8)
    warning = ABSTAINED_WARNING if abstained else verdict_warning(faithfulness, relevance)
    if warning:
        st.warning(warning)


def render_sources(citations):
    """Show each verified citation as a source card: page and chunk, then the verbatim quote."""
    note = sources_note(citations)
    if note:
        st.caption(note)
        return
    st.markdown("**Sources**")
    for citation in citations:
        with st.container(border=True):
            st.markdown(f"**{citation_title(citation)}**")
            st.text(citation["quote"])  # plain text: the quote may contain markdown characters


# Main Chat Interface
st.header("2. Ask the Agents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and not msg.get("abstained", False):
            render_sources(msg.get("citations", []))
        if msg["role"] == "assistant" and "metrics" in msg:
            render_metrics(msg["metrics"]["faithfulness"], msg["metrics"]["relevance"], msg.get("abstained", False))

if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.workflow is None:
        st.error("Please upload a document in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agents are researching, synthesizing, and evaluating..."):
                initial_state = {
                    "question": prompt, 
                    "context": "", 
                    "retrieved_chunks": [],
                    "answer": "", 
                    "citations": [],
                    "faithfulness": "", 
                    "relevance": "",
                    "retry_count": 0,
                    "abstained": False
                }
                
                try:
                    run_id = uuid.uuid4()  # lets the evaluator attach feedback to this trace
                    final_state = st.session_state.workflow.invoke(
                        initial_state,
                        config={"run_id": run_id, "configurable": {"ls_run_id": str(run_id)}},
                    )
                except JudgeModelNotFound as error:
                    st.error(str(error))
                    st.stop()
                
                response = final_state["answer"]
                faithfulness = final_state["faithfulness"]
                relevance = final_state["relevance"]
                
                citations = final_state.get("citations", [])
                abstained = final_state.get("abstained", False)

                st.markdown(response)
                if not abstained:
                    render_sources(citations)
                render_metrics(faithfulness, relevance, abstained)
                
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "citations": citations,
            "abstained": abstained,
            "metrics": {"faithfulness": faithfulness, "relevance": relevance}
        })
