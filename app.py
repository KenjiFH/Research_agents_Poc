import os
from dotenv import load_dotenv

# Load the LangSmith keys before the agents boot up
load_dotenv()
import streamlit as st
from document_processor import process_document_and_create_vdb
from agents import create_multi_agent_workflow

st.set_page_config(page_title="Multi-Agent RAG POC", page_icon="🤖", layout="centered")
st.title("🤖 Multi-Agent RAG Assistant")

# Initialize Session State
if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# NEW: State to hold the raw Markdown chunks for debugging
if "raw_chunks" not in st.session_state:
    st.session_state.raw_chunks = []

# Sidebar Logic
with st.sidebar:
    st.header("1. Upload Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    # Condition 1: File uploaded, agents not built yet
    if uploaded_file and st.session_state.workflow is None:
        with st.spinner("Processing document with Docling..."):
            # Catch both the retriever and the chunks
            retriever, chunks = process_document_and_create_vdb(uploaded_file)
            st.session_state.workflow = create_multi_agent_workflow(retriever)
            st.session_state.raw_chunks = chunks  # Save chunks to memory
        st.success("Document processed! Agents are ready.")
        
    # Condition 2: File removed, hard reset
    elif uploaded_file is None and st.session_state.workflow is not None:
        st.session_state.workflow = None
        st.session_state.messages = []
        st.session_state.raw_chunks = []
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
        st.session_state.workflow = None
        st.session_state.messages = []
        st.session_state.raw_chunks = []
        st.rerun()

# Main Chat Interface
st.header("2. Ask the Agents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metrics" in msg:
            col1, col2 = st.columns(2)
            col1.metric("🔍 Faithfulness", f"{msg['metrics']['faithfulness']}%")
            col2.metric("🎯 Relevance", f"{msg['metrics']['relevance']}%")

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
                    "answer": "", 
                    "faithfulness_score": 0, 
                    "relevance_score": 0
                }
                
                final_state = st.session_state.workflow.invoke(initial_state)
                
                response = final_state["answer"]
                f_score = final_state["faithfulness_score"]
                r_score = final_state["relevance_score"]
                
                st.markdown(response)
                col1, col2 = st.columns(2)
                col1.metric("🔍 Faithfulness (Hallucination Check)", f"{f_score}%")
                col2.metric("🎯 Relevance (Accuracy Check)", f"{r_score}%")
                
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "metrics": {"faithfulness": f_score, "relevance": r_score}
        })
