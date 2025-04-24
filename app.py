import streamlit as st
from pipeline.rag_engine import get_reranked_chunks, generate_answer
import json
from datetime import datetime
import os
import time

# Centered title and subtitle using HTML
st.markdown("""
    <div style='text-align: center;'>
        <h1>ASKME</h1>
        <p style='font-size: 1.2em; color: grey;'> An AI-powered College Advisor</p>
    </div>
""", unsafe_allow_html=True)

# Ensure output folder exists
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
os.makedirs(output_dir, exist_ok=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input and action row
col1, col2 = st.columns([10, 1])
with col1:
    user_query = st.text_input("", placeholder="Ask a question...")
with col2:
    submit = st.button("➤")

# Answer section rendered directly below the input
if submit and user_query.strip():
    with st.spinner("Thinking..."):
        start_time = time.time()
        top_chunks = get_reranked_chunks(user_query)

        # Re-enable backend logging by restoring generate_answer as-is
        answer = generate_answer(top_chunks, user_query)

        response_time = time.time() - start_time

        st.markdown("""
        <div style='text-align: left; padding-top: 1em;'>
            <h3> Final Answer</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(answer)

        st.markdown(f"⏱ **Response Time:** {response_time:.2f} seconds")

        with st.expander(" View Context Chunks"):
            for i, chunk in enumerate(top_chunks):
                st.markdown(f"**Chunk {i+1}:**\n{chunk}")

        st.session_state.chat_history.append((user_query, answer))

# Show previous Q&A
if st.session_state.chat_history:
    with st.expander(" Previous Q&A"):
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
            st.markdown("---")

# Button to clear chat history
if st.button("🗑 Clear Chat History"):
    st.session_state.chat_history = []
