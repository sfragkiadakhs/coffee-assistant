from pathlib import Path

import pandas as pd
import streamlit as st
from coffee_assistant.rag import rag
from coffee_assistant import retrieval, db

st.set_page_config(page_title="Coffee Assistant", page_icon="☕")

with st.spinner("Loading knowledge base..."): retrieval.get_vector_index()

st.title("☕ Coffee Assistant")

EVAL_CSV = Path(__file__).resolve().parent / "data" / "rag-eval-default.csv"

# Sample 3 example questions from the LLM-eval results, keeping only ones the judge
# scored RELEVANT (so the demo never leads with one flagged PARTLY_RELEVANT/NON_RELEVANT).
# Picked once per browser session (not reseeded on every rerun) via session_state, so
# each visit gets a different trio but they don't shuffle mid-interaction.
if "example_questions" not in st.session_state:
    eval_df = pd.read_csv(EVAL_CSV)
    relevant = eval_df[eval_df["relevance"] == "RELEVANT"]
    st.session_state.example_questions = relevant["question"].sample(3).tolist()

example_questions = st.session_state.example_questions

st.write("Try an example:")
cols = st.columns(len(example_questions))
for col, question in zip(cols, example_questions):
    with col:
        if st.button(question, key=f"example_{question}", use_container_width=True):
            st.session_state.user_input = question
            st.rerun()

with st.form("ask_form"):
    user_input = st.text_input("Enter your question:", key="user_input")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not user_input:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            answer = rag(user_input)
            st.session_state.answer = answer
            st.session_state.conversation_id = db.save_conversation(user_input, answer)

if "answer" in st.session_state:
    answer = st.session_state.answer
    st.success("Completed!")
    st.write(answer["answer"])
    st.caption(f"⏱️ {answer['response_time']:.2f}s · 💰 ${answer['cost']:.4f}")

conversation_id = st.session_state.get("conversation_id")

if conversation_id is not None:
    feedback_key = f"feedback_given_{conversation_id}"
    given = st.session_state.get(feedback_key)

    if given is None:
        st.write("Was this helpful?")
        col1, col2, _ = st.columns([1, 1, 6])

        with col1:
            if st.button("👍", key=f"feedback_up_{conversation_id}"):
                db.save_feedback(conversation_id, score=1)
                st.session_state[feedback_key] = "up"
                st.rerun()

        with col2:
            if st.button("👎", key=f"feedback_down_{conversation_id}"):
                db.save_feedback(conversation_id, score=-1)
                st.session_state[feedback_key] = "down"
                st.rerun()
    else:
        st.caption(f"Thanks for the feedback! {'👍' if given == 'up' else '👎'}")