import random

import streamlit as st
from coffee_assistant.rag import rag
from coffee_assistant import retrieval, db

st.set_page_config(page_title="Coffee Assistant", page_icon="☕")

with st.spinner("Loading knowledge base..."): retrieval.get_vector_index()

st.title("☕ Coffee Assistant")
st.caption(
    "Ask me anything about coffee, like origins, roasting, brewing methods or regions, and "
    "I'll answer from a set of coffee Wikipedia articles."
)

# Small hand-picked pool of simple, beginner-friendly openers. The LLM-eval ground-truth
# questions (data/rag-eval-default.csv) are all 9+ words by construction (one specific fact
# per chunk), too detailed to welcome a first-time visitor. 3 are sampled at random per
# browser session (not reseeded on every rerun) via session_state, so each visit gets a
# different trio but they don't shuffle mid-interaction.
EXAMPLE_QUESTION_POOL = [
    "What is Arabica coffee?",
    "What is Robusta coffee?",
    "How is coffee roasted?",
    "Espresso vs. drip coffee?",
    "How is decaf coffee made?",
    "What is cold brew coffee?",
    "Where does coffee come from?",
    "What is a moka pot?",
    "How much caffeine is in coffee?",
    "What is fair trade coffee?",
]

if "example_questions" not in st.session_state:
    st.session_state.example_questions = random.sample(EXAMPLE_QUESTION_POOL, 3)

example_questions = st.session_state.example_questions

st.write("Try an example, or type your own question below:")
cols = st.columns(len(example_questions))
for col, question in zip(cols, example_questions):
    with col:
        if st.button(question, key=f"example_{question}", use_container_width=True):
            st.session_state.example = question
            st.rerun()

def show(text):
    st.markdown(text.replace("$", "\\$"))  # "$" would start a math formula in Streamlit

def save_choice(conversation_id, key):
    choice = st.session_state[key]  # st.feedback("thumbs"): 1 = up, 0 = down
    if choice is not None:
        db.save_feedback(conversation_id, score=1 if choice == 1 else -1)


def thumbs(conversation_id):
    # same key on every rerun, so the chosen thumb stays highlighted
    # (not disabled after a vote: disabled greys out both thumbs and hides the choice)
    key = f"feedback_{conversation_id}"
    st.feedback("thumbs", key=key, on_change=save_choice, args=(conversation_id, key))


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        show(message["content"])
        if "caption" in message:
            st.caption(message["caption"])
        if message.get("conversation_id") is not None:
            thumbs(message["conversation_id"])

question = st.chat_input("Ask about coffee...") or st.session_state.pop("example", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching"):
            answer = rag(question, history=st.session_state.messages[:-1])
        show(answer["answer"])
        caption = f"⏱️ {answer['response_time']:.2f}s · 💰 ${answer['cost']:.4f}"
        st.caption(caption)
        conversation_id = db.save_conversation(question, answer)
        thumbs(conversation_id)

    st.session_state.messages.append({"role": "assistant", "content": answer["answer"],
                                      "caption": caption, "conversation_id": conversation_id})