import streamlit as st
from coffee_assistant.rag import rag, get_index 


with st.spinner("Loading knowledge base..."): get_index()

st.title("Coffee Assistant")

user_input = st.text_input("Enter your question:")


if st.button("Ask"):
    if not user_input:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            answer = rag(user_input)
            st.success("Completed!")
            st.write(answer["answer"])

            st.write(f"Prompt tokens: {answer['prompt_tokens']}")
            st.write(f"Completion tokens: {answer['completion_tokens']}")
            st.write(f"Total tokens: {answer['total_tokens']}")
  

        
        