from openai import OpenAI
from dotenv import load_dotenv
from coffee_assistant import retrieval
from time import time


load_dotenv()
client = OpenAI()


def get_index():
    return retrieval.get_index()


def search(query, num_results=10):
    return retrieval.hybrid_search(query, num_results=num_results)


prompt_template = """
You're a coffee assistant. Answer the QUESTION based on the CONTEXT from our coffee database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()


entry_template = """
doc_id: {doc_id}
chunk_id: {chunk_id}
title: {title}
content: {content}
""".strip()


def build_prompt(query, search_results, prompt_template=prompt_template):
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt



def llm(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return answer, token_stats

def calculate_openai_cost(model, tokens):
    openai_cost = 0

    if model == "gpt-4o-mini":
        openai_cost = (
            tokens["prompt_tokens"] * 0.00015 + tokens["completion_tokens"] * 0.0006
        ) / 1000
    else:
        print("Model not recognized. OpenAI cost calculation failed.")

    return openai_cost

def rag(query, model="gpt-4o-mini", prompt_template=prompt_template):
    t0 = time()

    search_results = search(query)
    prompt = build_prompt(query, search_results,prompt_template)
    answer, token_stats = llm(prompt, model=model)
    openai_cost_rag = calculate_openai_cost(model, token_stats)

    t1 = time()
    took = t1 - t0

    answer_data = {
        "answer": answer,
        "model_used": model,
        "question": query,
        "prompt": prompt,
        "response_time": took,
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "cost": openai_cost_rag,
    }
    return answer_data