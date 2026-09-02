from openai import OpenAI
from dotenv import load_dotenv
from coffee_assistant import ingest


load_dotenv()
client = OpenAI()

_index = None

def get_index():
    global _index
    if _index is None:
        _index = ingest.load_index()
    return _index


def search(query, index):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=4
    )

    return results


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


def build_prompt(query, search_results):
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


def rag(query, model="gpt-4o-mini"):
    index = get_index()
    search_results = search(query, index)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)

    answer_data = {
        "answer": answer,
        "model_used": model,
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
    }
    return answer_data