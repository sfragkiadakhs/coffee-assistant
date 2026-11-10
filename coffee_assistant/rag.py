from openai import OpenAI
from dotenv import load_dotenv
from coffee_assistant import retrieval
from time import time
import json 

load_dotenv()
client = OpenAI()


def get_index():
    return retrieval.get_index()


def search(query, num_results=10):
    return retrieval.hybrid_search(query, num_results=num_results)


prompt_template = """
You're a coffee assistant. Answer the QUESTION based on the CONTEXT from our coffee database.
Use only the facts from the CONTEXT when answering the QUESTION.
Don't substitute a related but different figure (for example a toxic or fatal dose when
asked about recommended intake).

QUESTION: {question}

CONTEXT:
{context}
""".strip()


rewrite_template = """
You rewrite user questions for a search engine over coffee-related Wikipedia articles.
Fix spelling mistakes and rephrase the question so it is clear and specific, using the
terms an encyclopedia article would use. Keep the original meaning. Don't answer it and
don't add new facts.

Return only the rewritten question.

QUESTION: {question}
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

def rewrite_query(query, model="gpt-4o-mini"):
    rewritten, token_stats = llm(rewrite_template.format(question=query), model=model)
    return rewritten.strip(), token_stats


def rag(query, model="gpt-4o-mini", prompt_template=prompt_template, rewrite=True):
    t0 = time()

    # The rewritten query only drives the search (fixes typos like "coffeine" that keyword
    # search can't match); the user's original question still goes into the answer prompt.
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    search_query = query
    if rewrite:
        search_query, rewrite_tokens = rewrite_query(query, model=model)
        for key in token_stats:
            token_stats[key] += rewrite_tokens[key]

    search_results = search(search_query)
    prompt = build_prompt(query, search_results,prompt_template)
    answer, answer_tokens = llm(prompt, model=model)
    for key in token_stats:
        token_stats[key] += answer_tokens[key]
    openai_cost_rag = calculate_openai_cost(model, token_stats)

    t1 = time()
    took = t1 - t0

    answer_data = {
        "answer": answer,
        "model_used": model,
        "question": query,
        "search_query": search_query,
        "prompt": prompt,
        "response_time": took,
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "cost": openai_cost_rag,
    }
    return answer_data


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for coffee facts that are not in the coffee database, "
                       "or that change over time (prices, news).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
}


def chat(messages, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model, messages=messages, tools=[WEB_SEARCH_TOOL]
    )
    return response.choices[0].message


def web_search(query):
    raise NotImplementedError("Web provider comes in the next step")


MAX_TOOL_ROUNDS = 5  # safety cap: a model that keeps calling tools can't loop forever


def answer_with_tools(question, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": question}]
    for _ in range(MAX_TOOL_ROUNDS):
        message = chat(messages, model=model)
        if not message.tool_calls:
            return message.content
        messages.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            results = web_search(**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content":results})
    raise RuntimeError("Too many tool rounds")
