from openai import OpenAI
from dotenv import load_dotenv
from coffee_assistant import retrieval
from time import time
from datetime import date
import json, os, requests

load_dotenv()
client = OpenAI()


def get_index():
    return retrieval.get_index()


def search(query, num_results=10):
    return retrieval.hybrid_search(query, num_results=num_results)


prompt_template = """
You're a coffee assistant. You only answer questions about coffee: beans, growing,
roasting, brewing, drinks, caffeine, and coffee history and business.
Answer the QUESTION with facts from the CONTEXT (our coffee database).

Rules:
- Answer in the language the QUESTION is written in. Coffee words from other languages (crema, latte, café) don't change it.
- If the QUESTION isn't about coffee, don't search the web. Say in one sentence that you only answer coffee questions.
- For greetings or small talk, reply in one short friendly sentence and invite a coffee question.
- These two short replies get no source line.
- Use the CONTEXT first. Use the web_search tool only if the CONTEXT can't answer or the QUESTION needs current data.
- If the web results don't answer the QUESTION, search again with a different query.
- Work out relative dates ("last month", "this year") from TODAY before you search or answer.
- Don't substitute a related but different figure (for example a fatal dose when asked about recommended intake).
- Write a short answer in plain text. No LaTeX or formulas, just the result (for example "about $6.09 per kg").
- End the answer with its source, for example (source: coffee database) or (source: www.website.com).

TODAY: {today}

QUESTION: {question}

CONTEXT:
{context}
""".strip()


rewrite_template = """
You rewrite user questions for a search engine over coffee-related Wikipedia articles.
Fix spelling mistakes and rephrase the question so it is clear and specific, using the
terms an encyclopedia article would use. Keep the original meaning. Don't answer it and
don't add new facts.
If the QUESTION refers to the CHAT HISTORY (for example "it", "that one", "and for espresso?"),
turn it into a standalone question that names the topic from the history.

Return only the rewritten question.

CHAT HISTORY:
{history}

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

    prompt = prompt_template.format(today=date.today().isoformat() ,question=query, context=context).strip()
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

MAX_HISTORY = 6  # last 6 messages = 3 question/answer turns: enough context, few tokens


def format_history(history):
    history = history[-MAX_HISTORY:]
    messages = []
    
    for message in history:
        messages.append(f"{message['role']}: {message['content']}")

    return '\n'.join(messages)

def rewrite_query(query, history=None, model="gpt-4o-mini"):
    rewritten, token_stats = llm(rewrite_template.format(question=query, history=format_history(history or [])), model=model)
    return rewritten.strip(), token_stats


def rag(query, history=None, model="gpt-4o-mini", prompt_template=prompt_template, rewrite=True):
    t0 = time()

    # The rewritten query only drives the search (fixes typos like "coffeine" that keyword
    # search can't match); the user's original question still goes into the answer prompt.
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    search_query = query
    if rewrite:
        search_query, rewrite_tokens = rewrite_query(query, history, model=model)
        for key in token_stats:
            token_stats[key] += rewrite_tokens[key]

    search_results = search(search_query)
    prompt = build_prompt(query, search_results,prompt_template)
    answer, answer_tokens = answer_with_tools(prompt, model=model)
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
        "strict": True,
        "description": "Search the web for coffee facts that are not in the coffee database, "
                       "or that change over time (prices, news).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def chat(messages, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model, messages=messages, tools=[WEB_SEARCH_TOOL]
    )
    message = response.choices[0].message

    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return message, token_stats


MAX_RESULT_CHARS = 500
def web_search(query):
    key = os.environ["TAVILY_API_KEY"]
    response = requests.post("https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {key}"},
            json={"query": query, "max_results": 3}, timeout=10)
    response.raise_for_status()
    results = response.json()["results"]
    lines=[]
    for r in results:
        lines.append(f"{r['content'][:MAX_RESULT_CHARS]} (source: {r['url']})")
    return "\n".join(lines)

MAX_TOOL_ROUNDS = 5  # safety cap: a model that keeps calling tools can't loop forever

TOOLS = {"web_search": web_search}

def run_tool(name, args):
    if name not in TOOLS:
        return f"Unknown tool: {name}"
    tool = TOOLS[name]
    try:
        return tool(**args)
    except TypeError as error:
        return f"Error: {error}. Call {name} again with the right arguments."

def answer_with_tools(question, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": question}]
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for _ in range(MAX_TOOL_ROUNDS):
        message, message_tokens = chat(messages, model=model)
        for key in message_tokens:
            token_stats[key] += message_tokens[key]
        
        if not message.tool_calls:
            return message.content, token_stats
        messages.append(message)
        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            results = run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content":results})
    raise RuntimeError("Too many tool rounds")
