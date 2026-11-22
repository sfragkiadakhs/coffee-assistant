from coffee_assistant import rag

MOKA_TURN = [
    {"role": "user", "content": "What is a moka pot?"},
    {"role": "assistant", "content": "A stovetop coffee maker."},
]


def test_format_history_one_line_per_message():
    text = rag.format_history(MOKA_TURN)

    assert text == "user: What is a moka pot?\nassistant: A stovetop coffee maker."


def test_format_history_keeps_only_the_last_messages():
    # 10 messages: q0, a0, q1, a1 ... q4, a4
    history = []
    for i in range(5):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})

    text = rag.format_history(history)

    assert text.splitlines() == ["user: q2", "assistant: a2", "user: q3",
                                 "assistant: a3", "user: q4", "assistant: a4"]


def test_format_history_empty():
    assert rag.format_history([]) == ""


def test_rewrite_prompt_contains_history_and_question(monkeypatch):
    prompts = []

    def fake_llm(prompt, model="gpt-4o-mini"):
        prompts.append(prompt)
        return "moka pot brewing time", {}

    monkeypatch.setattr(rag, "llm", fake_llm)

    rag.rewrite_query("How long does it take?", history=MOKA_TURN)

    assert "user: What is a moka pot?" in prompts[0]
    assert "How long does it take?" in prompts[0]


def test_rag_passes_history_to_the_rewrite(monkeypatch):
    seen = {}

    def fake_rewrite_query(query, history=None, model="gpt-4o-mini"):
        seen["history"] = history
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return "moka pot brewing time", tokens

    def fake_answer_with_tools(question, model="gpt-4o-mini"):
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return "About 5 minutes.", tokens

    monkeypatch.setattr(rag, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(rag, "search", lambda query, num_results=10: [])
    monkeypatch.setattr(rag, "answer_with_tools", fake_answer_with_tools)

    rag.rag("How long does it take?", history=MOKA_TURN)

    assert seen["history"] == MOKA_TURN


def test_answer_prompt_contains_todays_date():
    # real run 22.11: "last month" was answered for September 2023 (the model has no clock)
    from datetime import date

    prompt = rag.build_prompt("What did arabica cost last month?", [])

    assert date.today().isoformat() in prompt
