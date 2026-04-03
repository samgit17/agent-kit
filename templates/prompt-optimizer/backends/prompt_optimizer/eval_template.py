"""
eval_template.py — Static eval harness template.

This file is used by eval_builder_node to generate eval.py.
The LLM is NOT used to write eval.py — criteria are injected as data.
This makes eval generation instant and deterministic.
"""

EVAL_TEMPLATE = '''"""
eval.py — Generated eval harness. Do not modify.
Locked in git by eval_locker_node.
"""

import os
import sys
import json
from langchain_core.messages import SystemMessage, HumanMessage

# ── LLM setup ────────────────────────────────────────────────────────────────
# EVAL_MODEL overrides OLLAMA_MODEL so a fast small model can be used
# for scoring while the proposer uses a larger model.

provider = os.getenv("LLM_PROVIDER", "ollama").lower()
if provider == "openai":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("EVAL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")), temperature=0)
elif provider == "anthropic":
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model=os.getenv("EVAL_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")), temperature=0)
else:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("EVAL_MODEL", os.getenv("OLLAMA_MODEL", "qwen3.5:27b")),
        temperature=0,
    )

# ── Injected config (do not modify) ──────────────────────────────────────────

SKILL_FILE = {skill_file!r}
TEST_INPUTS = {test_inputs!r}
EVAL_CRITERIA = {eval_criteria!r}
OUTPUTS_PER_INPUT = {outputs_per_round}

# ── Eval logic ────────────────────────────────────────────────────────────────

def generate_output(skill: str, test_input: str) -> str:
    response = llm.invoke([
        SystemMessage(content=skill),
        HumanMessage(content=test_input),
    ])
    return response.content


def score_output(output: str, criteria: list[str]) -> int:
    criteria_text = "\\n".join(f"{{i+1}}. {{c}}" for i, c in enumerate(criteria))
    prompt = f"""Score this output against each criterion. Answer ONLY with a JSON array of 1 (pass) or 0 (fail).

OUTPUT:
{{output}}

CRITERIA:
{{criteria_text}}

Return ONLY a JSON array like [1, 0, 1, 1, 0]. No explanation."""
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    scores = json.loads(raw.strip())
    return sum(scores)


def main():
    skill_path = sys.argv[1] if len(sys.argv) > 1 else SKILL_FILE
    skill = open(skill_path, encoding="utf-8").read()

    total_passed = 0
    total_possible = len(TEST_INPUTS) * OUTPUTS_PER_INPUT * len(EVAL_CRITERIA)

    for test_input in TEST_INPUTS:
        for _ in range(OUTPUTS_PER_INPUT):
            output = generate_output(skill, test_input)
            total_passed += score_output(output, EVAL_CRITERIA)

    score = total_passed / total_possible if total_possible > 0 else 0.0
    print(f"SCORE: {{score:.4f}}")


if __name__ == "__main__":
    main()
'''
