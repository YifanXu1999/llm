import argparse
import json
import os
import random
import re
import time
from typing import List, Optional
from urllib.request import Request, urlopen

import wikienv
import wrappers


ROOT = os.path.dirname(os.path.abspath(__file__))
RULES_DIR = os.path.join(ROOT, "rules")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_rules() -> dict:
    return {
        "skill": read_text(os.path.join(ROOT, "SKILL.md")),
        "initial": read_text(os.path.join(RULES_DIR, "initial.md")),
        "search": read_text(os.path.join(RULES_DIR, "search.md")),
        "lookup": read_text(os.path.join(RULES_DIR, "lookup.md")),
        "finish": read_text(os.path.join(RULES_DIR, "finish.md")),
    }


def http_json(url: str, payload: Optional[dict] = None, timeout: int = 60) -> dict:
    if payload is None:
        req = Request(url, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_completion(base_url: str, model: str, messages: List[dict], temperature: float = 0.2) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = http_json(url, payload=payload, timeout=60)
    return data["choices"][0]["message"]["content"].strip()


def parse_action(text: str):
    thought_match = re.search(r"^Thought:\s*(.+)$", text, flags=re.MULTILINE)
    action_match = re.search(r"^Action:\s*([a-zA-Z]+)\[(.*)\]\s*$", text, flags=re.MULTILINE)
    thought = thought_match.group(1).strip() if thought_match else ""
    if not action_match:
        raise ValueError(f"Could not parse Action from LLM output:\n{text}")
    action_type = action_match.group(1).lower()
    action_arg = action_match.group(2).strip()
    return thought, action_type, action_arg


def build_orchestrator_messages(claim: str, history: List[dict], rules: dict) -> List[dict]:
    system = (
        "You are the main skill orchestrator. Use ONLY the main skill instructions below. "
        "Decide which subskill should be applied next. "
        "Output exactly one line in the format: Subskill: <initial|search|lookup|finish>\n\n"
        + rules["skill"]
    )
    user_lines = [f"Claim: {claim}"]
    if history:
        user_lines.append("History:")
        for i, h in enumerate(history, 1):
            user_lines.append(f"Step {i}")
            user_lines.append(f"Subskill: {h['subskill']}")
            user_lines.append(f"Thought: {h['thought']}")
            user_lines.append(f"Action: {h['action']}")
            user_lines.append(f"Observation: {h['observation']}")
    user_lines.append("Choose the next subskill.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def parse_subskill(text: str) -> str:
    m = re.search(
        r"subskill\s*[:\-]\s*(initial|search|lookup|finish)(?:\.md)?",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()
    lowered = text.lower()
    for name in ("initial", "search", "lookup", "finish"):
        if name in lowered:
            return name
    raise ValueError(f"Could not parse Subskill from LLM output:\n{text}")


def build_subskill_messages(claim: str, history: List[dict], rules: dict, subskill: str) -> List[dict]:
    rule_text = rules[subskill]
    system = (
        "You are executing a strict ReAct-style fact verification workflow. "
        "You must output exactly one Thought line and one Action line, nothing else.\n\n"
        + rules["skill"]
        + "\n\n"
        + rule_text
    )
    user_lines = [f"Claim: {claim}"]
    if history:
        user_lines.append("History:")
        for i, h in enumerate(history, 1):
            user_lines.append(f"Step {i}")
            user_lines.append(f"Subskill: {h['subskill']}")
            user_lines.append(f"Thought: {h['thought']}")
            user_lines.append(f"Action: {h['action']}")
            user_lines.append(f"Observation: {h['observation']}")
    user_lines.append("Provide the next Thought and Action only.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def webthink(
    env,
    idx: int,
    base_url: str,
    model: str,
    max_steps: int = 7,
    to_print: bool = False,
    log_steps: bool = False,
):
    claim = env.reset(idx)
    if to_print:
        print(f"Claim: {claim}")
    history = []
    rules = load_rules()

    for step in range(max_steps):
        orch_messages = build_orchestrator_messages(claim, history, rules)
        orch_output = chat_completion(base_url, model, orch_messages)
        subskill = parse_subskill(orch_output)

        messages = build_subskill_messages(claim, history, rules, subskill)
        output = chat_completion(base_url, model, messages)
        thought, action_type, action_arg = parse_action(output)
        action_str = f"{action_type}[{action_arg}]"

        if action_type == "finish":
            pred = action_arg.upper()
            gold = env.gold()
            em = 1 if pred == gold else 0
            if to_print or log_steps:
                print(f"Step {step + 1}")
                print(f"Subskill: {subskill}")
                print(f"Thought: {thought}")
                print(f"Action: finish[{pred}]")
                print()
            info = {
                "pred": pred,
                "gold": gold,
                "em": em,
                "steps": step + 1,
                "history": history,
            }
            return pred, info

        observation = env.step(action_str)
        history.append({
            "subskill": subskill,
            "thought": thought,
            "action": action_str,
            "observation": observation,
        })
        if to_print or log_steps:
            print(f"Step {step + 1}")
            print(f"Subskill: {subskill}")
            print(f"Thought: {thought}")
            print(f"Action: {action_str}")
            print(f"Observation: {observation}")
            print()

    pred = "NOT ENOUGH INFO"
    gold = env.gold()
    em = 1 if pred == gold else 0
    if to_print or log_steps:
        print(f"Step {max_steps}")
        print("Subskill: finish")
        print("Thought: Step limit reached. Returning NOT ENOUGH INFO.")
        print("Action: finish[NOT ENOUGH INFO]")
        print()
    info = {
        "pred": pred,
        "gold": gold,
        "em": em,
        "steps": max_steps,
        "history": history,
    }
    return pred, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--to-print", action="store_true", default=True)
    parser.add_argument("--log-steps", action="store_true", help="Print logs for each step")
    args = parser.parse_args()

    env = wikienv.WikiEnv()
    env = wrappers.FeverWrapper(env, split=args.split)
    env = wrappers.LoggingWrapper(env)

    idxs = list(range(len(env.env.data)))
    random.Random(args.seed).shuffle(idxs)

    rs = []
    infos = []
    old_time = time.time()
    for i in idxs[: args.n]:
        _r, info = webthink(
            env,
            i,
            base_url=args.base_url,
            model=args.model,
            max_steps=args.max_steps,
            to_print=args.to_print,
            log_steps=args.log_steps,
        )
        rs.append(info["em"])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print("-----------")
        print()


if __name__ == "__main__":
    main()
