#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen


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


def http_json(url: str, payload: Optional[dict] = None, timeout: int = 60, headers: Optional[dict] = None) -> dict:
    if payload is None:
        req = Request(url, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
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


def parse_action(text: str) -> Tuple[str, Optional[str], str]:
    # Expect a Thought line and Action line; return action_type, action_arg, thought
    thought_match = re.search(r"^Thought:\s*(.+)$", text, flags=re.MULTILINE)
    action_match = re.search(r"^Action:\s*([a-zA-Z]+)\[(.*)\]\s*$", text, flags=re.MULTILINE)
    thought = thought_match.group(1).strip() if thought_match else ""
    if not action_match:
        raise ValueError(f"Could not parse Action from LLM output:\n{text}")
    action_type = action_match.group(1).lower()
    action_arg = action_match.group(2).strip()
    return action_type, action_arg, thought


def wiki_search(entity: str) -> Tuple[str, Optional[str], Optional[str]]:
    # Returns (observation, page_title, full_text)
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": entity,
        "format": "json",
    }
    wiki_headers = {"User-Agent": "Tiny_Agent_Skills/0.1 (local)"}
    sdata = http_json(
        "https://en.wikipedia.org/w/api.php?" + urlencode(search_params),
        timeout=30,
        headers=wiki_headers,
    )
    results = sdata.get("query", {}).get("search", [])
    if not results:
        return "Similar: []", None, None

    title = results[0]["title"]
    extract_params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": 1,
        "exsentences": 5,
        "format": "json",
    }
    edata = http_json(
        "https://en.wikipedia.org/w/api.php?" + urlencode(extract_params),
        timeout=30,
        headers=wiki_headers,
    )
    pages = edata.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    intro = page.get("extract", "").strip()

    full_params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": 1,
        "format": "json",
    }
    fdata = http_json(
        "https://en.wikipedia.org/w/api.php?" + urlencode(full_params),
        timeout=30,
        headers=wiki_headers,
    )
    fpages = fdata.get("query", {}).get("pages", {})
    fpage = next(iter(fpages.values()))
    full_text = fpage.get("extract", "").strip()

    observation = intro if intro else "Similar: []"
    return observation, title, full_text


def split_sentences(text: str) -> List[str]:
    # Simple sentence split; good enough for prototype
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def wiki_lookup(full_text: str, keyword: str) -> str:
    if not full_text:
        return "No match found"
    sentences = split_sentences(full_text)
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    for sentence in sentences:
        if pattern.search(sentence):
            return sentence
    return "No match found"


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
    # Allow either "initial" or "initial.md" etc., and be tolerant of extra text.
    m = re.search(
        r"subskill\\s*[:\\-]\\s*(initial|search|lookup|finish)(?:\\.md)?",
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


def run_loop(
    claim: str,
    base_url: str,
    model: str,
    max_steps: int = 7,
    stream: bool = False,
) -> List[dict]:
    rules = load_rules()
    history = []
    current_full_text = None
    lookup_count = 0

    for step in range(max_steps):
        orch_messages = build_orchestrator_messages(claim, history, rules)
        orch_output = chat_completion(base_url, model, orch_messages)
        subskill = parse_subskill(orch_output)

        messages = build_subskill_messages(claim, history, rules, subskill)
        output = chat_completion(base_url, model, messages)
        action_type, action_arg, thought = parse_action(output)

        if action_type != "lookup":
            lookup_count = 0

        if action_type == "search":
            observation, _title, full_text = wiki_search(action_arg)
            current_full_text = full_text
            action_str = f"search[{action_arg}]"
        elif action_type == "lookup":
            lookup_count += 1
            if lookup_count > 3:
                # Enforce lookup limit (3 total lookups).
                observation = ""
                action_str = "finish[NOT ENOUGH INFO]"
                step_record = {
                    "subskill": subskill,
                    "thought": thought,
                    "action": action_str,
                    "observation": observation,
                }
                history.append(step_record)
                if stream:
                    step_num = len(history)
                    print(f"Step {step_num}")
                    print(f"Subskill: {step_record['subskill']}")
                    print(f"Thought: {step_record['thought']}")
                    print(f"Action: {step_record['action']}")
                    print()
                break
            observation = wiki_lookup(current_full_text or "", action_arg)
            action_str = f"lookup[{action_arg}]"
        elif action_type == "finish":
            observation = ""
            action_str = f"finish[{action_arg}]"
            step_record = {
                "subskill": subskill,
                "thought": thought,
                "action": action_str,
                "observation": observation,
            }
            history.append(step_record)
            if stream:
                step_num = len(history)
                print(f"Step {step_num}")
                print(f"Subskill: {step_record['subskill']}")
                print(f"Thought: {step_record['thought']}")
                print(f"Action: {step_record['action']}")
                print()
            break
        else:
            raise ValueError(f"Unsupported action: {action_type}")

        step_record = {
            "subskill": subskill,
            "thought": thought,
            "action": action_str,
            "observation": observation,
        }
        history.append(step_record)
        if stream:
            step_num = len(history)
            print(f"Step {step_num}")
            print(f"Subskill: {step_record['subskill']}")
            print(f"Thought: {step_record['thought']}")
            print(f"Action: {step_record['action']}")
            if step_record["observation"]:
                print(f"Observation: {step_record['observation']}")
            print()

    return history


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FEVER skill workflow with a local LLM.")
    parser.add_argument("--claim", required=True, help="Claim to verify")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234", help="LLM base URL")
    parser.add_argument("--model", default="local-model", help="Model name for OpenAI-compatible servers")
    parser.add_argument("--stream", action="store_true", help="Print output at each step")
    args = parser.parse_args()

    print(f"Claim: {args.claim}\n")

    history = run_loop(args.claim, args.base_url, args.model, stream=args.stream)

    if not args.stream:
        print("== FEVER RUN ==")
        for i, h in enumerate(history, 1):
            print(f"Step {i}")
            if "subskill" in h:
                print(f"Subskill: {h['subskill']}")
            print(f"Thought: {h['thought']}")
            print(f"Action: {h['action']}")
            if h["observation"]:
                print(f"Observation: {h['observation']}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
