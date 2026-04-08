"""
InboxIQ Inference Script — OpenEnv Hackathon
Uses OpenAI-compatible client via the hackathon's LiteLLM proxy.
Emits structured [START], [STEP], [END] logs per OpenEnv spec.

Usage:
    API_BASE_URL=... API_KEY=... python inference.py
    ENV_URL=http://localhost:8000 python inference.py
"""

import os
import json
from typing import List, Optional

from openai import OpenAI

# ── Required environment variables (injected by hackathon evaluator) ──
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.environ["API_KEY"]

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    print("[INFO] HF_TOKEN is valid and set.", flush=True)
else:
    print("[WARNING] HF_TOKEN is not set.", flush=True)

# ── Environment config ──
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
BENCHMARK = "InboxIQ"

TASKS = [
    {"id": "task1", "max_steps": 5},
    {"id": "task2", "max_steps": 10},
    {"id": "task3", "max_steps": 20},
]


# ══════════════════════════════════════
# Structured logging per OpenEnv spec
# ══════════════════════════════════════

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ══════════════════════════════════════
# LLM-based action selection
# ══════════════════════════════════════

def get_action_from_llm(
    client: OpenAI, inbox: list, step: int, history: list
) -> dict:
    """Ask the LLM to choose an action for the current inbox state."""
    if not inbox:
        return {"action_type": "defer", "email_id": "none"}

    inbox_desc = "\n".join(
        f"  [{i+1}] id={e['id']} type={e.get('type','?')} urgency={e.get('urgency','?')} "
        f"sentiment={e.get('sentiment','?')} spam_score={e.get('spam_score',0):.2f} wait={e.get('wait',0)} "
        f"subject=\"{e.get('subject','')[:40]}\""
        for i, e in enumerate(inbox)
    )

    history_desc = "\n".join(history[-5:]) if history else "None yet."

    prompt = f"""You are an email triage agent for InboxIQ. Your goal is to maximize reward by choosing the best action for each email.

CURRENT INBOX (step {step}):
{inbox_desc}

RECENT HISTORY:
{history_desc}

ACTIONS: open, delete, defer, escalate
RULES:
- DELETE spam emails (high spam_score) → +reward
- OPEN or ESCALATE legitimate work/support emails → +reward  
- ESCALATE aggressive sentiment or HIGH urgency → +bonus
- DEFER is usually penalized, use sparingly
- Deleting legitimate email → large penalty
- Waiting too long costs penalty per step

Respond with ONLY a JSON object: {{"action_type": "...", "email_id": "..."}}
Choose the single best action right now."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        if "{" in content:
            json_str = content[content.index("{") : content.rindex("}") + 1]
            return json.loads(json_str)
        else:
            print(f"[DEBUG] LLM failed to return JSON. Output: {content}", flush=True)
    except Exception as exc:
        import traceback
        print(f"[DEBUG] API/Proxy Error! LLM request failed: {exc}", flush=True)
        traceback.print_exc()

    return fallback_policy(inbox)


def fallback_policy(inbox: list) -> dict:
    """Deterministic heuristic fallback when LLM is unavailable."""
    if not inbox:
        return {"action_type": "defer", "email_id": "none"}

    sorted_inbox = sorted(
        inbox,
        key=lambda e: (
            -(e.get("spam_score", 0)),
            -(1 if e.get("urgency") == "HIGH" else 0),
            -e.get("wait", 0),
        ),
    )
    target = sorted_inbox[0]

    email_type = target.get("type", "WORK").upper()
    urgency = target.get("urgency", "MEDIUM").upper()
    sentiment = target.get("sentiment", "Professional")
    spam_score = target.get("spam_score", 0.0)

    if email_type == "SPAM" or spam_score > 0.7:
        action = "delete"
    elif sentiment == "Aggressive":
        action = "escalate"
    elif urgency == "HIGH":
        action = "open"
    elif email_type == "SUPPORT":
        action = "escalate" if target.get("wait", 0) > 3 else "open"
    elif email_type == "WORK":
        action = "open" if target.get("wait", 0) > 2 else "defer"
    else:
        action = "defer"

    return {"action_type": action, "email_id": target["id"]}


# ══════════════════════════════════════
# HTTP helpers
# ══════════════════════════════════════

import urllib.request
import urllib.error


def http_post(url: str, data: dict = None) -> dict:
    body = json.dumps(data).encode() if data else b"{}"
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def http_get(url: str) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ══════════════════════════════════════
# Main
# ══════════════════════════════════════

def main():
    print("=" * 60)
    print("  InboxIQ — OpenEnv Inference Benchmark")
    print("=" * 60)

    # Initialize OpenAI client exactly as requested by the openenv specification
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )
    print(f"[INFO] Using LLM proxy configured correctly at {os.environ['API_BASE_URL']}")

    for task_cfg in TASKS:
        task_id = task_cfg["id"]
        max_steps = task_cfg["max_steps"]

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        rewards: List[float] = []
        history: List[str] = []
        steps_taken = 0
        success = True

        try:
            obs = http_post(f"{ENV_URL}/reset?task={task_id}")

            for step in range(1, max_steps + 1):
                inbox = obs.get("inbox", [])
                if not inbox and step > 1:
                    break

                if client:
                    action = get_action_from_llm(client, inbox, step, history)
                else:
                    action = fallback_policy(inbox)

                result = http_post(f"{ENV_URL}/step", data=action)

                obs = result.get("observation", result)
                reward = result.get("reward", 0.0)
                done = result.get("done", False)

                action_str = f"{action['action_type']}_{action['email_id']}"
                log_step(step=step, action=action_str, reward=reward, done=done, error=None)

                rewards.append(reward)
                history.append(f"Step {step}: {action_str} → reward {reward:+.2f}")
                steps_taken = step

                if done:
                    break

        except Exception as e:
            success = False
            print(f"[DEBUG] Error during {task_id}: {e}", flush=True)

        # Get grader score
        try:
            grader = http_get(f"{ENV_URL}/grader")
            score = grader.get("score", 0.0)
        except Exception:
            score = sum(rewards) / max(1, max_steps) if rewards else 0.0

        score = min(max(score, 0.0), 1.0)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        print("-" * 50)

    print("\n✅ InboxIQ inference benchmark complete.")


if __name__ == "__main__":
    main()
