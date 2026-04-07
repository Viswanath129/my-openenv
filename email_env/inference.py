"""
TriageAI Inference Agent v3.0
- Uses classifier confidence to make smarter decisions
- Handles sentiment-aware escalation
- Urgency-based prioritization
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from environment import EmailEnv


def smart_policy(email: dict) -> str:
    """
    Intelligent policy that uses classifier signals.
    Priority: spam detection → urgency → sentiment → default.
    """
    email_type = email.get("type", "WORK").upper()
    urgency = email.get("urgency", "MEDIUM").upper()
    sentiment = email.get("sentiment", "Professional")
    spam_score = email.get("spam_score", 0.0)
    wait = email.get("wait", 0)

    # High-confidence spam → delete immediately
    if email_type == "SPAM" or spam_score > 0.7:
        return "delete"

    # Aggressive sentiment → escalate
    if sentiment == "Aggressive":
        return "escalate"

    # High urgency → open immediately
    if urgency == "HIGH":
        return "open"

    # Support emails → escalate if waiting too long, else open
    if email_type == "SUPPORT":
        return "escalate" if wait > 3 else "open"

    # Work emails → open
    if email_type == "WORK":
        return "open" if wait > 2 else "defer"

    return "defer"


def run_inference(task="task1", max_steps=10):
    env = EmailEnv()
    obs = env.reset(task=task)
    model = "smart_agent_v3"

    print(f"[START] task={task} env=email_env model={model}")

    steps = 0
    score = 0.0
    rewards_list = []
    success = "true"

    try:
        for step in range(max_steps):
            if not env.inbox:
                break

            # Pick highest-priority email (spam first, then high urgency, then oldest)
            inbox = sorted(
                env.inbox,
                key=lambda e: (
                    -(e.get("spam_score", 0)),  # Spam first
                    -(1 if e.get("urgency") == "HIGH" else 0),  # Then urgent
                    -e.get("wait", 0),  # Then oldest
                ),
            )
            target = inbox[0]
            action_type = smart_policy(target)

            action = {"action_type": action_type, "email_id": target["id"]}
            obs, reward, done, info = env.step(action)

            action_str = f"{action_type}_{target['id']}"
            print(
                f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null"
            )

            score += reward
            rewards_list.append(f"{reward:.2f}")
            steps += 1

            if done:
                break
    except Exception as e:
        success = "false"
        print(f"Exception encountered: {e}")
    finally:
        print(
            f"[END] success={success} steps={steps} score={score:.2f} rewards={','.join(rewards_list)}"
        )


if __name__ == "__main__":
    for t in ["task1", "task2", "task3"]:
        run_inference(task=t, max_steps=15)
        print("-" * 50)
