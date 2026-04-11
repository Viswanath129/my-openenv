#!/usr/bin/env python3
"""
Test Task3 with multiple seeds
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from environment import EmailEnv
from models import Action


def test_task3_with_seed(seed):
    env = EmailEnv()
    env.reset(task="task3", seed=seed)

    step = 0
    while step < 20:
        if not env.inbox:
            action = Action(action_type="defer", email_id="none")
            obs = env.step(action)
            if obs.done:
                break
            step += 1
            continue

        for email in env.inbox[:]:
            is_spam = email.get("type") == "SPAM" or email.get("ground_truth", 0) == 1
            if is_spam:
                action_type = "delete"
            else:
                urgency = email.get("urgency", "MEDIUM")
                sentiment = email.get("sentiment", "Professional")
                if urgency == "HIGH" or sentiment == "Aggressive":
                    action_type = "escalate"
                else:
                    action_type = "open"
            action = Action(action_type=action_type, email_id=email["id"])
            obs = env.step(action)
            if obs.done:
                break

        step += 1

    score = env.grader()
    return env.total_reward, score


if __name__ == "__main__":
    print("Testing Task3 with different seeds:")
    for seed in [0, 1, 2, 3, 42]:
        total, score = test_task3_with_seed(seed)
        print(f"Seed {seed}: total_reward={total:.4f}, score={score:.4f}")
