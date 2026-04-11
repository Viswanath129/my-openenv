#!/usr/bin/env python3
"""
Debug Task3 optimal reward
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from environment import EmailEnv
from models import Action


def debug_task3():
    env = EmailEnv()
    env.reset(task="task3", seed=42)

    print(f"Initial inbox: {len(env.inbox)} emails")
    for i, email in enumerate(env.inbox):
        print(
            f"  {i + 1}. {email['id']}: type={email['type']}, urgency={email['urgency']}, sentiment={email['sentiment']}, spam_score={email['spam_score']}"
        )

    step = 0
    total_reward = 0.0
    while step < 20 and env.inbox:
        # Process all emails optimally
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
            total_reward += obs.reward

            if obs.done:
                break

        step += 1

    print(f"\nAfter {step} steps:")
    print(f"  total_reward = {total_reward:.4f}")
    print(f"  processed_emails = {env.processed_emails}")
    print(f"  initial_email_count = {env.initial_email_count}")
    print(f"  inbox remaining = {len(env.inbox)}")

    score = env.grader()
    print(f"  grader score = {score:.4f}")


if __name__ == "__main__":
    random.seed(42)
    debug_task3()
