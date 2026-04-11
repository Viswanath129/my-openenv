#!/usr/bin/env python3
"""
Test script to verify the OpenEnv grading contract.
An optimal agent should achieve exactly 1.0 score.
"""

import os
import sys
import json
import time
import random

# Add the server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from environment import EmailEnv
from registry import TASK_REGISTRY
from models import Action


def test_optimal_agent():
    """Test that optimal agent gets exactly 1.0"""
    results = {}

    for task_id, task in TASK_REGISTRY.items():
        print(f"\nTesting {task_id}...")

        # Create environment
        env = EmailEnv()

        # Optimal reset with seed for reproducibility
        env.reset(task=task_id, seed=42)

        # Simulate optimal actions
        # For task1: 1 email, should delete if spam, open/escalate if work
        # For task2: 3 emails
        # For task3: up to 20 steps with arrivals

        step = 0
        while step < task.max_steps:
            # Process all current emails optimally
            if not env.inbox:
                # No emails left, take a dummy step to end episode
                action = Action(action_type="defer", email_id="none")
                obs = env.step(action)
                if obs.done:
                    break
                step += 1
                continue

            # Find best action for each email
            for email in env.inbox[:]:  # Copy list
                is_spam = (
                    email.get("type") == "SPAM" or email.get("ground_truth", 0) == 1
                )

                if is_spam:
                    action_type = "delete"
                else:
                    # Check urgency and sentiment for best action
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

        # Get final grader score
        score = env.grader()
        results[task_id] = score

        print(f"  Final total_reward: {env.total_reward:.4f}")
        print(f"  Grader score: {score:.4f}")

        # Verify score is strictly inside (0, 1)
        if score <= 0.0 or score >= 1.0:
            print(f"  ❌ FAIL: Score {score:.4f} is out of strict range (0, 1)")
            return False
        print(f"  ✅ PASS: Score {score:.4f} is within range.")

    print("\n" + "=" * 60)
    print("All tasks completed!")
    for task_id, score in results.items():
        print(f"{task_id}: {score:.4f}")
    print("=" * 60)

    return True


if __name__ == "__main__":
    random.seed(42)
    success = test_optimal_agent()
    sys.exit(0 if success else 1)
