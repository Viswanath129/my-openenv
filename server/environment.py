"""
InboxIQ RL Environment v3.0
- ML-powered email classification (TF-IDF + Naive Bayes)
- Confidence-weighted reward shaping
- Three difficulty tiers: easy → medium → hard
- Deterministic grading normalized to [0.0, 1.0]
"""

import os
import random
import csv
from typing import List, Dict, Optional, Any

try:
    from server.classifier import EmailClassifier, analyze_sentiment, detect_urgency
    from server.models import Action, Observation, State
except ImportError:
    from .classifier import EmailClassifier, analyze_sentiment, detect_urgency
    from .models import Action, Observation, State

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    # Fallback if OpenEnv not installed
    Environment = object

SENTIMENTS = ["Aggressive", "Professional", "Casual"]

# Resolve dataset path — try multiple locations
_possible_paths = [
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "dataset",
        "spam_assassin.csv",
    ),
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
        "spam_assassin.csv",
    ),
    "/app/dataset/spam_assassin.csv",  # Docker container path
    "dataset/spam_assassin.csv",
    "spam_assassin.csv",
]
_DEFAULT_DATASET = None
for path in _possible_paths:
    if os.path.exists(path):
        _DEFAULT_DATASET = path
        break


class EmailEnv:
    """
    InboxIQ — OpenEnv-compliant RL environment for email triage.

    Interface:
        reset(task) → observation
        step(action) → (observation, reward, done, info)
        state() → current observation
        grader() → normalized score [0.0, 1.0]
    """

    # Enable concurrent sessions since state is isolated per instance
    SUPPORTS_CONCURRENT_SESSIONS = True

    # Task configs: (initial_emails, max_steps, optimal_reward_estimate)
    TASK_CONFIG = {
        "task1": {"count": 1, "max_steps": 5, "optimal": 3.0},
        "task2": {"count": 3, "max_steps": 10, "optimal": 8.0},
        "task3": {"count": 5, "max_steps": 20, "optimal": 15.0},
        "train": {"count": 2, "max_steps": 15, "optimal": 6.0},
    }

    def __init__(self, dataset_path: Optional[str] = None):
        self.inbox: List[Dict] = []
        self.total_reward: float = 0.0
        self.steps: int = 0
        self.max_steps: int = 10
        self.current_task: str = "train"
        self.dataset_path = dataset_path or _DEFAULT_DATASET
        self.raw_data: List[Dict] = []
        self._episode_rewards: List[float] = []
        self.initial_email_count: int = 0  # Track initial inbox size for progress
        self.processed_emails: int = 0  # Track emails processed correctly

        # ML classifier for intelligent grading
        self.classifier = EmailClassifier(dataset_path=self.dataset_path)
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset rows into memory."""
        if not self.dataset_path:
            print("[InboxIQ] No dataset path configured — using simulation fallback.")
            return
        try:
            print(f"[InboxIQ] Loading dataset from {self.dataset_path}...")
            rows = []
            with open(self.dataset_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("text") and row.get("target") is not None:
                        rows.append(row)

            if len(rows) > 500:
                self.raw_data = random.sample(rows, 500)
            else:
                self.raw_data = rows
            print(f"[InboxIQ] Dataset loaded. {len(self.raw_data)} samples available.")
        except Exception as e:
            print(f"[InboxIQ ERROR] Failed to load dataset: {e}")

    def _get_random_email(self, is_train: bool = True) -> Dict:
        """Generate a random email, either from the dataset or via simulation."""
        if is_train and self.raw_data:
            row = random.choice(self.raw_data)
            text = str(row["text"])
            target = int(row["target"])

            subject = "Dataset Message"
            for line in text.split("\n"):
                if line.lower().startswith("subject:"):
                    subject = line[8:].strip()
                    break

            classification = self.classifier.classify(text, subject)

            return {
                "id": f"DB-{random.randint(1000, 9999)}",
                "sender": "dataset@archive.org",
                "subject": subject[:60] + ("..." if len(subject) > 60 else ""),
                "type": "SPAM" if target == 1 else "WORK",
                "urgency": classification["urgency"],
                "sentiment": classification["sentiment"],
                "sentiment_confidence": classification["sentiment_confidence"],
                "spam_score": classification["spam_score"],
                "confidence": classification["confidence"],
                "wait": 0,
                "ground_truth": target,
            }

        # Simulation fallback
        sim_type = random.choice(["WORK", "SUPPORT", "SPAM"])
        return {
            "id": f"MSG-{random.randint(100, 999)}",
            "sender": "bot@simulation.io",
            "subject": "Simulated Email",
            "type": sim_type,
            "urgency": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "sentiment": random.choice(SENTIMENTS),
            "sentiment_confidence": 0.5,
            "spam_score": 0.8 if sim_type == "SPAM" else 0.1,
            "confidence": 0.5,
            "wait": 0,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a new episode. Returns the initial observation."""
        self.total_reward = 0.0
        self.steps = 0
        self.current_task = task or kwargs.get("task", "train")
        self._episode_rewards = []

        if seed is not None:
            random.seed(seed)

        import uuid

        self.episode_id = episode_id or str(uuid.uuid4())

        cfg = self.TASK_CONFIG.get(task, self.TASK_CONFIG["train"])
        self.max_steps = cfg["max_steps"]
        count = cfg["count"]
        self.inbox = [
            self._get_random_email(
                is_train=(task in ["train", "task1", "task2", "task3"])
            )
            for _ in range(count)
        ]
        self.initial_email_count = len(self.inbox)
        self.processed_emails = 0

        # Reset rubric if present
        if hasattr(self, "_reset_rubric"):
            self._reset_rubric()

        return self._create_observation()

    def _is_correct_action(self, email: Dict, action_type: str) -> bool:
        """Determine if the action is correct for the given email."""
        is_spam = email.get("type") == "SPAM" or (email.get("ground_truth", 0) == 1)

        if is_spam:
            return action_type == "delete"
        else:
            # For legitimate emails, opening or escalating is generally correct
            return action_type in ["open", "escalate"]

    def complex_grader(self, email: Dict, action_type: str) -> float:
        """
        Enhanced reward function using classifier confidence.
        Higher confidence → stronger reward signal.
        Provides incremental feedback throughout the trajectory.
        """
        reward = 0.0
        is_spam = email.get("type") == "SPAM"
        if "ground_truth" in email:
            is_spam = email["ground_truth"] == 1

        confidence = email.get("confidence", 0.5)
        confidence_multiplier = 0.5 + confidence  # Range: 0.5 – 1.5

        # ── Core spam/ham reward ──
        if is_spam:
            if action_type == "delete":
                reward += 4.0 * confidence_multiplier
            else:
                reward -= 5.0 * confidence_multiplier
        else:
            if action_type in ["open", "escalate"]:
                reward += 2.0 * confidence_multiplier
            elif action_type == "delete":
                reward -= 6.0  # Critical error — deleting real email

        # ── Sentiment bonus ──
        if email.get("sentiment") == "Aggressive" and action_type == "escalate":
            reward += 3.0
        elif email.get("sentiment") == "Aggressive" and action_type not in [
            "escalate",
            "open",
        ]:
            reward -= 2.0

        # ── Urgency bonus ──
        if email.get("urgency") == "HIGH" and action_type in ["open", "escalate"]:
            reward += 1.0
        elif email.get("urgency") == "HIGH" and action_type == "defer":
            reward -= 2.0

        # ── Wait penalty (capped) ──
        wait_penalty = min(0.2 * email.get("wait", 0), 4.0)
        reward -= wait_penalty

        self.classifier.record_reward(reward)
        return round(reward, 2)

    def step(self, action: Action, timeout_s: Optional[float] = None) -> Observation:
        """
        Execute one step in the environment.
        Returns: Observation with reward, done status, and metadata
        """
        action_type = action.action_type.lower()
        email_id = action.email_id

        target_email = next((e for e in self.inbox if e["id"] == email_id), None)
        step_reward = -0.1  # Small cost per step (penalizes infinite loops)

        # Track observation telemetry for this step
        step_error_trace = None
        current_step_feedback = None

        if target_email:
            step_reward += self.complex_grader(target_email, action_type)
            if action_type in ["open", "delete", "escalate"]:
                self.inbox = [e for e in self.inbox if e["id"] != email_id]
                # Track correct processing
                is_correct_action = self._is_correct_action(target_email, action_type)
                if is_correct_action:
                    self.processed_emails += 1
            current_step_feedback = (
                f"Action '{action_type}' successfully executed on {email_id}."
            )
        else:
            # Graceful System Degradation: intercept hallucinated IDs mathematically
            if email_id and email_id.lower() != "none":
                step_error_trace = f"InvalidTargetError: Action '{action_type}' failed. Identifier '{email_id}' does not exist in inbox."
                step_reward -= 0.5  # minor negative penalty for hallucination
            else:
                current_step_feedback = "Deferred action; no target specified."

        # Add progress-based rewards for investigative loop
        if self.initial_email_count > 0:
            progress_ratio = self.processed_emails / self.initial_email_count
            progress_reward = progress_ratio * 0.5  # Up to 0.5 bonus for full progress
            step_reward += progress_reward

        # Tick wait counters for remaining emails
        for email in self.inbox:
            email["wait"] += 1

        # Dynamic email arrivals (task3 pressure)
        if random.random() < 0.3 and len(self.inbox) < 8:
            self.inbox.append(
                self._get_random_email(is_train=(self.current_task != "eval"))
            )

        # Normalize reward to [0.0, 1.0] range per OpenEnv specification
        # Max possible per-step reward is around 10.0 (spam delete + bonuses)
        normalized_reward = max(0.0, min(1.0, (step_reward + 10.0) / 20.0))

        self.total_reward += step_reward
        self._episode_rewards.append(normalized_reward)
        self.steps += 1
        done = self.steps >= self.max_steps or (len(self.inbox) == 0 and self.steps > 1)

        # ── Completion Bonus ──
        if done and len(self.inbox) == 0:
            bonus = 10.0
            step_reward += bonus
            self.total_reward += bonus

        # Apply rubric if present
        if hasattr(self, "_apply_rubric"):
            rubric_reward = self._apply_rubric(
                action, None
            )  # We don't have observation yet
            if rubric_reward != 0.0:
                normalized_reward = max(
                    0.0, min(1.0, normalized_reward + rubric_reward)
                )

        return self._create_observation(reward=normalized_reward, done=done)

    def grader(self) -> float:
        """
        Normalize episode total reward to [0.0, 1.0] with partial progress credit.
        Includes efficiency bonus for completing tasks with fewer steps.
        Deterministic given the same episode trajectory.
        """
        cfg = self.TASK_CONFIG.get(self.current_task, self.TASK_CONFIG["train"])
        optimal = cfg["optimal"]
        worst = -optimal * 1.5

        # Base score from total reward
        if optimal == worst:
            base_score = 0.5
        else:
            raw = (self.total_reward - worst) / (optimal - worst)
            base_score = min(max(raw, 0.0), 1.0)

        # Progress bonus: reward for processing emails correctly
        progress_score = 0.0
        if self.initial_email_count > 0:
            progress_ratio = self.processed_emails / self.initial_email_count
            progress_score = (
                progress_ratio * 0.3
            )  # Up to 30% bonus for perfect processing

        # Efficiency bonus: reward for completing quickly
        efficiency_score = 0.0
        if self.steps > 0 and self.initial_email_count > 0:
            expected_steps = self.initial_email_count * 1.5  # Expected steps per email
            efficiency_ratio = min(expected_steps / self.steps, 1.0)
            efficiency_score = efficiency_ratio * 0.2  # Up to 20% bonus for efficiency

        # Combine scores
        final_score = base_score + progress_score + efficiency_score
        final_score = max(0.0, min(1.0, final_score))

        return round(final_score, 4)

    def state(self) -> Dict:
        """Return the current environment state as an observation dict."""
        if not hasattr(self, "episode_id") or getattr(self, "episode_id") is None:
            import uuid

            self.episode_id = str(uuid.uuid4())

        return {
            "inbox": self.inbox,
            "steps": self.steps,
            "step_count": self.steps,
            "episode_id": self.episode_id,
            "task": self.current_task,
            "max_steps": self.max_steps,
            "total_reward": round(self.total_reward, 2),
            "classifier_stats": self.classifier.stats,
        }

    def _create_observation(
        self, reward: float = 0.0, done: bool = False
    ) -> Observation:
        """Create an Observation object from current state."""
        info = {
            "total": round(self.total_reward, 2),
            "classifier_stats": self.classifier.stats,
        }

        if done:
            gs = self.grader()
            info["grader_score"] = max(0.01, min(0.99, gs))

        return Observation(
            inbox=self.inbox,
            steps=self.steps,
            task=self.current_task,
            max_steps=self.max_steps,
            total_reward=round(self.total_reward, 2),
            classifier_stats=self.classifier.stats,
            reward=reward,
            done=done,
            info=info,
        )
