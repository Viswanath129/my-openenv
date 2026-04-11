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
    from server.registry import TASK_REGISTRY
except ImportError:
    from .classifier import EmailClassifier, analyze_sentiment, detect_urgency
    from .models import Action, Observation, State
    from .registry import TASK_REGISTRY

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
        self.trajectory: List[Dict[str, Any]] = []

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

        task_info = TASK_REGISTRY.get(task, TASK_REGISTRY.get("task1"))
        self.max_steps = task_info.max_steps
        count = task_info.config.get("count", 1)
        self.inbox = [
            self._get_random_email(
                is_train=(task in ["train", "task1", "task2", "task3"])
            )
            for _ in range(count)
        ]
        self.initial_email_count = len(self.inbox)
        self.processed_emails = 0
        self.trajectory = []

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
        Native 0.0 – 1.0 reward function.
        Every action maps to a reward strictly within [0.0, 1.0].
        No negative values exist.
        """
        is_spam = email.get("type") == "SPAM"
        if "ground_truth" in email:
            is_spam = email["ground_truth"] == 1

        confidence = email.get("confidence", 0.5)
        urgency = email.get("urgency", "MEDIUM")
        sentiment = email.get("sentiment", "Professional")

        # ── Action-specific reward mapping ──
        if action_type == "delete":
            if is_spam:
                reward = 0.4 + (0.4 * confidence)  # 0.4–0.8
            else:
                reward = 0.0  # CRITICAL: Deleting real mail = total failure
        elif action_type == "open":
            if not is_spam:
                reward = 0.35 + (0.35 * confidence)  # 0.35–0.70
            else:
                reward = 0.05  # Opening spam = near-failure
        elif action_type == "escalate":
            if urgency == "HIGH" or sentiment == "Aggressive":
                reward = 0.7 + (0.2 * confidence)  # 0.7–0.9
            elif not is_spam:
                reward = 0.3  # Unnecessary escalation
            else:
                reward = 0.05  # Escalating spam
        elif action_type == "defer":
            if urgency == "HIGH":
                reward = 0.1  # Deferring urgent = bad
            else:
                reward = 0.2  # Low-urgency defer = acceptable
        else:
            reward = 0.05  # Unknown action

        # ── Wait Decay (multiplier, never goes negative) ──
        wait_steps = email.get("wait", 0)
        reward *= (0.9 ** wait_steps)

        # ── Final clamp ──
        reward = float(max(0.0, min(1.0, reward)))
        self.classifier.record_reward(reward)
        return round(reward, 4)

    def step(self, action: Action, timeout_s: Optional[float] = None) -> Observation:
        """
        Execute one step in the environment.
        Returns: Observation with reward, done status, and metadata
        """
        action_type = action.action_type.lower()
        email_id = action.email_id

        target_email = next((e for e in self.inbox if e["id"] == email_id), None)
        step_reward = 0.05  # Minimal baseline reward for participating (0.0–1.0 compliant)

        # Track observation telemetry for this step
        step_error_trace = None
        current_step_feedback = None

        if target_email:
            step_reward = self.complex_grader(target_email, action_type)
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
                step_reward = 0.05  # Near-zero reward for hallucinated target
            else:
                step_reward = 0.1  # Small reward for choosing to defer
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

        # ── HARD CLAMP: Guarantee 0.0–1.0 output ──
        # This is the absolute last line of defense.
        normalized_reward = float(max(0.0, min(1.0, step_reward)))

        self.total_reward += normalized_reward
        self._episode_rewards.append(normalized_reward)
        self.steps += 1
        done = self.steps >= self.max_steps or (len(self.inbox) == 0 and self.steps > 1)

        # ── Completion Bonus ──
        if done and len(self.inbox) == 0:
            bonus = 0.5 # Normalized bonus
            self.total_reward += bonus
            normalized_reward = min(1.0, normalized_reward + 0.1) # Small step nudge

        # Apply rubric if present
        if hasattr(self, "_apply_rubric"):
            rubric_reward = self._apply_rubric(
                action, None
            )  # We don't have observation yet
            if rubric_reward != 0.0:
                normalized_reward = max(
                    0.0, min(1.0, normalized_reward + rubric_reward)
                )

        obs = self._create_observation(
            reward=normalized_reward,
            done=done,
            error_trace=step_error_trace,
            step_feedback=current_step_feedback,
        )
        
        # Record to trajectory for standardized grading
        self.trajectory.append({
            "action": action,
            "observation": obs,
            "reward": normalized_reward,
            "done": done
        })
        
        return obs

    def grader(self) -> float:
        """
        Min-Max normalized grader per OpenEnv Phase 2 specification.
        Uses task-specific theoretical max to produce a [0.0, 1.0] score.
        An optimal agent gets 1.0. A random agent gets ~0.2.
        """
        # Task-specific theoretical max rewards
        TASK_BASELINES = {
            "task1": {"min": 0.0, "max": 1.5},   # 1 email: ~0.9 + bonus
            "task2": {"min": 0.0, "max": 4.0},   # 3 emails: ~0.8*3 + progress + bonus
            "task3": {"min": 0.0, "max": 8.0},   # 5 emails: ~0.8*5 + progress + bonus
        }
        
        limits = TASK_BASELINES.get(self.current_task, {"min": 0.0, "max": 3.0})
        
        if limits["max"] == limits["min"]:
            return 0.5
        
        score = (self.total_reward - limits["min"]) / (limits["max"] - limits["min"])
        return float(max(0.0, min(1.0, score)))

    def state(self) -> State:
        """Return the current environment state."""
        if not hasattr(self, "episode_id") or self.episode_id is None:
            import uuid

            self.episode_id = str(uuid.uuid4())
        return State(
            episode_id=self.episode_id,
            step_count=self.steps,
        )

    def _create_observation(
        self,
        reward: float = 0.0,
        done: bool = False,
        error_trace: Optional[str] = None,
        step_feedback: Optional[str] = None,
    ) -> Observation:
        """Create an Observation object from current state."""
        info = {
            "total": round(self.total_reward, 2),
            "classifier_stats": self.classifier.stats,
            "episode_id": getattr(self, "episode_id", "N/A")
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
            error_trace=error_trace,
            step_feedback=step_feedback,
            html_observation=self._generate_html_table()
        )

    def _generate_html_table(self) -> str:
        """Generate a visual HTML representation of the current inbox."""
        if not self.inbox:
            return "<p style='font-family: sans-serif; color: #666;'>Your inbox is currently empty.</p>"

        html = "<table border='1' style='border-collapse: collapse; width: 100%; font-family: sans-serif; border: 1px solid #ddd;'>"
        html += "<tr style='background-color: #f8f9fa; color: #333;'><th>ID</th><th>Sender</th><th>Subject</th><th>Urgency</th><th>Sentiment</th><th>Spam Prob</th></tr>"
        for email in self.inbox:
            urgency = email.get('urgency', 'LOW')
            row_bg = "#ffffff"
            if urgency == "HIGH":
                row_bg = "#fff5f5"
            elif email.get("type") == "SPAM":
                row_bg = "#fcfcfc"

            html += f"<tr style='background-color: {row_bg};'>"
            html += f"<td style='padding: 8px;'>{email['id']}</td>"
            html += f"<td style='padding: 8px;'>{email['sender']}</td>"
            html += f"<td style='padding: 8px;'>{email['subject']}</td>"
            html += f"<td style='padding: 8px;'>{urgency}</td>"
            html += f"<td style='padding: 8px;'>{email.get('sentiment', 'Professional')}</td>"
            html += f"<td style='padding: 8px;'>{email.get('spam_score', 0.0):.2f}</td>"
            html += "</tr>"
        html += "</table>"
        return html
