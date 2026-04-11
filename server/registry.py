from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
SCORE_EPSILON = 1e-4


def calculate_success(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return SCORE_EPSILON

    last_step = trajectory[-1]
    observation = last_step.get("observation", {})
    if not isinstance(observation, dict):
        observation = (
            observation.dict()
            if hasattr(observation, "dict")
            else getattr(observation, "__dict__", {})
        )

    total_reward = observation.get("total_reward", 0.0)
    task_id = observation.get("task", "task1")

    task_cfg = TASK_REGISTRY.get(task_id)
    if task_cfg:
        if "reward_ceiling" in task_cfg.config:
            r_max = float(task_cfg.config["reward_ceiling"])
        else:
            count = max(1, task_cfg.config.get("count", 1))
            r_max = float(sum(0.9**i for i in range(count)))
    else:
        r_max = 1.0
    if r_max <= 0.0:
        return SCORE_EPSILON

    score = float(total_reward / r_max)
    score = min(max(score, 0.0), 1.0)
    if score <= 0.0:
        return SCORE_EPSILON
    if score >= 1.0:
        return 1.0 - SCORE_EPSILON
    return score


class Task(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    prompt: str
    max_steps: int = 10
    config: Dict[str, Any] = Field(default_factory=dict)

    def grade_task(self, trajectory) -> float:
        # Scaler validator requires strict open interval (0, 1).
        score = calculate_success(trajectory)
        if score <= 0.0:
            return SCORE_EPSILON
        if score >= 1.0:
            return 1.0 - SCORE_EPSILON
        return score


TASK_REGISTRY = {
    "task1": Task(
        task_id="task1",
        name="Precise Triage",
        description="Classify and process a single high-priority email correctly.",
        difficulty="easy",
        prompt="Your inbox contains one email. Analyze its sentiment, urgency, and spam score. Execute the most appropriate action (open, delete, or escalate) to maximize reward.",
        max_steps=5,
        config={"count": 1, "reward_ceiling": 1.0},
    ),
    "task2": Task(
        task_id="task2",
        name="Incentive Cleanup",
        description="Clear a mixed backlog of 3 emails with varying priorities.",
        difficulty="medium",
        prompt="You have a backlog of 3 emails. Some are spam, some are urgent work requests. Process all of them efficiently. Remember: deleting a legitimate email is heavily penalized.",
        max_steps=10,
        config={"count": 3, "reward_ceiling": 2.71},
    ),
    "task3": Task(
        task_id="task3",
        name="Chaos Management",
        description="Handle a stream of 5 emails with dynamic arrivals and high-pressure sentiment.",
        difficulty="hard",
        prompt="Total Chaos! 5 emails are pending, and more may arrive. Prioritize 'Aggressive' sentiment for escalation and 'HIGH' urgency for immediate opening. Efficiency and accuracy are critical.",
        max_steps=20,
        config={"count": 5, "reward_ceiling": 12.0},
    ),
}
