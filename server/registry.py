from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Task(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    prompt: str
    max_steps: int = 10
    config: Dict[str, Any] = Field(default_factory=dict)

    def grade_task(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Standard OpenEnv grader interface.
        Evaluates the agent's performance across the entire trajectory.
        Must return a float between 0.0 and 1.0.
        """
        if not trajectory:
            return 0.0

        # Extract context from the last step (contains environment-calculated rewards)
        last_step = trajectory[-1]
        observation = last_step.get("observation", {})
        info = observation.get("info", {}) if isinstance(observation, dict) else getattr(observation, "info", {})
        
        # Calculate success based on total reward and progress
        # We use a weighted blend of normalized reward and completion efficiency
        total_reward = observation.get("total_reward", 0.0) if isinstance(observation, dict) else getattr(observation, "total_reward", 0.0)
        steps = len(trajectory)
        
        # Calculate success based on mean normalized reward across steps
        if steps > 0:
            score = total_reward / steps
        else:
            score = 0.0

        # Add efficiency bonus (fewer steps -> higher score)
        efficiency = max(0, 1.0 - (steps / self.max_steps))
        score = (score * 0.8) + (efficiency * 0.2)

        # Strictly clamp to 0.0 - 1.0 range per Phase 2 requirements
        return float(min(max(score, 0.0), 1.0))

TASK_REGISTRY = {
    "task1": Task(
        task_id="task1",
        name="Precise Triage",
        description="Classify and process a single high-priority email correctly.",
        difficulty="easy",
        prompt="Your inbox contains one email. Analyze its sentiment, urgency, and spam score. Execute the most appropriate action (open, delete, or escalate) to maximize reward.",
        max_steps=5,
        config={"count": 1}
    ),
    "task2": Task(
        task_id="task2",
        name="Incentive Cleanup",
        description="Clear a mixed backlog of 3 emails with varying priorities.",
        difficulty="medium",
        prompt="You have a backlog of 3 emails. Some are spam, some are urgent work requests. Process all of them efficiently. Remember: deleting a legitimate email is heavily penalized.",
        max_steps=10,
        config={"count": 3}
    ),
    "task3": Task(
        task_id="task3",
        name="Chaos Management",
        description="Handle a stream of 5 emails with dynamic arrivals and high-pressure sentiment.",
        difficulty="hard",
        prompt="Total Chaos! 5 emails are pending, and more may arrive. Prioritize 'Aggressive' sentiment for escalation and 'HIGH' urgency for immediate opening. Efficiency and accuracy are critical.",
        max_steps=20,
        config={"count": 5}
    )
}
