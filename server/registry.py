from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

def calculate_success(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0
        
    last_step = trajectory[-1]
    observation = last_step.get("observation", {})
    if not isinstance(observation, dict):
        observation = observation.dict() if hasattr(observation, "dict") else getattr(observation, "__dict__", {})
        
    total_reward = observation.get("total_reward", 0.0)
    task_id = observation.get("task", "task1")
    
    # Normalizing bounds mapping to avoid gradient explosion issue
    task_benchmarks = {
        "task1": 1.35,  
        "task2": 3.05,  
        "task3": 7.00,  
    }
    
    r_max = task_benchmarks.get(task_id, 1.0)
    if r_max <= 0.0:
        return 0.0
        
    return float(total_reward / r_max)

class Task(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    prompt: str
    max_steps: int = 10
    config: Dict[str, Any] = Field(default_factory=dict)

    def grade_task(self, trajectory) -> float:
        # Must return a float between 0.0 and 1.0
        score = calculate_success(trajectory)
        return min(max(score, 0.0), 1.0)

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
