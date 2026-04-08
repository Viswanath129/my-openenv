"""
InboxIQ — Pydantic models for the OpenEnv RL interface.
Typed Observation, Action, and Reward models per OpenEnv spec.
"""

from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional


class Action(BaseModel):
    """Agent action — choose a triage operation for a specific email."""
    action_type: Literal["open", "delete", "defer", "escalate"] = Field(
        description="The triage action: open (process), delete (remove), defer (delay), escalate (priority)"
    )
    email_id: str = Field(description="Unique identifier of the target email")


class EmailItem(BaseModel):
    """Single email in the inbox observation."""
    id: str
    sender: str = ""
    subject: str = ""
    type: str = "WORK"
    urgency: str = "MEDIUM"
    sentiment: str = "Professional"
    sentiment_confidence: float = 0.5
    spam_score: float = 0.0
    confidence: float = 0.5
    wait: int = 0


class Observation(BaseModel):
    """Environment observation returned after reset/step."""
    inbox: List[Dict[str, Any]] = Field(default_factory=list)
    steps: int = 0
    task: str = "train"
    max_steps: int = 10
    total_reward: float = 0.0
    classifier_stats: Optional[Dict[str, Any]] = None


class StepResult(BaseModel):
    """Result returned by the step endpoint."""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class GraderResult(BaseModel):
    """Grader score for the current episode."""
    score: float = Field(gt=0.0, lt=1.0)
    total_reward: float
    task: str
