from pydantic import BaseModel
from typing import Literal, List, Dict, Any, Optional


class Action(BaseModel):
    action_type: Literal["open", "delete", "defer", "escalate"]
    email_id: str


class Observation(BaseModel):
    inbox: List[Dict[str, Any]]
    steps: int = 0
    task: str = "train"
    classifier_stats: Optional[Dict[str, Any]] = None
