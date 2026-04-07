"""
TriageAI FastAPI Backend v3.0
- ML-powered email classification
- Live IMAP inbox with intelligent categorization
- RL environment endpoints
- Classifier stats & performance tracking
"""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from .environment import EmailEnv
    from .models import Action
    from .imap_client import fetch_live_emails
    from .classifier import EmailClassifier, analyze_sentiment, detect_urgency
except ImportError:
    from environment import EmailEnv
    from models import Action
    from imap_client import fetch_live_emails
    from classifier import EmailClassifier, analyze_sentiment, detect_urgency
import os
from dotenv import load_dotenv

from pydantic import BaseModel
from typing import Dict, Optional

# Load env vars early
load_dotenv()

app = FastAPI(title="TriageAI - Email RL Environment", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RL Environment ---
env = EmailEnv()

# --- Standalone classifier for live email classification ---
_dataset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "dataset",
    "spam_assassin.csv",
).replace("\\", "/")  # Normalize path for Windows
live_classifier = EmailClassifier(dataset_path=_dataset_path)

# --- In-memory account store ---
ACCOUNTS: Dict[str, str] = {}

if os.getenv("EMAIL_1_USER") and os.getenv("EMAIL_1_PASS"):
    ACCOUNTS[os.getenv("EMAIL_1_USER")] = os.getenv("EMAIL_1_PASS").replace(" ", "")
if os.getenv("EMAIL_2_USER") and os.getenv("EMAIL_2_PASS"):
    ACCOUNTS[os.getenv("EMAIL_2_USER")] = os.getenv("EMAIL_2_PASS").replace(" ", "")


class AccountRequest(BaseModel):
    username: str
    password: str


class ClassifyRequest(BaseModel):
    text: str
    subject: Optional[str] = ""


class FeedbackRequest(BaseModel):
    predicted_spam: bool
    actual_spam: bool


# --- Routes ---


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.post("/accounts")
def add_account(req: AccountRequest):
    ACCOUNTS[req.username] = req.password.replace(" ", "")
    return {"status": "added", "count": len(ACCOUNTS)}


@app.get("/accounts")
def get_accounts():
    return {"accounts": list(ACCOUNTS.keys())}


@app.get("/live-inbox")
def live_inbox():
    """Fetch live emails from all accounts with ML classification."""
    all_emails = []

    for user, pwd in ACCOUNTS.items():
        if pwd == "your_app_password_here":
            continue
        try:
            emails = fetch_live_emails(user, pwd)
            # Classify each email with ML
            for email in emails:
                result = live_classifier.classify(
                    text=email.get("subject", ""), subject=email.get("subject", "")
                )
                email["type"] = result["type"]
                email["urgency"] = result["urgency"]
                email["sentiment"] = result["sentiment"]
                email["sentiment_confidence"] = result["sentiment_confidence"]
                email["spam_score"] = result["spam_score"]
                email["confidence"] = result["confidence"]
            all_emails.extend(emails)
        except Exception as e:
            print(f"[API] Error fetching {user}: {e}")

    all_emails.sort(key=lambda x: x.get("createdAt", 0), reverse=True)
    return {"emails": all_emails}


@app.post("/classify")
def classify_email(req: ClassifyRequest):
    """Classify a single email text using the ML pipeline."""
    result = live_classifier.classify(req.text, req.subject)
    return result


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """Submit classification feedback for online accuracy tracking."""
    live_classifier.update_feedback(req.predicted_spam, req.actual_spam)
    return {"status": "recorded", "stats": live_classifier.stats}


@app.get("/classifier-stats")
def classifier_stats():
    """Get classifier performance metrics."""
    return {
        "live_classifier": live_classifier.stats,
        "env_classifier": env.classifier.stats,
    }


@app.post("/reset")
def reset(task: str = "train"):
    return env.reset(task=task)


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action.model_dump())
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
def state():
    return env.state()


@app.get("/grader")
def grader():
    """Return the normalized grader score (0.0-1.0) for the current episode."""
    return {"score": env.grader(), "total_reward": env.total_reward, "task": env.current_task}


@app.get("/tasks")
def tasks():
    """Enumerate all available tasks with difficulty metadata."""
    return {
        "tasks": [
            {"id": "task1", "name": "Single Email Triage", "difficulty": "easy", "max_steps": 5},
            {"id": "task2", "name": "Backlog Processing", "difficulty": "medium", "max_steps": 10},
            {"id": "task3", "name": "Dynamic Inbox Stream", "difficulty": "hard", "max_steps": 20},
        ]
    }

