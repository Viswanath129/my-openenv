"""
InboxIQ — FastAPI Backend v3.0
OpenEnv-compliant REST API for the InboxIQ RL environment.

Endpoints:
  POST /reset?task={id}  → Initial observation
  POST /step             → (observation, reward, done, info)
  GET  /state            → Current state
  GET  /grader           → Normalized score [0.0, 1.0]
  POST /classify         → ML email classification
  GET  /classifier-stats → Classifier performance metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Optional

try:
    from server.environment import EmailEnv
    from server.models import Action
    from server.classifier import EmailClassifier
    from server.imap_client import validate_credentials, fetch_live_emails
except ImportError:
    from .environment import EmailEnv
    from .models import Action
    from .classifier import EmailClassifier
    from .imap_client import validate_credentials, fetch_live_emails

# Load env vars early
load_dotenv()

app = FastAPI(
    title="InboxIQ — Email Triage RL Environment",
    description="OpenEnv-compliant reinforcement learning environment for intelligent email inbox triage",
    version="3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── RL Environment ──
env = EmailEnv()

# ── Standalone classifier for classification endpoint ──
_dataset_path = None
for candidate in [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "spam_assassin.csv"),
    "/app/dataset/spam_assassin.csv",
    "dataset/spam_assassin.csv",
]:
    if os.path.exists(candidate):
        _dataset_path = candidate
        break

live_classifier = EmailClassifier(dataset_path=_dataset_path)


class ClassifyRequest(BaseModel):
    text: str
    subject: Optional[str] = ""


class FeedbackRequest(BaseModel):
    predicted_spam: bool
    actual_spam: bool


class AccountRequest(BaseModel):
    username: str
    password: str


# ── In-memory account store ──
connected_accounts: Dict[str, str] = {}


# ── Static Files & Frontend ──
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

if os.path.isdir(frontend_path):
    assets_dir = os.path.join(frontend_path, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/InboxIQ.png")
    def serve_icon():
        icon_path = os.path.join(frontend_path, "InboxIQ.png")
        if os.path.exists(icon_path):
            return FileResponse(icon_path, media_type="image/png")
        # Fallback to root-level copy
        root_icon = os.path.join(os.path.dirname(os.path.dirname(__file__)), "InboxIQ.png")
        if os.path.exists(root_icon):
            return FileResponse(root_icon, media_type="image/png")
        raise HTTPException(status_code=404, detail="Icon not found")

    @app.get("/favicon.svg")
    def serve_favicon():
        fav_path = os.path.join(frontend_path, "favicon.svg")
        if os.path.exists(fav_path):
            return FileResponse(fav_path, media_type="image/svg+xml")
        raise HTTPException(status_code=404, detail="Favicon not found")
else:
    @app.get("/")
    def root():
        return RedirectResponse(url="/docs")


# ══════════════════════════════════════════════════════════════════════════════
# OpenEnv Core Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/reset")
def reset(task: str = "train"):
    """Reset the environment and return the initial observation."""
    return env.reset(task=task)


@app.post("/step")
def step(action: Action):
    """Execute one action and return (observation, reward, done, info)."""
    obs, reward, done, info = env.step(action.model_dump())
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
def state():
    """Return the current environment state."""
    return env.state()


@app.get("/grader")
def grader():
    """Return the normalized grader score for the current episode."""
    return {
        "score": env.grader(),
        "total_reward": env.total_reward,
        "task": env.current_task,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ML Classification Endpoints
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Account & Live Email Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/accounts")
def list_accounts():
    """List all connected IMAP accounts."""
    return {"accounts": list(connected_accounts.keys())}


@app.post("/accounts")
def add_account(req: AccountRequest):
    """Validate and connect a new IMAP account."""
    # ── Demo Mode Bypass ──
    if req.username.lower() in ["demo", "demo@inbox-iq.ai"]:
        connected_accounts[req.username] = "demo-pass"
        return {"status": "connected", "username": req.username, "is_demo": True}

    success, msg = validate_credentials(req.username, req.password)
    # Fast bypass: If network is unreachable or auth fails, we drop into Demo Mode automatically!
    if not success:
        connected_accounts[req.username] = "demo-pass"
        return {"status": "connected", "username": req.username, "is_demo": True, "notice": "Fell back to simulation mode"}
    
    connected_accounts[req.username] = req.password
    return {"status": "connected", "username": req.username}


@app.get("/live-inbox")
def get_live_inbox():
    """Fetch recent emails from all accounts and classify them."""
    all_emails = []
    import time
    for user, pwd in connected_accounts.items():
        if pwd == "demo-pass":
            # ── Simulated 'Live' Inbox for Demo Mode (Randomized) ──
            pool = [
                {"sender": "hr@corporation.com", "subject": "Action Required: Complete your mandatory compliance training"},
                {"sender": "alerts@aws.amazon.com", "subject": "AWS Budget Alert: Monthly threshold exceeded"},
                {"sender": "newsletter@marketing.io", "subject": "10 best practices for maximizing your workflow"},
                {"sender": "boss@company.tech", "subject": "Can we sync up later today? Need your input on the Q3 roadmap"},
                {"sender": "noreply@github.com", "subject": "[Repo/Core] Urgent Security Vulnerability Detected (Dependabot)"},
                {"sender": "promo@deal-hub.com", "subject": "URGENT: Your free claim prize inside! Click now"},
                {"sender": "client.contact@external.io", "subject": "Checking in: Revisions to the updated project proposal?"},
                {"sender": "support@cloud-platform.io", "subject": "CRITICAL: Service degradation report — ticket #4512"},
                {"sender": "lunch-club@office.net", "subject": "Pizza in the breakroom at 12!"},
                {"sender": "scam.alert@baddomain.biz", "subject": "Your Bank Account is Frozen - Reset Password Now"}
            ]
            import random
            import uuid
            # Slowing down the simulation rate to feel like normal realistic mail volume
            selected = random.sample(pool, 1)
            raw_emails = []
            for idx, e in enumerate(selected):
                offset = random.randint(10000, 1500000)
                raw_emails.append({
                    "id": f"demo-{uuid.uuid4().hex[:6]}",
                    "sender": e["sender"],
                    "subject": e["subject"],
                    "createdAt": int(time.time() * 1000) - offset
                })
        else:
            raw_emails = fetch_live_emails(user, pwd)
        
        for email_data in raw_emails:
            # Classify using the ML pipeline
            classification = live_classifier.classify(email_data["subject"])
            email_data.update({
                "type": classification["type"],
                "urgency": classification["urgency"],
                "sentiment": classification["sentiment"],
                "confidence": classification["confidence"],
                "spam_score": classification["spam_score"],
            })
            all_emails.append(email_data)
    
    return {"emails": all_emails}


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)

# ── Direct execution ──
if __name__ == "__main__":
    main()
