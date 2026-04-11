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

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Optional
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _clamp_score(v: float) -> float:
    """Clamp to strict (0.0, 1.0) range per OpenEnv Phase 2 requirements."""
    eps = 1e-4
    return max(eps, min(1.0 - eps, v))


try:
    from server.environment import EmailEnv
    from server.models import Action, StepResult, Observation, State
    from server.classifier import EmailClassifier
    from server.imap_client import validate_credentials, fetch_live_emails
    from server.registry import TASK_REGISTRY, Task
except ImportError:
    from .environment import EmailEnv
    from .models import Action, StepResult, Observation, State
    from .classifier import EmailClassifier
    from .imap_client import validate_credentials, fetch_live_emails
    from .registry import TASK_REGISTRY, Task

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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    Gracefully handles action schema hallucinations by the OpenEnv agents.
    If the agent submits a malformed Action, a 0.0 reward step is recorded.
    """
    error_msg = f"Action Schema Error: {str(exc)}"

    # Register a faulty step with 0.0 reward (failure baseline)
    env.steps += 1
    env.total_reward += 0.0
    env._episode_rewards.append(0.0)

    # Wrap in the mandatory StepResult format
    response_payload = {
        "observation": {
            "inbox": [e for e in env.inbox] if hasattr(env, "inbox") else [],
            "steps": env.steps,
            "task": getattr(env, "current_task", "train"),
            "max_steps": getattr(env, "max_steps", 10),
            "total_reward": round(env.total_reward, 2),
            "reward": 0.0,
            "done": env.steps >= env.max_steps,
            "error_trace": error_msg,
            "step_feedback": "Invalid action parameters provided. Learn the schema.",
        },
        "reward": 0.0,
        "done": env.steps >= env.max_steps,
        "info": {},
    }

    return JSONResponse(status_code=200, content=response_payload)


# ── Standalone classifier for classification endpoint ──
_dataset_path = None
for candidate in [
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
        "spam_assassin.csv",
    ),
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
frontend_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "frontend", "dist"
)

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
        root_icon = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "InboxIQ.png"
        )
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


@app.post("/reset", response_model=Observation)
def reset(
    task: str = "task1", seed: Optional[int] = None, episode_id: Optional[str] = None
):
    """Reset the environment and return the initial observation."""
    return env.reset(task=task, seed=seed, episode_id=episode_id)


@app.post("/step")
def step(action: Action):
    """Execute one action and return (observation, reward, done, info)."""
    obs = env.step(action)
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
        "reward": _clamp_score(obs.reward if hasattr(obs, "reward") else 0.0),
        "done": obs.done if hasattr(obs, "done") else False,
        "info": obs.info if hasattr(obs, "info") else {},
    }


@app.get("/state")
def state():
    """Return the current environment state."""
    return env.state()


@app.get("/metadata")
def metadata():
    """Return OpenEnv metadata for runtime validators and clients."""
    return {
        "name": "InboxIQ",
        "description": "OpenEnv-compliant reinforcement learning environment for intelligent email inbox triage",
        "version": "3.0",
        "tasks": [
            {
                "id": task.task_id,
                "name": task.name,
                "difficulty": task.difficulty,
                "max_steps": task.max_steps,
            }
            for task in TASK_REGISTRY.values()
        ],
    }


@app.get("/tasks")
def tasks():
    """Get metadata for all available tasks."""
    return list(TASK_REGISTRY.values())


@app.get("/schema")
def schema():
    """Return action/observation/state JSON schemas."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema(),
    }


@app.post("/mcp")
def mcp(payload: Dict = Body(default_factory=dict)):
    """
    Lightweight MCP compatibility endpoint.
    This keeps runtime validators and MCP-aware clients from hard failing.
    """
    req_id = payload.get("id")
    method = payload.get("method")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "InboxIQ", "version": "3.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": []}}

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": "Method not found"},
    }


@app.get("/health")
def health():
    """Health check endpoint for Docker and monitoring."""
    return {"status": "healthy", "environment": "InboxIQ"}


@app.get("/grader")
def grader():
    """Return the normalized grader score for the current episode."""
    return {
        "score": _clamp_score(env.grader()),
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
        return {
            "status": "connected",
            "username": req.username,
            "is_demo": True,
            "notice": "Fell back to simulation mode",
        }

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
                {
                    "sender": "hr@corporation.com",
                    "subject": "Action Required: Complete your mandatory compliance training",
                },
                {
                    "sender": "alerts@aws.amazon.com",
                    "subject": "AWS Budget Alert: Monthly threshold exceeded",
                },
                {
                    "sender": "newsletter@marketing.io",
                    "subject": "10 best practices for maximizing your workflow",
                },
                {
                    "sender": "boss@company.tech",
                    "subject": "Can we sync up later today? Need your input on the Q3 roadmap",
                },
                {
                    "sender": "noreply@github.com",
                    "subject": "[Repo/Core] Urgent Security Vulnerability Detected (Dependabot)",
                },
                {
                    "sender": "promo@deal-hub.com",
                    "subject": "URGENT: Your free claim prize inside! Click now",
                },
                {
                    "sender": "client.contact@external.io",
                    "subject": "Checking in: Revisions to the updated project proposal?",
                },
                {
                    "sender": "support@cloud-platform.io",
                    "subject": "CRITICAL: Service degradation report — ticket #4512",
                },
                {
                    "sender": "lunch-club@office.net",
                    "subject": "Pizza in the breakroom at 12!",
                },
                {
                    "sender": "scam.alert@baddomain.biz",
                    "subject": "Your Bank Account is Frozen - Reset Password Now",
                },
            ]
            import random
            import uuid

            # Slowing down the simulation rate to feel like normal realistic mail volume
            selected = random.sample(pool, 1)
            raw_emails = []
            for idx, e in enumerate(selected):
                offset = random.randint(10000, 1500000)
                raw_emails.append(
                    {
                        "id": f"demo-{uuid.uuid4().hex[:6]}",
                        "sender": e["sender"],
                        "subject": e["subject"],
                        "createdAt": int(time.time() * 1000) - offset,
                    }
                )
        else:
            raw_emails = fetch_live_emails(user, pwd)

        for email_data in raw_emails:
            # Classify using the ML pipeline
            classification = live_classifier.classify(email_data["subject"])
            email_data.update(
                {
                    "type": classification["type"],
                    "urgency": classification["urgency"],
                    "sentiment": classification["sentiment"],
                    "confidence": classification["confidence"],
                    "spam_score": classification["spam_score"],
                }
            )
            all_emails.append(email_data)

    return {"emails": all_emails}


def main():
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)


# ── Direct execution ──
if __name__ == "__main__":
    main()
