---
title: TriageAI - Email RL Environment
emoji: 📧
colorFrom: indigo
colorTo: violet
sdk: docker
app_port: 8000
pinned: false
---

# TriageAI - Email RL Environment

**Version 3.0** - ML-enhanced Reinforcement Learning Email Triage System

---

## 🎯 The Problem: Attention Under Constraints

Information overload is real. Traditional email systems use static filters that answer:
> *"What is this email?"*

TriageAI answers a different question:
> *"What action should I take NOW to get the best long-term outcome?"*

---

## 🧠 Why This Is RL (Not Classification)

| Classification Approach | RL Approach |
|------------------------|-------------|
| Input → Email | Input → State (full inbox) |
| Output → Label | Output → Action (open/delete/defer/escalate) |
| One-step decision | Sequential decisions with consequences |
| No memory | Wait time accumulates |
| No trade-offs | Must balance urgent vs backlog |

### The Core RL Idea

At every step, the agent must think ahead:
- Deferring an urgent email **looks fine now** but incurs **delayed penalty**
- Deleting spam is correct, but only if you **caught it**
- Every action changes the future state

---

## 🧩 RL Components in TriageAI

### 1. State (Observation Space)
```python
{
  "inbox": [
    {
      "id": "DB-4821",
      "type": "WORK",           # SPAM, WORK, SUPPORT
      "urgency": "HIGH",         # HIGH, MEDIUM, LOW
      "sentiment": "Aggressive", # Aggressive, Professional, Casual
      "wait": 2,                # Steps pending
      "spam_score": 0.85,       # ML confidence
      "confidence": 0.92        # Overall confidence
    }
  ],
  "steps": 5,
  "task": "task3",
  "classifier_stats": {...}
}
```

### 2. Action Space
```python
class Action(BaseModel):
    action_type: Literal["open", "delete", "defer", "escalate"]
    email_id: str
```

### 3. Reward Function (Complex Grader)
```python
def complex_grader(email, action_type):
    # Base reward from correctness
    if is_spam:
        reward += 2.0 if action == "delete" else -2.5
    else:
        reward += 1.0 if action in ["open", "escalate"] else -3.0
    
    # Sentiment bonus
    if email.sentiment == "Aggressive" and action == "escalate":
        reward += 1.5
    
    # Urgency penalty
    if email.urgency == "HIGH" and action == "defer":
        reward -= 1.0
    
    # Wait penalty (capped)
    reward -= min(0.1 * email.wait, 2.0)
    
    return reward
```

### 4. Transition Dynamics
After each action:
- Selected email: removed (open/delete) or stays (defer)
- All other emails: `wait += 1`
- 30% chance: new email arrives
- Episode ends: `steps >= 20`

### 5. Episode (Task Definitions)
- **Task 1**: Single email triage (baseline)
- **Task 2**: Static backlog (3 emails)
- **Task 3**: Dynamic stream with continuous arrivals

---

## 🤖 ML Intelligence (v3.0)

TriageAI includes an integrated ML pipeline that enhances the RL environment:

### Naive Bayes Classifier
- Trained on SpamAssassin dataset (500 samples)
- TF-IDF vectorization with 2500 features
- Laplace smoothing
- Returns spam/ham with confidence

### Sentiment Analysis
- **Aggressive**: urgent, immediately, threat, complaint, lawsuit...
- **Professional**: regards, meeting, proposal, invoice...
- **Casual**: hey, thanks, cool, awesome...

### Urgency Detection
- **HIGH**: "urgent", "asap", "deadline", "action required"
- **LOW**: "newsletter", "unsubscribe", "promotion"

### Confidence-Weighted Rewards
```python
confidence_multiplier = 0.5 + confidence  # Range: 0.5 - 1.5
reward = base_reward * confidence_multiplier
```

---

## 💰 Detailed Reward Matrix

| Scenario | Action | Reward |
|----------|--------|--------|
| Delete spam (high conf) | delete | +2.0 × conf_mult |
| Delete spam (low conf) | delete | +1.0 × conf_mult |
| Open spam | open | -2.5 × conf_mult |
| Open work | open | +1.0 × conf_mult |
| Escalate work | escalate | +1.0 × conf_mult |
| Delete work | delete | -3.0 (always bad) |
| Escalate aggressive | escalate | +1.5 bonus |
| Ignore aggressive | defer | -1.0 penalty |
| Defer urgent | defer | -1.0 penalty |
| Open urgent | open | +0.5 bonus |
| Per-step wait | any | -0.1 × wait (capped) |
| Step cost | any | -0.05 |

---

## 📡 API Reference

### RL Environment Endpoints

```bash
# Reset environment for a task
POST /reset?task=task3
# Returns: {"inbox": [...], "steps": 0, "task": "task3"}

# Execute action
POST /step
{
  "action_type": "delete",
  "email_id": "DB-4821"
}
# Returns: {"observation": {...}, "reward": 2.45, "done": false, "info": {...}}

# Get current state
GET /state
# Returns: {"inbox": [...], "steps": 5, "task": "task3"}
```

### ML Integration Endpoints

```bash
# Classify email text
POST /classify
{
  "text": "Your invoice is overdue...",
  "subject": "Payment Required"
}
# Returns: {"type": "SUPPORT", "urgency": "HIGH", "sentiment": "Aggressive", ...}

# Submit feedback for online learning
POST /feedback
{
  "predicted_spam": true,
  "actual_spam": true
}
# Returns: {"status": "recorded", "stats": {...}}

# Get classifier performance
GET /classifier-stats
# Returns: {"live_classifier": {"accuracy": 0.92, "total_classified": 150, ...}}
```

### Live Gmail Integration

```bash
# Fetch and classify live emails
GET /live-inbox
# Returns: {"emails": [{..., "type": "SPAM", "urgency": "HIGH", ...}]}
```

---

## 🧪 Testing

### Run Inference Agent
```bash
python inference.py
```

Output:
```
[START] task=task3 env=email_env model=smart_agent_v3
[STEP] step=0 action=delete_DB-4821 reward=2.45 done=false error=null
[STEP] step=1 action=escalate_DB-3291 reward=2.95 done=false error=null
[END] success=true steps=2 score=5.40 rewards=2.45,2.95
```

### Test via OpenEnv
```bash
openenv run email_env --task task3 --episodes 10
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TriageAI System                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Frontend  │◄──►│  FastAPI    │◄──►│    RL       │     │
│  │  (React)   │    │  Server     │    │Environment  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                           │                  │             │
│                    ┌──────▼──────┐    ┌──────▼──────┐      │
│                    │  Gmail      │    │   ML        │      │
│                    │  IMAP       │    │Classifier   │      │
│                    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backend
cd email_env
uvicorn src.app:app --host 0.0.0.0 --port 8000

# Run frontend (new terminal)
cd frontend
npm install
npm run dev

# Test inference
python inference.py
```

---

## 🔗 Compatible With

- **OpenEnv**: Standardized RL environment interface
- **HuggingFace Spaces**: Cloud deployment ready
- **LLM Agents**: Connect via REST API
- **Q-Learning/PPO**: Plug any RL algorithm

---

## 📊 Performance

- **Dataset**: SpamAssassin (500 samples, 80/20 split)
- **ML Accuracy**: ~92% on test set
- **Episode Length**: Max 20 steps
- **Action Space**: 4 discrete actions
- **State Space**: Variable (inbox size up to 8)

---

## 📄 License

MIT
