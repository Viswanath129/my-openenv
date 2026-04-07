---
title: TriageAI - Email RL Environment
emoji: 📧
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# TriageAI - Reinforcement Learning Email Triage System

<p align="center">
  <img src="https://img.shields.io/badge/RL-v3.0-blue?style=flat-square" alt="RL Version">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-19-orange?style=flat-square" alt="React">
</p>

## 🎯 What is TriageAI?

TriageAI frames **email triage as a reinforcement learning problem** where an agent must make sequential decisions under time and capacity constraints to maximize long-term efficiency.

> **Not classification. Decision-making over time.**

---

## 🧠 Why RL for Email Triage?

Traditional email systems use static rule-based filters. They answer:
> *"What is this email?"*

TriageAI answers:
> *"What action should I take NOW to get the best long-term outcome?"*

### The Core Problem

Information overload is real. The challenge is not classifying emails, but **optimizing attention under constraints**:

- Multiple emails arrive simultaneously
- Limited time to respond
- Actions have consequences that affect future state
- Some rewards are delayed (deferring urgent = looks fine now, penalty later)

---

## 🧩 RL Components

### 1. State (Observation)
```python
{
  "inbox": [
    {"type": "WORK", "urgency": "HIGH", "sentiment": "Aggressive", "wait": 2},
    {"type": "SPAM", "urgency": "LOW", "sentiment": "Casual", "wait": 1}
  ],
  "steps": 5,
  "task": "task3"
}
```
- Current inbox contents with metadata
- Wait time (how long each email has been pending)
- Urgency distribution
- Episode progress

### 2. Actions
```python
["open", "delete", "defer", "escalate"]
```
Applied to a specific email by ID.

### 3. Reward Function
| Action | Outcome | Reward |
|--------|---------|--------|
| Delete spam | Correct | +2.0 × confidence |
| Open spam | Missed spam | -2.5 × confidence |
| Open urgent | Good | +1.0 |
| Delete work | Mistake | -3.0 |
| Escalate aggressive | Smart | +1.5 |
| Defer urgent | Bad | -1.0 |
| Per-step wait penalty | -0.1 × wait_time |

### 4. Transition (State Changes)
After each action:
- Email removed or updated
- All wait counters increment
- 30% chance of new email arrival
- Backlog grows if unchecked

### 5. Episode
- **Task 1**: Single email triage
- **Task 2**: Static backlog (3 emails)
- **Task 3**: Dynamic stream (continuous arrivals)
- Max 20 steps per episode

---

## 🔥 Key RL Concepts Modeled

### Delayed Reward
```
defer urgent email → looks fine NOW
→ after 3 steps → -1.0 penalty
→ agent learns to think ahead
```

### Trade-offs
- **Speed vs Correctness**: Fast decisions may be wrong
- **Urgent vs Backlog**: Handle urgent or clear inbox?
- **Immediate vs Future**: Delete spam now or risk later?

### Attention as Resource
Think of it like a strategy game where:
- **Emails** = Tasks/Enemies
- **Time** = Pressure
- **Actions** = Moves
- **Reward** = Score

---

## 🤖 ML-Enhanced Intelligence

TriageAI includes an integrated ML pipeline:

- **Spam Classification**: TF-IDF + Naive Bayes trained on SpamAssassin
- **Sentiment Analysis**: Aggressive/Professional/Casual detection
- **Urgency Detection**: HIGH/MEDIUM/LLOW from content signals
- **Confidence Weighting**: Higher confidence = stronger reward signal

### Online Learning
The classifier tracks accuracy from user feedback:
```python
POST /feedback {predicted_spam: bool, actual_spam: bool}
```

---

## 🚀 Quick Start

### Backend (FastAPI + RL Environment)
```bash
cd email_env
pip install -r ../requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### Frontend (React Dashboard)
```bash
cd frontend
npm install
npm run dev
```

### Test RL Agent
```bash
python email_env/inference.py
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment (task1/task2/task3) |
| `/step` | POST | Execute action, get reward |
| `/state` | GET | Current inbox state |
| `/live-inbox` | GET | Fetch from Gmail with ML classification |
| `/classify` | POST | Classify single email text |
| `/feedback` | POST | Submit classification feedback |
| `/classifier-stats` | GET | ML model performance metrics |

---

## 📁 Project Structure

```
├── email_env/
│   ├── src/
│   │   ├── app.py          # FastAPI server
│   │   ├── environment.py # RL environment
│   │   ├── classifier.py  # ML pipeline
│   │   ├── imap_client.py # Gmail IMAP
│   │   └── models.py      # Pydantic schemas
│   ├── inference.py       # Baseline agent
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # React dashboard
│   │   └── index.css      # Tailwind
│   └── vite.config.js
└── requirements.txt       # Python deps
```

---

## 🐳 Docker Deployment

```bash
# Build and run
docker build -t triageai .
docker run -p 8000:8000 triageai
```

---

## 📊 Example RL Agent Output

```
[START] task=task3 env=email_env model=smart_agent_v3
[STEP] step=0 action=delete_DB-4821 reward=2.45 done=false error=null
[STEP] step=1 action=escalate_DB-3291 reward=2.95 done=false error=null
[STEP] step=2 action=open_DB-7102 reward=0.85 done=false error=null
[END] success=true steps=3 score=6.25 rewards=2.45,2.95,0.85
```

---

## 🔗 Integrations

TriageAI is compatible with:
- **OpenEnv**: Standardized RL environment interface
- **LLM Agents**: Can connect via `/step` and `/state` endpoints
- **Q-Learning/PPO**: Plug in any RL algorithm via the environment API
- **HuggingFace Spaces**: Ready for cloud deployment

---

## 📄 License

MIT License
