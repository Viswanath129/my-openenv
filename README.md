---
title: InboxIQ
emoji: 📧
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - email-triage
---

<p align="center">
  <img src="InboxIQ.png" alt="InboxIQ Logo" width="200"/>
</p>

<h1 align="center">InboxIQ</h1>

<p align="center">
  <b>Reinforcement Learning Environment for Intelligent Email Inbox Triage</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/OpenEnv-Compliant-brightgreen?style=for-the-badge" alt="OpenEnv Compliant">
  <img src="https://img.shields.io/badge/PyTorch-Hackathon-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Hackathon">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/HuggingFace-Spaces-yellow?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Spaces">
</p>

---

## 🎯 Overview & Motivation

**InboxIQ** frames **email triage as a real-world sequential decision-making problem** — a task humans perform daily under time pressure, cognitive overload, and uncertainty.

Unlike toy RL environments, InboxIQ simulates the actual challenge of managing an inbox:
- Emails arrive dynamically with varying urgency, sentiment, and spam probability
- Every action (open, delete, defer, escalate) has downstream consequences
- The agent must balance throughput, accuracy, and response time

> **Why RL?** Email triage isn't just classification — it's an optimization problem over time. Deferring an urgent email saves attention *now* but costs more *later*. Deleting real mail is catastrophic. InboxIQ captures these sequential trade-offs.

---

## 🧩 Environment Design

### Observation Space

Each observation contains the current inbox state:

| Field | Type | Description |
|:------|:-----|:------------|
| `inbox` | `List[EmailItem]` | List of email objects in the current inbox |
| `steps` | `int` | Current step number in the episode |
| `task` | `str` | Active task identifier |
| `max_steps` | `int` | Maximum steps allowed for the task |
| `total_reward` | `float` | Cumulative reward so far |

**EmailItem** fields:
| Field | Type | Description |
|:------|:-----|:------------|
| `id` | `str` | Unique email identifier |
| `sender` | `str` | Email sender address |
| `subject` | `str` | Email subject line |
| `type` | `str` | Classification: `SPAM`, `WORK`, `SUPPORT` |
| `urgency` | `str` | Urgency level: `LOW`, `MEDIUM`, `HIGH` |
| `sentiment` | `str` | Tone: `Aggressive`, `Professional`, `Casual` |
| `sentiment_confidence` | `float` | Sentiment prediction confidence |
| `spam_score` | `float` | ML spam probability (0.0–1.0) |
| `confidence` | `float` | Overall classification confidence |
| `wait` | `int` | Steps this email has been waiting |

### Action Space

```json
{
  "action_type": "open | delete | defer | escalate",
  "email_id": "MSG-123"
}
```

| Action | Description | Best For |
|:-------|:-----------|:---------|
| `open` | Process/read the email | Legitimate work emails |
| `delete` | Permanently remove | Spam/junk emails |
| `defer` | Delay processing | Low-priority items (penalty applies) |
| `escalate` | Priority handling | Aggressive sentiment or HIGH urgency |

### Reward Function

The reward function provides **incremental feedback throughout the trajectory**, not just at episode completion:

| Outcome | Reward | Condition |
|:--------|:-------|:----------|
| ✅ Blocked Spam | `+2.0 × confidence` | Delete a spam email |
| ✅ Escalated Urgent | `+1.5` | Escalate aggressive/high-urgency |
| ✅ Processed Work | `+1.0 × confidence` | Open legitimate email |
| ⚠️ Deferred Urgent | `-1.0` | Defer a HIGH urgency email |
| ❌ Deleted Real Email | `-3.0` | Critical error — deleting legitimate mail |
| ❌ Allowed Spam | `-2.5 × confidence` | Failing to delete spam |
| 🕐 Wait Penalty | `-0.1 × wait` | Per-step delay penalty (capped at -2.0) |
| 🔄 Step Cost | `-0.05` | Small cost per step (prevents infinite loops) |

**Design rationale**: Confidence-weighted rewards incentivize the agent to leverage ML signals. The step cost prevents degenerate policies.

---

## 📋 Tasks (Benchmarks)

InboxIQ includes **three tasks** spanning increasing difficulty:

### Task 1: Single Email Triage (Easy)
- **Objective**: Correctly classify and act on a single email
- **Initial inbox**: 1 email
- **Max steps**: 5
- **Focus**: Basic action selection — distinguish spam from legitimate mail
- **Optimal reward**: ~3.0

### Task 2: Backlog Processing (Medium)
- **Objective**: Process a backlog of 3 emails with mixed types
- **Initial inbox**: 3 emails (spam, work, support mix)
- **Max steps**: 10
- **Focus**: Batch prioritization and decision-making under diversity
- **Optimal reward**: ~8.0

### Task 3: Dynamic Inbox Stream (Hard)
- **Objective**: Handle 5 initial emails with continuous random arrivals
- **Initial inbox**: 5 emails + dynamic arrivals (30% chance per step)
- **Max steps**: 20
- **Focus**: Real-time triage under time pressure and evolving state
- **Optimal reward**: ~15.0

### Grading

Each task has a **deterministic programmatic grader** that normalizes the total episode reward to a score between `0.0` and `1.0`:

```
score = clip((total_reward - worst) / (optimal - worst), 0.0, 1.0)
```

The grader is deterministic: identical action trajectories produce identical scores.

---

## 🤖 ML Pipeline

InboxIQ integrates a real-time ML classification engine built from scratch (no sklearn dependency):

- **Spam Filter**: TF-IDF + Multinomial Naive Bayes trained on SpamAssassin dataset
- **Sentiment Analysis**: Keyword-based detection for Aggressive/Professional/Casual tones
- **Urgency Detection**: Content signal analysis for HIGH/MEDIUM/LOW urgency
- **Online Learning**: Accuracy tracking from agent feedback

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend build)

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/Viswanath129/my-openenv.git
cd my-openenv

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the RL Environment (FastAPI server)
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000

# 4. Run Inference Benchmark
HF_TOKEN=hf_your_token python inference.py
```

### Docker

```bash
# Build the container
docker build -t inboxiq .

# Run the container
docker run -p 8000:8000 inboxiq
```

### Frontend (Optional)

```bash
cd frontend
npm install
npm run build
```

---

## 📡 OpenEnv Compliance

InboxIQ fully implements the OpenEnv specification:

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/reset?task={id}` | POST | Reset environment, returns initial observation |
| `/step` | POST | Execute action, returns `(observation, reward, done, info)` |
| `/state` | GET | Returns current environment state |
| `/grader` | GET | Returns normalized score `[0.0, 1.0]` |

### Typed Models (Pydantic)
- `Action`: Typed action model with `action_type` and `email_id`
- `Observation`: Typed observation with inbox state
- `StepResult`: Typed step response with observation, reward, done, info
- `GraderResult`: Typed grader score with constraints

### Inference Logging Format
```
[START] task=task1 env=InboxIQ model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=delete_DB-1234 reward=2.45 done=true error=null
[END] success=true steps=1 score=1.0000 rewards=2.45
```

---

## 📊 Baseline Performance

Baseline scores using **heuristic fallback policy** (no LLM):

| Task | Difficulty | Avg Score | Avg Steps | Strategy |
|:-----|:-----------|:----------|:----------|:---------|
| task1 | Easy | 0.75–0.90 | 1–2 | Spam → delete, else open |
| task2 | Medium | 0.60–0.80 | 3–5 | Priority-sorted processing |
| task3 | Hard | 0.45–0.65 | 10–15 | Reactive triage with pressure |

With **LLM agent** (Qwen2.5-72B via HuggingFace Router):

| Task | Difficulty | Avg Score | Improvement |
|:-----|:-----------|:----------|:------------|
| task1 | Easy | 0.85–0.95 | +10–15% |
| task2 | Medium | 0.70–0.85 | +10–15% |
| task3 | Hard | 0.55–0.75 | +15–20% |

---

## 🏗️ Project Structure

```
InboxIQ/
├── src/
│   ├── __init__.py          # Package init
│   ├── app.py               # FastAPI server (OpenEnv endpoints)
│   ├── environment.py       # RL environment (reset, step, state, grader)
│   ├── models.py            # Pydantic models (Action, Observation, Reward)
│   └── classifier.py        # ML pipeline (TF-IDF + NB + sentiment + urgency)
├── frontend/                # React dashboard (Vite + Lucide)
├── dataset/                 # SpamAssassin training data
├── inference.py             # Baseline inference script (OpenAI client)
├── openenv.yaml             # OpenEnv metadata specification
├── Dockerfile               # Multi-stage Docker build
├── requirements.txt         # Python dependencies
├── InboxIQ.png              # Project logo
└── README.md                # This file
```

---

## 🏆 Meta OpenEnv Hackathon

This project was built for the **Meta PyTorch OpenEnv Hackathon 2026**, in collaboration with Meta, Hugging Face, PyTorch, and Scaler School of Technology.

**Team**: Viswanath129

---

## 📄 License

MIT License
