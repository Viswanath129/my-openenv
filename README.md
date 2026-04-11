---
title: InboxIQ
emoji: 📧
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
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
  <a href="https://huggingface.co/spaces/KasiViswanath/InboxIQ"><img src="https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Live Demo"></a>
</p>

<p align="center">
  <b>🔗 Live Demo: <a href="https://huggingface.co/spaces/KasiViswanath/InboxIQ">https://huggingface.co/spaces/KasiViswanath/InboxIQ</a></b>
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
| `html_observation` | `str` | **[NEW]** Visual HTML table of the inbox for multimodal agents |

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

### 🖼️ Multimodal Support

InboxIQ is future-proofed for **Vision-Language Models** (e.g., Llama-3-Vision, Qwen-2.5-VL). Every observation includes a semantically styled HTML representation of the inbox, allowing multimodal agents to "see" the priority levels and urgency through visual cues.

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
| ✅ Blocked Spam | `+0.6 – 0.9` | Delete a spam email (confidence weighted) |
| ✅ Escalated Urgent | `+0.7` | Escalate aggressive/high-urgency |
| ✅ Processed Work | `+0.5 – 0.8` | Open legitimate email (confidence weighted) |
| ⚠️ Deferred Urgent | `+0.2` | Defer a HIGH urgency email (penalty applies) |
| ❌ Deleted Real Email | `0.0` | Critical error — deleting legitimate mail |
| ❌ Allowed Spam | `0.1 – 0.3` | Failing to delete spam |
| 🕐 Wait Penalty | `-0.05` | Per-step delay penalty |
| 🏆 Completion Bonus | `+0.5` | Episode successfully cleared |

**Reward Normalization**: All per-step rewards and cumulative scores are strictly constrained to the **0.0 – 1.0 range** to prevent gradient explosion in RL algorithms. **Partial progress signals** are provided via `step_feedback` telemetry.

**Design rationale**: Confidence-weighted rewards incentivize the agent to leverage ML signals. The step cost prevents degenerate policies. **Partial progress signals** provide continuous feedback during the investigative loop, rewarding correct email processing and efficiency.

---

## 📋 Tasks (Benchmarks)

InboxIQ includes **three tasks** spanning increasing difficulty:

### Task 1: Precise Triage (Easy)
- **Objective**: Correctly classify and act on a single high-priority email.
- **Initial inbox**: 1 email
- **Max steps**: 5
- **Focus**: Foundational action selection and state manipulation.

### Task 2: Incentive Cleanup (Medium)
- **Objective**: Clear a mixed backlog of 3 emails with varying priorities.
- **Initial inbox**: 3 emails (spam, work, support mix)
- **Max steps**: 10
- **Focus**: Prioritization and sequential tool usage.

### Task 3: Chaos Management (Hard)
- **Objective**: Handle 5 initial emails with continuous random arrivals and aggressive sentiment.
- **Initial inbox**: 5 emails + dynamic arrivals.
- **Max steps**: 20
- **Focus**: Real-time triage under pressure and anomaly detection.

### Task Registry & Grading

InboxIQ utilizes a **Standardized Task Registry** (`server/registry.py`) providing a unified interface for evaluation:

```python
def grade_task(self, trajectory) -> float:
    # Deterministic grading based on episode history
    # Returns strictly between 0.0 and 1.0
    return calculate_success(trajectory)
```

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
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# 4. Run Inference Benchmark
HF_TOKEN=hf_your_token python inference.py
```

### Docker

```bash
# Build the container
docker build -t inboxiq .

# Run the container
docker run -p 7860:7860 inboxiq
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
- `Action`: Typed with `action_type` and `email_id`.
- `Observation`: Includes `inbox`, `reward`, `done`, `error_trace`, and `step_feedback`.
- `State`: Mandatory persistence with `episode_id` and `step_count`.
- `GraderResult`: Normalized score with strict constraints.

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
├── server/
│   ├── __init__.py          # Package init
│   ├── app.py               # FastAPI server (OpenEnv endpoints)
│   ├── environment.py       # RL environment (reset, step, state, grader)
│   ├── registry.py          # Standardized Task Registry & Graders
│   ├── models.py            # Pydantic models (Action, Observation, State)
│   ├── classifier.py        # ML pipeline (TF-IDF + NB + sentiment + urgency)
│   └── imap_client.py       # IMAP client for live email fetching
├── frontend/                # React dashboard (Vite + Lucide)
├── dataset/                 # SpamAssassin training data
├── inference.py             # Baseline inference script (OpenAI client)
├── openenv.yaml             # OpenEnv metadata specification
├── Dockerfile               # Multi-stage Docker build
├── pyproject.toml           # Python dependencies and build config
├── uv.lock                  # Dependency lock file
├── InboxIQ.png              # Project logo
└── README.md                # This file
```

---

## 🏆 Meta OpenEnv Hackathon

This project was built for the **Meta PyTorch OpenEnv Hackathon 2026**, in collaboration with Meta, Hugging Face, PyTorch, and Scaler School of Technology.

**Team**: Viswanath129

---

## 🧪 Testing & Evaluation

### 1. Manual UI Walkthrough (Demo Mode)
If your environment (like Hugging Face) restricts outbound IMAP port 993, use the following **Demo Credentials** to bypass the login and view the live triage dashboard:

- **Username**: `demo`
- **Password**: `any`

Once connected, the system will generate three diverse emails. You can manually **Open**, **Trash**, or **Escalate** them to see the real-time reward feedback and ML classification signals.

### 2. Automated Baseline (Inference)
The project includes a robust inference script that uses a deterministic heuristic policy to establish a performance baseline across all three tasks.

```bash
# Run the benchmark against the local server
python inference.py
```

**Expected Outputs:**
- `[START] task=task1 ...`
- `[STEP] step=1 ... reward=2.45`
- `[END] success=true steps=1 score=1.0000`

### 3. OpenEnv Validation
To ensure the environment perfectly adheres to the OpenEnv standard:

```bash
# Install openenv CLI
pip install openenv-core

# Validate the local environment
openenv validate
```

---

## 📄 License

MIT License
