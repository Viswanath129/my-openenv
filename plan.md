# Email RL Environment Build Plan

Execute the Master Build Plan for the Email RL Environment (OpenEnv). This involves scaffolding the project, building the environment dynamics, integrating task graders, adding an inference script, and deploying the solution following a structured 9-step pipeline.

### 1. Dependencies & Setup
- Install dependencies: `openenv-core`, `fastapi`, `uvicorn`, `pydantic`, `numpy`, `huggingface_hub`.
- Initialize OpenEnv project `email_env`.
- Ensure basic structure exists (`src/models.py`, `src/environment.py`, `src/app.py`, `Dockerfile`, `openenv.yaml`).

### 2. Build Basic Version
- Write minimalist `models.py` with simple `Action` (open/delete) and `Observation`.
- Write minimalist `environment.py` with static `reset()` and basic `step()` logic. Goal: Ensure it runs without crashing.

### 3. Real RL Dynamics
- **State**: Implement an inbox array with email objects (`type`, `urgency`, `wait`).
- **Actions**: Expand to `open`, `delete`, `defer`.
- **Dynamics**: Increment `wait` time each step, add new emails dynamically.
- **Reward Shaping**:
  - `+1.0` for correct action.
  - `-0.5` for delaying urgent emails.
  - `-0.1 * len(inbox)` backlog penalty.

### 4. Tasks & Graders
- **Task 1 (Easy)**: Basic triage of 1 email.
- **Task 2 (Medium)**: Priority handling for multiple emails.
- **Task 3 (Hard)**: Dynamic inbox management with limited steps.
- **Grader**: Deterministic function mapping actions to `0.0 - 1.0` scores.

### 5. Baseline Inference
- Create `inference.py` running a baseline agent.
- Output MUST follow STRICT formatted logs (`[START]`, `[STEP]`, `[END]`).

### 6. Validation
- Run local OpenEnv server and verify `/reset` and `/step` endpoints.
- Run `openenv validate` to ensure no errors.

### 7. Docker
- Verify `Dockerfile` can build (`docker build -t email-env .`).
- Run container locally to ensure it works.

### 8. Deployment
- Deploy environment to Hugging Face Spaces using `openenv push`.

### 9. Documentation
- Write a comprehensive `README.md` covering the Problem, RL Justification, Environment Design, Tasks, Rewards, and Run Instructions.
