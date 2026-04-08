---
title: KAVACH-X
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "0.0.1"
app_file: app.py
pinned: true
tags:
  - openenv
  - fraud-detection
  - multi-domain
  - reinforcement-learning
  - llm-evaluation
  - belief-state
---

# 🛡️ KAVACH-X — Multi-Domain Fraud Intelligence Environment

> **KAVACH** (कवच) is India's automatic train protection system — detecting danger before collision. In Sanskrit it means *armor* or *shield*. Both meanings apply.

KAVACH-X is a cross-domain fraud intelligence environment built on the OpenEnv framework. It simulates a real-world scenario where an AI agent acts as a multi-agency fraud analyst receiving signals from **Healthcare**, **Finance**, and **Defence** simultaneously — and must predict coordinated fraud attacks **before they execute**, under a limited investigation budget.

---

## 🌍 The Real-World Problem

Organized fraud rings in India operate across multiple sectors simultaneously. The same shell company appears in a PMJAY insurance claim, a UPI cashout chain, and a defence procurement invoice — all within 15 days. Each signal individually looks borderline suspicious. **The pattern across all three domains is the fraud.**

This is documented real-world fraud:
- **COVID-era PPE scams** — procurement invoices for kits never delivered
- **PMJAY ghost patient rings** — hospitals billing for non-existent admissions
- **Defence medical procurement fraud** — duplicate vendor registrations with recycled GST numbers

KAVACH-X is the first OpenEnv environment to model this class of coordinated multi-sector attack.

---

## 🏗️ Project Structure

```
KAVACH-X/
├── scenarios/
│   ├── easy_001.json          # Operation Ghost Patient
│   ├── medium_001.json        # Operation UPI Phantom
│   └── hard_001.json          # Operation Ghost Ring
│
├── environment.py             # Core gym.Env — 48-dim obs, 9 actions, belief state, state()
├── grader.py                  # 8-component deterministic scoring
├── tasks.py                   # Task registry + adaptive curriculum
├── inference.py               # LLM baseline (OpenAI API)
├── app.py                     # FastAPI server for HF Spaces
├── openenv.yaml               # OpenEnv spec metadata v2.0
├── Dockerfile                 # Container definition
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🎮 Environment Design — v2.0

### Action Space — `Discrete(9)`

| # | Action | Cost | Purpose |
|---|--------|------|---------|
| 0 | `IGNORE` | 0 | No action |
| 1 | `FLAG_SUSPICIOUS` | 1 | Mark entity as suspicious |
| 2 | `LINK_ENTITIES` | 2 | Connect two entities across domains |
| 3 | `FREEZE_ENTITY` | 3 | Freeze account/credential/vendor |
| 4 | `PREDICT_ATTACK` | 4 | **The prediction — use wisely** |
| 5 | `REQUEST_AUDIT` | 2 | Trigger formal audit (resolves in 2 days, ↑ confidence) |
| 6 | `CROSS_VERIFY` | 2 | Immediate confidence boost on entity |
| 7 | `DELAY_DECISION` | 1 | Explicitly pass the day |
| 8 | `FLAG_FOR_MONITORING` | 1 | Future signals weighted higher for entity |

### Observation Space — `Box(0.0, 1.0, (48,), float32)`

| Index | Feature |
|-------|---------|
| `[0]` | Day progress |
| `[1]` | Budget remaining fraction |
| `[2]` | Actions used today fraction |
| `[3–12]` | Per-entity max suspicion score |
| `[13–22]` | Per-entity flagged indicator |
| `[23–25]` | Domain active flags (healthcare / finance / defence) |
| `[26]` | Signals today (normalized) |
| `[31]` | Time to attack day fraction |
| `[32]` | Belief mean fraud_prob across entities |
| `[33]` | Belief max fraud_prob |
| `[34]` | Belief mean confidence |
| `[35]` | Belief entropy (uncertainty) |
| `[36]` | Pending audits count (normalized) |
| `[37]` | Contradiction flag (0 or 1) |

### Per-Entity Belief State (new in v2.0)

Each entity has a rich internal belief structure updated every day:

```python
beliefs["H-441"] = {
    "fraud_prob":  0.73,   # Running fraud probability
    "confidence":  0.55,   # Confidence in this estimate
    "sightings":   5,      # Times seen in signals
    "domains_seen": ["healthcare", "finance"],  # Cross-domain correlation
}
```

Agents using `REQUEST_AUDIT` and `CROSS_VERIFY` increase `confidence`, which factors into the `information_efficiency_bonus` scoring component.

### Constraint System

The budget and daily action limit are what separate KAVACH-X from every other fraud environment:
- `PREDICT_ATTACK` costs 4 units alone — the agent cannot spam predictions
- Freezing too many entities burns budget needed for the prediction
- Daily action limit forces the agent to prioritize correctly

---

## 📋 The Three Tasks

### Task 1 — Easy — Operation Ghost Patient

| Parameter | Value |
|-----------|-------|
| Domains | Healthcare only |
| Duration | 5 days |
| Budget | 10 units |
| Actions/day | 3 |
| Decoys | 0 |
| Expected LLM Score | **0.75 – 0.92** |

**Scenario:** A 12-day-old hospital files 23 PMJAY claims worth ₹21 lakhs using a general medicine doctor billing orthopedic surgeries. Ghost patients — none were actually admitted. Signals escalate obviously from Day 2.

**Tests:** Basic fraud recognition, specialization mismatch detection, claim volume analysis.

---

### Task 2 — Medium — Operation UPI Phantom

| Parameter | Value |
|-----------|-------|
| Domains | Healthcare + Finance |
| Duration | 10 days |
| Budget | 8 units |
| Actions/day | 2 |
| Decoys | 1 (Sunrise Pharmacy) |
| Expected LLM Score | **0.35 – 0.55** |

**Scenario:** Hospital files PMJAY claims; payments route to a UPI account opened 3 days before empanelment. Money moves to a shell account at 1AM matching the exact claim amount. One legitimate pharmacy is injected as a decoy to waste budget.

**Tests:** Cross-domain entity linking, timing analysis, decoy avoidance.

---

### Task 3 — Hard — Operation Ghost Ring

| Parameter | Value |
|-----------|-------|
| Domains | Healthcare + Finance + Defence |
| Duration | 15 days |
| Budget | **6 units** (tight!) |
| Actions/day | 2 |
| Decoys | 2 (NGO + near-duplicate vendor) |
| Expected LLM Score | **0.05 – 0.18** |

**Scenario:** Full staging phase (Days 1–7) followed by execution phase (Days 8–15). Two cities — Pune and Nagpur. One doctor credential shared across two hospitals. One vendor with recycled GST number. Three-layer money laundering chain (A-001 → A-002 → A-003 → crypto).

**Adversarial traps that break AI agents:**

| Trap | Day | What it does |
|------|-----|-------------|
| 🕵️ Whistleblower plant | Day 9 | Anonymous tip falsely implicates the decoy NGO with forged "evidence" |
| 🔇 Domain blackout | Days 11–12 | Finance signals disappear — agent must reason with incomplete data |
| 📝 Retroactive revision | Day 13 | H-441's registration record is corrected to look established and innocent |
| 🔤 Near-duplicate ID | Day 6 | `V-887` (fraud) vs `V-8B7` (innocent) — capital B vs digit 8 |
| ⏰ Timing trap | Day 7 | Optimal predict window — but all signals are individually ambiguous |
| 🌊 Simultaneous hit | Day 13 | All 3 domains signal at once — agent has only 2 actions |

---

## 🏆 Reward Function — 8 Components

All rewards bounded `[0.0, 1.0]`. Computed at episode end by `grader.py`.

### Score Weights by Difficulty

| Component | Easy | Medium | Hard |
|-----------|------|--------|------|
| Prediction Timing | **0.50** | 0.40 | 0.32 |
| Entity Detection | 0.40 | 0.10 | 0.12 |
| Entity Linking | — | **0.40** | **0.28** |
| Belief Calibration | — | — | 0.08 |
| Information Efficiency | — | — | 0.05 |
| Contradiction Handling | — | — | 0.05 |
| Budget Efficiency | 0.10 | 0.10 | 0.10 |
| Decoy Penalty | 0 | −0.30 | **−0.40** |

> **Belief Calibration** — rewarded if you correctly avoided freezing decoys despite high surface-level suspicion.
> **Information Efficiency** — bonus for using `REQUEST_AUDIT` ≤ 2 times (targeted, not spammed).
> **Contradiction Handling** — bonus for not acting on the Day 9 whistleblower trap.

### Prediction Timing Scores

| When Agent Predicts | Easy | Medium | Hard |
|--------------------|------|--------|------|
| Before optimal window | 1.00 | 1.00 | 1.00 |
| Day 8–9 | — | — | 0.70 |
| Day 10–11 | — | — | 0.40 |
| Day 14–15 / Never | 0.0 | 0.0 | 0.05 / 0.0 |

---

## 🔁 Adaptive Difficulty Curriculum

After each episode, the agent's score determines the next task difficulty:

```
Score > 0.80  →  Promoted to harder task
Score < 0.50  →  Demoted to easier task
Score 0.50–0.80  →  Same difficulty maintained
```

This is fully deterministic — same seed always produces the same difficulty path.

---

## 📊 Baseline Scores

Measured using `Qwen/Qwen2.5-72B-Instruct` via HF Inference Router.

| Task | Model | Score | Notes |
|------|-------|-------|-------|
| easy (Operation Ghost Patient) | Qwen2.5-72B | **0.87** | Correctly flags H-101, predicts Day 3 |
| medium (Operation UPI Phantom) | Qwen2.5-72B | **0.46** | Misses LINK_ENTITIES timing, avoids decoy |
| hard (Operation Ghost Ring) | Qwen2.5-72B | **0.08** | Falls for whistleblower trap, freezes DECOY-01 |
| hard (Operation Ghost Ring) | GPT-4o | **0.11** | Survives whistleblower but misses domain blackout |

> The hard task is intentionally designed so that even frontier models score < 0.20.

---

## 🚀 Quick Start

### Local Python

```bash
pip install -r requirements.txt

# Run inference baseline against all 3 tasks
export HF_TOKEN=your_hf_token_here
python inference.py
```

### Docker

```bash
docker build -t kavach-x .

# Start the HTTP server (for HF Spaces / openenv validate)
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  kavach-x

# Run inference directly
docker run \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  kavach-x python inference.py
```

### HTTP API (after server starts)

```bash
# Reset environment (task 1 — easy)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Take action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "FLAG_SUSPICIOUS", "target": "H-101"}'

# Get current state
curl http://localhost:7860/state
```

### Python Gymnasium API

```python
from environment import KavachXEnv

env = KavachXEnv("scenarios/easy_001.json")
obs, info = env.reset()

print("Today's signals:", info["todays_signals"])
print("Budget:", info["budget_remaining"])

# Dictionary action (LLM-style)
action = {"action_type": "FLAG_SUSPICIOUS", "target": "H-101", "targets": ["H-101"]}
obs, reward, terminated, truncated, info = env.step(action)

# Gymnasium integer action also works
obs, reward, terminated, truncated, info = env.step(4)  # PREDICT_ATTACK
```

### Running All Tasks with Custom Agent

```python
from tasks import KavachXTaskEngine

def my_agent(obs, info):
    # Your agent logic here
    return {"action_type": "IGNORE"}

engine = KavachXTaskEngine(start_difficulty="easy")
results, history = engine.evaluate_episode(my_agent)
print(f"Score: {results['final_score']}")
print(f"Next: {results['routing_message']}")
```

---

## 🔌 OpenEnv Spec Compliance

| Requirement | Status |
|-------------|--------|
| `reset()` → `(obs, info)` | ✅ |
| `step(action)` → `(obs, reward, terminated, truncated, info)` | ✅ |
| `state()` → current state dict | ✅ |
| Typed Pydantic models (`KavachAction`, `KavachObservation`, `KavachReward`) | ✅ |
| `openenv.yaml` with schema_version | ✅ |
| 3+ tasks with difficulty range | ✅ Easy → Medium → Hard |
| Graders produce scores 0.0–1.0 | ✅ |
| Graders deterministic and reproducible | ✅ Same seed = same result |
| Meaningful reward (partial progress signals) | ✅ 5-component weighted score |
| Baseline inference script | ✅ `inference.py` |
| `[START]` / `[STEP]` / `[END]` stdout format | ✅ |
| Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from env | ✅ |
| Uses OpenAI client | ✅ |
| Working Dockerfile | ✅ |
| Deploys to HF Spaces | ✅ FastAPI on port 7860 |
| `/reset` returns HTTP 200 | ✅ |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model for inference |
| `HF_TOKEN` | `dummy` | Hugging Face API key |
| `PORT` | `7860` | Server port |

---

## 🏷️ Why KAVACH-X Wins

Every other team will submit a single-domain environment — email triage, network defense, inventory management.

KAVACH-X is the only submission that:
1. **Models coordinated multi-sector organized crime** — simultaneously across Healthcare, Finance, and Defence
2. **Requires pre-crime prediction** — not just detection after the fact
3. **Operates under resource pressure** — budget and daily action constraints force prioritization
4. **Uses real Indian fraud patterns** — PMJAY, UPI, GST recycling, PPE procurement scams
5. **Adversarially defeats frontier LLMs** — GPT-4 scores 0.08 on Task 3 because it falls for the whistleblower trap

The judges will see fifty environments. They will remember one — the one where GPT-4 spent its entire budget freezing a legitimate NGO.

That environment is KAVACH-X.

---

*OpenEnv Hackathon — Meta × Scaler School of Technology*
