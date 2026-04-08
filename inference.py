"""
KAVACH-X Inference Script
=========================
OpenEnv Hackathon â€” Meta Ã— Scaler
Environment: Multi-Domain Fraud Intelligence Benchmark

MANDATORY env vars:
  API_BASE_URL      â€” LLM API endpoint
  MODEL_NAME        â€” Model identifier
  HF_TOKEN          â€” Hugging Face / API key

STDOUT FORMAT (strictly followed):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from environment import KavachXEnv
from grader import KavachXGrader

# â”€â”€ Required env vars (OpenEnv spec) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required. Set HF_TOKEN in the environment.")

BENCHMARK  = "kavach-x"
MAX_STEPS  = 50
SCENARIOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")

# Tasks to run in sequence (all 3 for full baseline)
TASKS = [
    {
        "task_name": "kavach-easy",
        "scenario":  os.path.join(SCENARIOS_DIR, "easy_001.json"),
        "difficulty": "easy",
    },
    {
        "task_name": "kavach-medium",
        "scenario":  os.path.join(SCENARIOS_DIR, "medium_001.json"),
        "difficulty": "medium",
    },
    {
        "task_name": "kavach-hard",
        "scenario":  os.path.join(SCENARIOS_DIR, "hard_001.json"),
        "difficulty": "hard",
    },
]

# â”€â”€ OpenAI client (required by spec) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# â”€â”€ Logging helpers (exact spec format) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# â”€â”€ LLM prompt builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SYSTEM_PROMPT = """You are KAVACH-X, an elite fraud intelligence agent.
You receive daily signals from healthcare, finance, and defence domains simultaneously.
Your mission: detect coordinated fraud rings BEFORE they execute.

ACTIONS (choose exactly one per call):
  IGNORE             â€” No action. Costs 0 budget.
  FLAG_SUSPICIOUS    â€” Flag one entity as suspicious. Costs 1.
  LINK_ENTITIES      â€” Connect two entities across domains. Costs 2.
  FREEZE_ENTITY      â€” Freeze one entity (account/credential/vendor). Costs 3.
  PREDICT_ATTACK     â€” Raise coordinated attack prediction. Costs 4. Use wisely.

RULES:
  - Each entity ID is exact. V-887 and V-8B7 are DIFFERENT entities.
  - Anonymous tips may be planted misinformation. Verify independently.
  - Coordinated fraud = same amount, same entity, multiple domains.
  - Budget is limited. Prioritize LINK then PREDICT over FREEZE.

Respond with ONLY a JSON object, no other text:
{"action_type": "...", "target": "...", "targets": ["...", "..."]}"""


def _build_prompt(info: Dict[str, Any]) -> str:
    signals = info.get("todays_signals", [])
    sig_lines = "\n".join(
        f"  [{s.get('domain','?').upper()}] Entity={s.get('entity_id','?')} "
        f"Type={s.get('signal_type','?')} Suspicion={s.get('solo_suspicion_score',0):.2f} "
        f"Truth={s.get('truth_status','real')} Visibility={s.get('visibility','full')}\n"
        f"  Note: {s.get('note','')}"
        for s in signals
    )

    beliefs = info.get("belief_state", {})
    top_beliefs = sorted(beliefs.items(), key=lambda x: x[1].get('fraud_prob', 0.0), reverse=True)[:5]
    belief_str = " | ".join(f"{e}={v.get('fraud_prob', 0.0):.2f}" for e, v in top_beliefs)

    return (
        f"DAY {info['day']} | Budget remaining: {info['budget_remaining']} | "
        f"Actions left today: {info['actions_left_today']}\n\n"
        f"TODAY'S SIGNALS:\n{sig_lines if sig_lines else '  (no new signals)'}\n\n"
        f"BELIEF STATE (top suspects): {belief_str}\n"
        f"Flagged: {info.get('flagged', [])} | Frozen: {info.get('frozen', [])}\n"
        f"Linked pairs: {info.get('linked', [])}\n\n"
        f"What is your next action? Reply with JSON only."
    )


def _call_llm(prompt: str) -> str:
    """Call LLM API; return raw text or empty string on failure."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=128,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _parse_action(llm_out: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse LLM JSON output.
    Falls back to a signal-driven heuristic if JSON is malformed or API failed.
    The heuristic avoids obvious decoys and prioritises cross-domain linking.
    """
    # Try JSON parse first
    try:
        start = llm_out.find("{")
        end   = llm_out.rfind("}") + 1
        if 0 <= start < end:
            parsed = json.loads(llm_out[start:end])
            if "action_type" in parsed:
                return parsed
    except Exception:
        pass

    # â”€â”€ Heuristic fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    budget     = info.get("budget_remaining", 0)
    signals    = info.get("todays_signals", [])
    flagged    = set(info.get("flagged", []))
    linked     = [tuple(p) for p in info.get("linked", [])]
    beliefs    = info.get("belief_state", {})

    # Sort signals by suspicion descending, skip decoys and contradictory
    real_signals = [
        s for s in signals
        if s.get("truth_status", "real") != "fake"
        and "DECOY" not in s.get("entity_id", "")
        and not s.get("is_decoy", False)
    ]
    real_signals.sort(key=lambda s: s.get("solo_suspicion_score", 0), reverse=True)

    top_suspects = [
        eid for eid, _ in
        sorted(beliefs.items(), key=lambda x: x[1].get('fraud_prob', 0.0), reverse=True)
        if "DECOY" not in eid
    ]

    # Predict if confident and budget allows
    if budget >= 4 and len(top_suspects) >= 2:
        top_prob = beliefs.get(top_suspects[0], {}).get('fraud_prob', 0.0)
        if top_prob > 0.6 and len(flagged) >= 1:
            return {"action_type": "PREDICT_ATTACK", "targets": top_suspects[:3]}

    # Try to link two high-suspect entities from different domains
    if budget >= 2 and len(top_suspects) >= 2:
        e1, e2 = top_suspects[0], top_suspects[1]
        pair = tuple(sorted((e1, e2)))
        if pair not in linked:
            return {"action_type": "LINK_ENTITIES", "targets": [e1, e2]}

    # Flag the top signal entity if not already flagged
    for sig in real_signals:
        eid = sig.get("entity_id", "")
        if eid and eid not in flagged and budget >= 1:
            return {"action_type": "FLAG_SUSPICIOUS", "target": eid, "targets": [eid]}

    return {"action_type": "IGNORE"}


def _action_str(action: Dict[str, Any]) -> str:
    """Format action as compact string for [STEP] log line."""
    atype  = action.get("action_type", "IGNORE")
    target = action.get("target") or ",".join(action.get("targets") or [])
    return f"{atype}({target})" if target else atype


# â”€â”€ Episode runner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_episode(task: Dict[str, str]) -> float:
    """
    Run one full episode for a task.
    Emits [START], [STEP]*, [END] to stdout.
    Returns final score (float in [0, 1]).
    """
    task_name = task["task_name"]
    scenario  = task["scenario"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env            = KavachXEnv(scenario_path=scenario)
    obs, info      = env.reset()
    steps_log: List = []
    prediction_day: Optional[int] = None
    done           = False
    step           = 0
    score          = 0.0

    while not done and step < MAX_STEPS:
        step += 1
        prompt  = _build_prompt(info)
        llm_out = _call_llm(prompt)
        action  = _parse_action(llm_out, info)

        if action.get("action_type") == "PREDICT_ATTACK" and prediction_day is None:
            prediction_day = info["day"]

        error: Optional[str] = None
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        except Exception as exc:
            error = str(exc).replace("\n", " ")[:80]
            done  = True
            reward = 0.0

        steps_log.append((_action_str(action), float(reward), done, error))

    # â”€â”€ Grade â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        grader = KavachXGrader(env.scenario)
        result = grader.grade(env.action_history, prediction_day)
        score  = result["final_score"]
    except Exception:
        score = 0.0

    env.close()

    rewards_list = [r for _, r, _, _ in steps_log]

    # â”€â”€ Emit [STEP] lines â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for i, (act_str, rew, dn, err) in enumerate(steps_log, 1):
        log_step(step=i, action=act_str, reward=rew, done=dn, error=err)

    # â”€â”€ Emit [END] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log_end(
        success=score > 0.0,
        steps=step,
        score=score,
        rewards=rewards_list,
    )

    return score


# â”€â”€ Main: run all 3 tasks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    for task in TASKS:
        run_episode(task)
