"""
KAVACH-X Tasks
==============
Task definitions, difficulty routing, and adaptive curriculum engine.

Three tasks:
  Task 1 (easy)   — Operation Ghost Patient    — Single domain, 5 days
  Task 2 (medium) — Operation UPI Phantom      — Two domains, 10 days, 1 decoy
  Task 3 (hard)   — Operation Ghost Ring       — Three domains, 15 days, full adversarial

Adaptive curriculum:
  - Score > 0.8  on current task → promote to harder
  - Score < 0.5  on current task → stay / demote (only from medium/hard)
  - Score 0.5–0.8 → stay at current difficulty
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from environment import KavachXEnv
from grader import KavachXGrader


# ─────────────────────────────────────────────────────────────────────────────
#  Task registry
# ─────────────────────────────────────────────────────────────────────────────

TASK_REGISTRY: List[Dict[str, Any]] = [
    {
        "task_id": "kavach-easy",
        "name": "Operation Ghost Patient",
        "difficulty": "easy",
        "scenario_file": "easy_001.json",
        "description": (
            "Healthcare only. A 12-day-old hospital files 23 PMJAY claims worth ₹21L "
            "using a general medicine doctor billing orthopedic surgeries. "
            "Ghost patients — none were actually admitted."
        ),
        "expected_llm_score_range": (0.75, 0.92),
        "domains": ["healthcare"],
        "total_days": 5,
    },
    {
        "task_id": "kavach-medium",
        "name": "Operation UPI Phantom",
        "difficulty": "medium",
        "scenario_file": "medium_001.json",
        "description": (
            "Healthcare + Finance. Hospital files PMJAY claims; payments routed to a "
            "UPI account opened 3 days before empanelment. Money moves to a shell account "
            "at 1AM matching exact claim amount. One decoy injected. "
            "REQUEST_AUDIT on A-331 Day 4 pays off here — resolves with high confidence."
        ),
        "expected_llm_score_range": (0.35, 0.55),
        "domains": ["healthcare", "finance"],
        "total_days": 10,
    },
    {
        "task_id": "kavach-hard",
        "name": "Operation Ghost Ring",
        "difficulty": "hard",
        "scenario_file": "hard_001.json",
        "description": (
            "Healthcare + Finance + Defence. Full staging → execution phase. Two cities "
            "(Pune and Nagpur). Shared doctor credential, recycled GST vendor, 3-layer "
            "money laundering. Adversarial traps: whistleblower plant, domain blackout "
            "(Days 11-12), retroactive record revision (Day 13), near-duplicate entity IDs. "
            "CROSS_VERIFY and REQUEST_AUDIT scored via information_efficiency_bonus. "
            "Contradiction handling scored for NOT freezing decoys after whistleblower."
        ),
        "expected_llm_score_range": (0.05, 0.18),
        "domains": ["healthcare", "finance", "defence"],
        "total_days": 15,
    },
]

DIFFICULTIES = ["easy", "medium", "hard"]
DIFFICULTY_INDEX = {d: i for i, d in enumerate(DIFFICULTIES)}

SCENARIOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")


# ─────────────────────────────────────────────────────────────────────────────
#  Task Engine
# ─────────────────────────────────────────────────────────────────────────────

class KavachXTaskEngine:
    """
    Orchestrates KAVACH-X episodes across three difficulty levels.

    Adaptive curriculum routing:
      - After each episode the score determines the next difficulty.
      - Fully deterministic: same seed → same scenario → same difficulty path.

    Usage:
        engine = KavachXTaskEngine(start_difficulty="easy")
        results, history = engine.evaluate_episode(my_agent_policy)
    """

    PROMOTE_THRESHOLD = 0.80   # score > 0.8 → harder task
    DEMOTE_THRESHOLD  = 0.50   # score < 0.5 → stay / easier task

    def __init__(self, start_difficulty: str = "easy") -> None:
        if start_difficulty not in DIFFICULTY_INDEX:
            raise ValueError(
                f"Unknown difficulty '{start_difficulty}'. "
                f"Choose from {DIFFICULTIES}."
            )
        self.current_idx: int = DIFFICULTY_INDEX[start_difficulty]
        self.episode_results: List[Dict[str, Any]] = []

    @property
    def current_difficulty(self) -> str:
        return DIFFICULTIES[self.current_idx]

    @property
    def current_task(self) -> Dict[str, Any]:
        return TASK_REGISTRY[self.current_idx]

    def get_env(self) -> KavachXEnv:
        """Instantiate the environment for the current difficulty."""
        task = self.current_task
        scenario_path = os.path.join(SCENARIOS_DIR, task["scenario_file"])
        return KavachXEnv(scenario_path=scenario_path)

    def evaluate_episode(
        self,
        agent_policy: Callable[[Any, Dict], Dict],
        max_steps: int = 60,
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Run one full episode using agent_policy and return graded results.

        Args:
            agent_policy: Callable(obs_vector, info_dict) → action_dict
                          action_dict must have "action_type" key.
            max_steps:    Safety cap on environment steps.

        Returns:
            (results_dict, action_history)
            results_dict contains "final_score" (float in [0,1]).
        """
        env = self.get_env()
        obs, info = env.reset()

        done = False
        step = 0
        prediction_day: Optional[int] = None

        while not done and step < max_steps:
            action_dict = agent_policy(obs, info)

            if action_dict.get("action_type") == "PREDICT_ATTACK" and prediction_day is None:
                prediction_day = info["day"]

            obs, _reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated
            step += 1

        grader = KavachXGrader(env.scenario)
        results = grader.grade(env.action_history, prediction_day)

        # Adaptive routing
        score = results["final_score"]
        routing_msg = self._route(score)
        results["routing_message"] = routing_msg
        results["task_name"] = self.current_task["name"]
        results["task_id"] = self.current_task["task_id"]

        self.episode_results.append(results)
        return results, env.action_history

    def _route(self, score: float) -> str:
        """Deterministically route to next difficulty based on score."""
        if score > self.PROMOTE_THRESHOLD:
            if self.current_idx < len(DIFFICULTIES) - 1:
                self.current_idx += 1
                return f"Score {score:.3f} > {self.PROMOTE_THRESHOLD} → promoted to {self.current_difficulty}"
            return f"Score {score:.3f} > {self.PROMOTE_THRESHOLD} → already at hardest difficulty"
        elif score < self.DEMOTE_THRESHOLD:
            if self.current_idx > 0:
                self.current_idx -= 1
                return f"Score {score:.3f} < {self.DEMOTE_THRESHOLD} → demoted to {self.current_difficulty}"
            return f"Score {score:.3f} < {self.DEMOTE_THRESHOLD} → already at easiest difficulty"
        return f"Score {score:.3f} in [{self.DEMOTE_THRESHOLD}, {self.PROMOTE_THRESHOLD}] → maintained {self.current_difficulty}"

    def run_all_tasks(
        self,
        agent_policy: Callable[[Any, Dict], Dict],
        max_steps: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Run all three tasks in order (easy → medium → hard).
        Ignores adaptive routing — runs all three regardless of score.
        Used for producing reproducible baseline reports.

        Returns list of 3 result dicts.
        """
        all_results = []
        for task in TASK_REGISTRY:
            scenario_path = os.path.join(SCENARIOS_DIR, task["scenario_file"])
            env = KavachXEnv(scenario_path=scenario_path)
            obs, info = env.reset()

            done = False
            step = 0
            prediction_day: Optional[int] = None

            while not done and step < max_steps:
                action_dict = agent_policy(obs, info)
                if action_dict.get("action_type") == "PREDICT_ATTACK" and prediction_day is None:
                    prediction_day = info["day"]
                obs, _, terminated, truncated, info = env.step(action_dict)
                done = terminated or truncated
                step += 1

            grader = KavachXGrader(env.scenario)
            result = grader.grade(env.action_history, prediction_day)
            result["task_id"] = task["task_id"]
            result["task_name"] = task["name"]
            result["difficulty"] = task["difficulty"]
            all_results.append(result)

        return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  Task metadata helper (for openenv validate / external tooling)
# ─────────────────────────────────────────────────────────────────────────────

def list_tasks() -> List[Dict[str, Any]]:
    """Return metadata for all tasks (used by openenv validate)."""
    return [
        {
            "task_id": t["task_id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "domains": t["domains"],
            "total_days": t["total_days"],
            "description": t["description"],
            "expected_score_range": t["expected_llm_score_range"],
            "scenario_file": t["scenario_file"],
        }
        for t in TASK_REGISTRY
    ]


if __name__ == "__main__":
    # Quick sanity check
    import json

    for task in list_tasks():
        print(json.dumps(task, indent=2))
