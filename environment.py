"""
KAVACH-X Environment — Enhanced v2.0
=====================================
Multi-Domain Cross-Sector Fraud Intelligence Environment
OpenEnv Hackathon — Meta × Scaler

Upgrades over v1.0:
  - Full per-entity belief state (fraud_prob, confidence, sightings, domains_seen)
  - 9-action space: adds REQUEST_AUDIT, CROSS_VERIFY, DELAY_DECISION, FLAG_FOR_MONITORING
  - 48-dimensional observation with belief metrics + contradiction flag
  - Audit resolution system (audits resolve after 2 days, boosting confidence)
  - Contradiction / whistleblower handling
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic Models (OpenEnv typed spec)
# ─────────────────────────────────────────────────────────────────────────────

class KavachAction(BaseModel):
    """Typed action model for KAVACH-X."""
    action_type: str = Field(
        ...,
        description=(
            "One of: IGNORE | FLAG_SUSPICIOUS | LINK_ENTITIES | FREEZE_ENTITY | "
            "PREDICT_ATTACK | REQUEST_AUDIT | CROSS_VERIFY | DELAY_DECISION | "
            "FLAG_FOR_MONITORING"
        )
    )
    target: Optional[str] = Field(None, description="Single entity ID")
    targets: Optional[List[str]] = Field(None, description="Entity ID list")

    model_config = {"extra": "allow"}


class KavachObservation(BaseModel):
    """Typed 48-dim observation model for KAVACH-X."""
    day_progress: float = Field(..., ge=0.0, le=1.0)
    budget_remaining_frac: float = Field(..., ge=0.0, le=1.0)
    actions_used_frac: float = Field(..., ge=0.0, le=1.0)
    entity_suspicion: List[float]
    entity_flagged: List[float]
    domain_healthcare: float
    domain_finance: float
    domain_defence: float
    signals_today: float
    time_to_attack_frac: float
    belief_mean_prob: float
    belief_max_prob: float
    belief_mean_conf: float
    belief_entropy: float
    pending_audits_frac: float
    contradiction_flag: float
    vector: List[float] = Field(..., description="Full 48-dim observation vector")


class KavachReward(BaseModel):
    """Typed reward model for KAVACH-X."""
    value: float = Field(..., ge=-1.0, le=1.0)
    is_final: bool = False
    components: Dict[str, float] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  Action space — 9 actions
# ─────────────────────────────────────────────────────────────────────────────

ACTION_NAMES: List[str] = [
    "IGNORE",               # 0 — cost 0
    "FLAG_SUSPICIOUS",      # 1 — cost 1
    "LINK_ENTITIES",        # 2 — cost 2
    "FREEZE_ENTITY",        # 3 — cost 3
    "PREDICT_ATTACK",       # 4 — cost 4
    "REQUEST_AUDIT",        # 5 — cost 2 (resolves in 2 days, ↑confidence)
    "CROSS_VERIFY",         # 6 — cost 2 (immediate confidence boost)
    "DELAY_DECISION",       # 7 — cost 1 (explicitly pass the day)
    "FLAG_FOR_MONITORING",  # 8 — cost 1 (weight future signals higher)
]

ACTION_COSTS: Dict[str, int] = {
    "IGNORE": 0,
    "FLAG_SUSPICIOUS": 1,
    "LINK_ENTITIES": 2,
    "FREEZE_ENTITY": 3,
    "PREDICT_ATTACK": 4,
    "REQUEST_AUDIT": 2,
    "CROSS_VERIFY": 2,
    "DELAY_DECISION": 1,
    "FLAG_FOR_MONITORING": 1,
}

VALID_ACTIONS: Set[str] = set(ACTION_COSTS.keys())


# ─────────────────────────────────────────────────────────────────────────────
#  Core Environment
# ─────────────────────────────────────────────────────────────────────────────

class KavachXEnv(gym.Env):
    """
    KAVACH-X: Multi-Domain Fraud Intelligence Environment v2.0

    Observation space: Box(0, 1, (48,), float32)
    Action space:      Discrete(9)
    Episode:           Episodic, 5–15 days depending on difficulty
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, scenario_path: Optional[str] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        if scenario_path is None:
            scenario_path = os.path.join(
                os.path.dirname(__file__), "scenarios", "easy_001.json"
            )

        with open(scenario_path, "r", encoding="utf-8") as f:
            self.scenario: Dict[str, Any] = json.load(f)

        self.total_days: int = self.scenario["total_days"]
        self.total_budget: int = self.scenario["total_budget"]
        self.max_actions_per_day: int = self.scenario["max_actions_per_day"]
        self.events: List[Dict] = self.scenario["events"]
        self.entities: Dict[str, Dict] = self.scenario["entities"]

        # 48-dim observation space
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(48,), dtype=np.float32)
        # 9 discrete actions
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        self._reset_state()

    # ─── Internal state ───────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self.current_day: int = 1
        self.budget_remaining: int = self.total_budget
        self.actions_used_today: int = 0

        self.flagged_entities: Set[str] = set()
        self.frozen_entities: Set[str] = set()
        self.linked_pairs: Set[Tuple[str, str]] = set()
        self.monitored_entities: Set[str] = set()
        self.pending_audits: List[Dict] = []   # [{entity, resolve_day}]

        self.prediction_made: bool = False
        self.prediction_day: Optional[int] = None
        self.action_history: List[Dict] = []

        # Per-entity belief state
        self.beliefs: Dict[str, Dict] = {
            eid: {
                "fraud_prob": 0.1,
                "confidence": 0.1,
                "sightings": 0,
                "domains_seen": set(),
                "last_update_day": 0,
            }
            for eid in self.entities.keys()
        }
        self.contradiction_flag: bool = False
        self._processed_event_ids: Set[str] = set()

    # ─── Belief system ────────────────────────────────────────────────────────

    def _update_beliefs(self) -> None:
        """Update per-entity beliefs from events up to current day."""
        self.contradiction_flag = False

        # Resolve pending audits
        for audit in list(self.pending_audits):
            if self.current_day >= audit["resolve_day"]:
                eid = audit["entity"]
                if eid in self.beliefs:
                    b = self.beliefs[eid]
                    b["confidence"] = float(np.clip(b["confidence"] + 0.2, 0.0, 1.0))
                    b["fraud_prob"]  = float(np.clip(max(b["fraud_prob"], 0.5), 0.0, 1.0))
                self.pending_audits.remove(audit)

        for event in self.events:
            eid_event = event.get("event_id", "")
            if event["day"] > self.current_day:
                continue
            if eid_event in self._processed_event_ids:
                continue

            entity = event.get("entity_id")
            if not entity or entity not in self.beliefs:
                self._processed_event_ids.add(eid_event)
                continue

            b = self.beliefs[entity]
            b["sightings"] += 1
            b["domains_seen"].add(event.get("domain", ""))
            b["last_update_day"] = self.current_day

            score      = event.get("solo_suspicion_score", 0.0)
            truth      = event.get("truth_status", "real")
            visibility = event.get("visibility", "full")

            weight = 0.35
            if truth == "fake":
                weight = -0.15
            elif truth == "uncertain":
                weight = 0.10

            if visibility == "contradictory":
                weight *= 0.4
                self.contradiction_flag = True
            elif visibility == "partial":
                weight *= 0.6

            # Monitored entities get boosted signal weight
            if entity in self.monitored_entities:
                weight *= 1.25

            delta = score * weight
            b["fraud_prob"]  = float(np.clip(b["fraud_prob"]  + delta,        0.0, 1.0))
            b["confidence"]  = float(np.clip(b["confidence"]  + abs(delta) * 0.5, 0.0, 1.0))

            self._processed_event_ids.add(eid_event)

    def _belief_metrics(self) -> Tuple[float, float, float, float]:
        """Return (mean_prob, max_prob, mean_conf, entropy)."""
        probs = [b["fraud_prob"]  for b in self.beliefs.values()]
        confs = [b["confidence"]  for b in self.beliefs.values()]
        if not probs:
            return 0.0, 0.0, 0.0, 0.0
        mean_prob = float(np.mean(probs))
        max_prob  = float(np.max(probs))
        mean_conf = float(np.mean(confs))
        eps = 1e-6
        entropy = -sum(p * math.log(p + eps) for p in probs) / max(1, len(probs))
        entropy  = float(entropy / max(eps, math.log(len(probs) + eps)))
        return mean_prob, max_prob, mean_conf, min(1.0, entropy)

    # ─── Observation ─────────────────────────────────────────────────────────

    def _get_obs_vector(self) -> np.ndarray:
        self._update_beliefs()
        obs = np.zeros(48, dtype=np.float32)

        obs[0] = self.current_day / max(1, self.total_days)
        obs[1] = self.budget_remaining / max(1, self.total_budget)
        obs[2] = self.actions_used_today / max(1, self.max_actions_per_day)

        entity_ids = list(self.entities.keys())[:10]
        for i, eid in enumerate(entity_ids):
            suspicion = max(
                (e.get("solo_suspicion_score", 0.0)
                 for e in self.events
                 if e["day"] <= self.current_day and e.get("entity_id") == eid),
                default=0.0,
            )
            obs[3 + i]  = float(suspicion)
            obs[13 + i] = 1.0 if eid in self.flagged_entities else 0.0

        domains = self.scenario.get("domains", [])
        obs[23] = 1.0 if "healthcare" in domains else 0.0
        obs[24] = 1.0 if "finance"    in domains else 0.0
        obs[25] = 1.0 if "defence"    in domains else 0.0

        signals_today = len([e for e in self.events if e["day"] == self.current_day])
        obs[26] = min(1.0, signals_today / 10.0)

        attack_day = self.scenario["ground_truth"]["attack_starts_day"]
        obs[31] = max(0, attack_day - self.current_day) / max(1, self.total_days)

        # Belief metrics (dims 32–37)
        mean_prob, max_prob, mean_conf, entropy = self._belief_metrics()
        obs[32] = mean_prob
        obs[33] = max_prob
        obs[34] = mean_conf
        obs[35] = entropy
        obs[36] = min(1.0, len(self.pending_audits) / 5.0)
        obs[37] = 1.0 if self.contradiction_flag else 0.0

        return obs

    def _get_typed_obs(self) -> KavachObservation:
        vec = self._get_obs_vector()
        return KavachObservation(
            day_progress=float(vec[0]),
            budget_remaining_frac=float(vec[1]),
            actions_used_frac=float(vec[2]),
            entity_suspicion=[float(vec[3 + i]) for i in range(10)],
            entity_flagged=[float(vec[13 + i]) for i in range(10)],
            domain_healthcare=float(vec[23]),
            domain_finance=float(vec[24]),
            domain_defence=float(vec[25]),
            signals_today=float(vec[26]),
            time_to_attack_frac=float(vec[31]),
            belief_mean_prob=float(vec[32]),
            belief_max_prob=float(vec[33]),
            belief_mean_conf=float(vec[34]),
            belief_entropy=float(vec[35]),
            pending_audits_frac=float(vec[36]),
            contradiction_flag=float(vec[37]),
            vector=vec.tolist(),
        )

    def _get_info(self) -> Dict[str, Any]:
        todays_events = [e for e in self.events if e["day"] == self.current_day]
        belief_snapshot = {
            eid: {
                "fraud_prob":  round(b["fraud_prob"], 3),
                "confidence":  round(b["confidence"], 3),
                "sightings":   b["sightings"],
                "domains_seen": sorted(b["domains_seen"]),
            }
            for eid, b in self.beliefs.items()
        }
        return {
            "day": self.current_day,
            "budget_remaining": self.budget_remaining,
            "actions_left_today": self.max_actions_per_day - self.actions_used_today,
            "todays_signals": todays_events,
            "flagged": sorted(self.flagged_entities),
            "frozen":  sorted(self.frozen_entities),
            "linked":  [list(p) for p in self.linked_pairs],
            "monitored": sorted(self.monitored_entities),
            "pending_audits": self.pending_audits,
            "prediction_made": self.prediction_made,
            "prediction_day":  self.prediction_day,
            "belief_state":    belief_snapshot,
            "contradiction_flag": self.contradiction_flag,
            "scenario_id": self.scenario["scenario_id"],
            "difficulty":  self.scenario["difficulty"],
        }

    def _ground_truth_sets(self) -> Tuple[Set[str], Set[str]]:
        """Return fraud and decoy entity sets from scenario ground truth."""
        gt = self.scenario.get("ground_truth", {})
        fraud = set(gt.get("fraud_ring_entities", []))
        decoys = set(gt.get("decoy_entities", []))
        return fraud, decoys

    def _compute_step_reward(
        self,
        act_type: str,
        targets: List[str],
        valid: bool,
        cost: int,
        day_before: int,
        prediction_before: bool,
    ) -> float:
        """
        Compute shaped reward in [0.0, 1.0].
        Rewards partial progress and lowers reward for wasteful/destructive actions.
        """
        fraud_entities, decoy_entities = self._ground_truth_sets()
        attack_day = int(
            self.scenario.get("ground_truth", {}).get("attack_starts_day", self.total_days)
        )

        unique_targets = list(dict.fromkeys(t for t in targets if t))

        reward = 0.08

        if not valid:
            reward -= 0.20

        reward -= min(0.12, cost * 0.03)

        if act_type in {
            "FLAG_SUSPICIOUS",
            "FREEZE_ENTITY",
            "REQUEST_AUDIT",
            "CROSS_VERIFY",
            "FLAG_FOR_MONITORING",
        }:
            if not unique_targets:
                reward -= 0.08
            for eid in unique_targets[:2]:
                if eid in fraud_entities:
                    reward += 0.16
                elif eid in decoy_entities:
                    reward -= 0.22 if act_type == "FREEZE_ENTITY" else 0.14
                else:
                    reward -= 0.05

        elif act_type == "LINK_ENTITIES":
            if len(unique_targets) >= 2:
                pair = unique_targets[:2]
                fraud_hits = sum(1 for eid in pair if eid in fraud_entities)
                touched_decoy = any(eid in decoy_entities for eid in pair)

                if fraud_hits == 2:
                    pair_domains = {
                        self.entities.get(eid, {}).get("domain")
                        for eid in pair
                        if eid in self.entities
                    }
                    reward += 0.24 if len(pair_domains) >= 2 else 0.16
                elif fraud_hits == 1:
                    reward += 0.05

                if touched_decoy:
                    reward -= 0.16
            else:
                reward -= 0.10

        elif act_type == "PREDICT_ATTACK":
            if prediction_before:
                reward -= 0.20
            elif day_before <= attack_day:
                reward += 0.35
            elif day_before <= attack_day + 2:
                reward += 0.18
            else:
                reward -= 0.16

        elif act_type in {"IGNORE", "DELAY_DECISION"}:
            if day_before >= attack_day:
                reward -= 0.14
            else:
                reward += 0.02

        contradiction_today = any(
            event.get("day") == day_before and event.get("visibility") == "contradictory"
            for event in self.events
        )
        if contradiction_today:
            acted_on_decoy = any(eid in decoy_entities for eid in unique_targets)
            if acted_on_decoy and act_type in {"FLAG_SUSPICIOUS", "FREEZE_ENTITY"}:
                reward -= 0.18
            elif act_type in {"REQUEST_AUDIT", "CROSS_VERIFY", "IGNORE", "DELAY_DECISION"}:
                reward += 0.05

        return float(np.clip(reward, 0.0, 1.0))

    # ─── OpenEnv required methods ─────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs_vector(), self._get_info()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Parse action
        day_before = self.current_day
        prediction_before = self.prediction_made

        if isinstance(action, int):
            act_type = ACTION_NAMES[action % len(ACTION_NAMES)]
            action_dict: Dict[str, Any] = {"action_type": act_type}
        elif isinstance(action, KavachAction):
            action_dict = action.model_dump()
        elif isinstance(action, dict):
            action_dict = action
        else:
            action_dict = {"action_type": "IGNORE"}

        act_type = action_dict.get("action_type", "IGNORE")
        if act_type not in VALID_ACTIONS:
            act_type = "IGNORE"

        target:  Optional[str]  = action_dict.get("target")
        targets: List[str]      = action_dict.get("targets") or []
        if target and not targets:
            targets = [target]

        cost  = ACTION_COSTS.get(act_type, 0)
        valid = False

        can_act = (
            self.budget_remaining >= cost
            and self.actions_used_today < self.max_actions_per_day
        )

        if can_act:
            if act_type == "IGNORE":
                valid = True

            elif act_type == "FLAG_SUSPICIOUS" and targets:
                self.flagged_entities.add(targets[0])
                valid = True

            elif act_type == "LINK_ENTITIES" and len(targets) >= 2:
                self.linked_pairs.add(tuple(sorted((targets[0], targets[1]))))
                valid = True

            elif act_type == "FREEZE_ENTITY" and targets:
                self.frozen_entities.add(targets[0])
                valid = True

            elif act_type == "PREDICT_ATTACK":
                if not self.prediction_made:
                    self.prediction_made = True
                    self.prediction_day  = self.current_day
                valid = True

            elif act_type == "REQUEST_AUDIT" and targets:
                eid = targets[0]
                if not any(a["entity"] == eid for a in self.pending_audits):
                    self.pending_audits.append({
                        "entity": eid,
                        "resolve_day": self.current_day + 2,
                    })
                valid = True

            elif act_type == "CROSS_VERIFY" and targets:
                eid = targets[0]
                if eid in self.beliefs:
                    b = self.beliefs[eid]
                    b["confidence"] = float(np.clip(b["confidence"] + 0.15, 0.0, 1.0))
                valid = True

            elif act_type == "FLAG_FOR_MONITORING" and targets:
                self.monitored_entities.add(targets[0])
                valid = True

            elif act_type == "DELAY_DECISION":
                valid = True

            if valid:
                self.budget_remaining  -= cost
                self.actions_used_today += 1
                self.action_history.append({
                    "day":     self.current_day,
                    "action":  act_type,
                    "target":  target,
                    "targets": targets,
                })

        # Advance day when limit hit OR IGNORE/DELAY_DECISION OR invalid
        advance = (
            self.actions_used_today >= self.max_actions_per_day
            or act_type in ("IGNORE", "DELAY_DECISION")
            or not valid
        )
        if advance:
            self.current_day += 1
            self.actions_used_today = 0

        terminated = (
            self.current_day > self.total_days
            or self.prediction_made
            or self.budget_remaining <= 0
        )

        reward = self._compute_step_reward(
            act_type=act_type,
            targets=targets,
            valid=valid,
            cost=cost,
            day_before=day_before,
            prediction_before=prediction_before,
        )

        if terminated:
            attack_day = int(
                self.scenario.get("ground_truth", {}).get("attack_starts_day", self.total_days)
            )
            if self.prediction_made:
                if self.prediction_day is not None and self.prediction_day <= attack_day:
                    reward = min(1.0, reward + 0.10)
                else:
                    reward = max(0.0, reward - 0.08)
            elif self.current_day > self.total_days:
                reward = max(0.0, reward - 0.12)

        return self._get_obs_vector(), reward, terminated, False, self._get_info()

    def state(self) -> Dict[str, Any]:
        """Return full current state (OpenEnv spec requirement)."""
        belief_snapshot = {
            eid: {
                "fraud_prob":  round(b["fraud_prob"], 3),
                "confidence":  round(b["confidence"], 3),
                "sightings":   b["sightings"],
                "domains_seen": sorted(b["domains_seen"]),
            }
            for eid, b in self.beliefs.items()
        }
        return {
            "current_day":       self.current_day,
            "total_days":        self.total_days,
            "budget_remaining":  self.budget_remaining,
            "total_budget":      self.total_budget,
            "actions_used_today": self.actions_used_today,
            "max_actions_per_day": self.max_actions_per_day,
            "flagged_entities":  sorted(self.flagged_entities),
            "frozen_entities":   sorted(self.frozen_entities),
            "linked_pairs":      [list(p) for p in self.linked_pairs],
            "monitored_entities": sorted(self.monitored_entities),
            "pending_audits":    self.pending_audits,
            "prediction_made":   self.prediction_made,
            "prediction_day":    self.prediction_day,
            "belief_state":      belief_snapshot,
            "contradiction_flag": self.contradiction_flag,
            "action_history":    self.action_history,
            "scenario_id":       self.scenario["scenario_id"],
            "difficulty":        self.scenario["difficulty"],
            "domains":           self.scenario.get("domains", []),
        }

    def render(self) -> None:
        if self.render_mode == "human":
            mean_p, max_p, _, _ = self._belief_metrics()
            print(
                f"[KAVACH-X] Day {self.current_day}/{self.total_days} | "
                f"Budget {self.budget_remaining}/{self.total_budget} | "
                f"Belief max={max_p:.2f} mean={mean_p:.2f} | "
                f"Flagged:{sorted(self.flagged_entities)} Frozen:{sorted(self.frozen_entities)}"
            )

    def close(self) -> None:
        pass
