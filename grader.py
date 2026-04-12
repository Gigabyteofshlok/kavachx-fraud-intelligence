"""
KAVACH-X Grader — Enhanced v2.0
================================
8-component deterministic scoring for all 3 tasks.

Components:
  1. Prediction Timing          — Early PREDICT_ATTACK scores highest
  2. Entity Detection           — Fraud entities flagged/frozen
  3. Entity Linking             — Cross-domain entity links
  4. Belief Calibration         — Avoided overconfident wrong decisions
  5. Information Efficiency     — Wise use of REQUEST_AUDIT actions
  6. Contradiction Handling     — Correctly navigated contradictory signals
  7. Budget Efficiency          — Completed task under budget
  8. Decoy Penalty              — Flat deduction per decoy frozen

Weights differ by difficulty:
  Easy:   timing 50%, detection 40%, budget 10%
  Medium: timing 40%, linking 40%, detection 10%, budget 10% − decoy
  Hard:   timing 32%, linking 28%, detection 12%, belief 8%,
          info 5%, contradiction 5%, budget 10% − decoy

All final scores bounded [0.0, 1.0].
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple


class KavachXGrader:
    """
    Grades a completed KAVACH-X episode deterministically.

    Usage:
        grader = KavachXGrader(scenario)
        result = grader.grade(action_history, prediction_day)
        print(result["final_score"])  # float in [0.0, 1.0]
    """

    def __init__(self, scenario: Dict[str, Any]) -> None:
        self.scenario       = scenario
        self.ground_truth   = scenario["ground_truth"]
        self.grader_config  = scenario["grader"]
        self.entities       = scenario.get("entities", {})
        self.difficulty     = scenario.get("difficulty", "easy")

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _digits(token: Any) -> int:
        digits = "".join(ch for ch in str(token) if ch.isdigit())
        return int(digits) if digits else 0

    def _parse_history(
        self, history: List[Dict]
    ) -> Tuple[Set[str], Set[str], Set[Tuple], int, int, int, Optional[int]]:
        """
        Returns: flagged, frozen, linked, budget_used, audits_used,
                 monitor_count, prediction_day
        """
        flagged: Set[str] = set()
        frozen:  Set[str] = set()
        linked:  Set[Tuple[str, str]] = set()
        budget_used   = 0
        audits_used   = 0
        monitor_count = 0
        prediction_day: Optional[int] = None

        cost_map = {
            "FLAG_SUSPICIOUS": 1, "LINK_ENTITIES": 2, "FREEZE_ENTITY": 3,
            "PREDICT_ATTACK": 4,  "REQUEST_AUDIT": 2, "CROSS_VERIFY": 2,
            "DELAY_DECISION": 1,  "FLAG_FOR_MONITORING": 1, "IGNORE": 0,
        }

        for act in history:
            atype   = act.get("action", act.get("action_type", "IGNORE"))
            targets = list(act.get("targets") or [])
            target  = act.get("target")
            if target and target not in targets:
                targets = [target] + targets

            budget_used += cost_map.get(atype, 0)

            if atype == "FLAG_SUSPICIOUS":
                flagged.update(targets)
            elif atype == "FREEZE_ENTITY":
                frozen.update(targets)
            elif atype == "LINK_ENTITIES" and len(targets) >= 2:
                linked.add(tuple(sorted((targets[0], targets[1]))))
            elif atype == "PREDICT_ATTACK":
                if prediction_day is None:
                    prediction_day = act.get("day")
            elif atype in ("REQUEST_AUDIT", "CROSS_VERIFY"):
                audits_used += 1
            elif atype == "FLAG_FOR_MONITORING":
                monitor_count += 1

        return flagged, frozen, linked, budget_used, audits_used, monitor_count, prediction_day

    # ─── Scoring components ───────────────────────────────────────────────────

    def _score_timing(self, prediction_day: Optional[int]) -> float:
        if prediction_day is None:
            return float(self.grader_config["prediction_timing_score"].get("no_prediction", 0.0))
        for key, val in self.grader_config["prediction_timing_score"].items():
            if not key.startswith("day_"):
                continue
            body = key[4:]
            if "_to_" in body:
                lo, hi = body.split("_to_")
                if self._digits(lo) <= prediction_day <= self._digits(hi):
                    return float(val)
            else:
                if prediction_day == self._digits(body):
                    return float(val)
        return 0.0

    def _score_detection(
        self, flagged: Set[str], frozen: Set[str], fraud: Set[str]
    ) -> float:
        detected = (flagged | frozen) & fraud
        if not fraud:
            return 1.0
        ratio = len(detected) / len(fraud)
        cfg = self.grader_config.get("entity_detection_score", {})
        if ratio >= 1.0:
            return 1.0
        elif ratio >= 0.5:
            return float(cfg.get("one_entity_flagged", 0.5))
        elif ratio > 0.0:
            return float(cfg.get("partial_detection", 0.25))
        return 0.0

    def _score_linking(
        self, linked: Set[Tuple], fraud: Set[str]
    ) -> float:
        cfg = self.grader_config.get("entity_linking_score")
        if not cfg:
            return 0.0
        correct = [p for p in linked if p[0] in fraud and p[1] in fraud]
        domains: Set[str] = set()
        for p in correct:
            for eid in p:
                d = self.entities.get(eid, {}).get("domain")
                if d:
                    domains.add(d)
        n = len(domains)
        if n >= 3:
            return float(cfg.get("all_3_domains_linked", 1.0))
        elif n == 2:
            return float(cfg.get("2_domains_linked", cfg.get("both_domains_linked", 0.6)))
        elif n == 1:
            return float(cfg.get("1_domain_only", cfg.get("one_domain_only", 0.2)))
        return 0.0

    def _score_belief_calibration(
        self, frozen: Set[str], decoys: Set[str]
    ) -> float:
        cfg = self.grader_config.get("belief_calibration", {})
        if not cfg:
            return 0.0
        if decoys and not (frozen & decoys):
            return float(cfg.get("well_calibrated_bonus", 0.0))
        if frozen & decoys:
            return float(cfg.get("overconfident_wrong_penalty", 0.0))
        return 0.0

    def _score_info_efficiency(self, audits_used: int) -> float:
        cfg = self.grader_config.get("information_efficiency_bonus", {})
        for key, val in cfg.items():
            if not key.startswith("used_"):
                continue
            body = key[5:]
            if "or_less" in body:
                if audits_used <= self._digits(body):
                    return float(val)
            elif "_to_" in body:
                lo, hi = body.split("_to_")
                if self._digits(lo) <= audits_used <= self._digits(hi):
                    return float(val)
            else:
                if audits_used == self._digits(body):
                    return float(val)
        return 0.0

    def _score_contradiction(
        self, frozen: Set[str], decoys: Set[str]
    ) -> float:
        has_contradiction = any(
            e.get("visibility") == "contradictory"
            for e in self.scenario.get("events", [])
        )
        if has_contradiction and not (frozen & decoys):
            return float(self.grader_config.get("contradiction_handling_bonus", 0.0))
        return 0.0

    def _score_budget(self, budget_used: int) -> float:
        cfg = self.grader_config.get("budget_efficiency_bonus", {})
        for key, val in cfg.items():
            if not key.startswith("used_"):
                continue
            body = key[5:]
            if "or_less" in body:
                if budget_used <= self._digits(body):
                    return float(val)
            elif "_to_" in body:
                lo, hi = body.split("_to_")
                if self._digits(lo) <= budget_used <= self._digits(hi):
                    return float(val)
            else:
                if budget_used == self._digits(body):
                    return float(val)
        return 0.0

    # ─── Final grade ─────────────────────────────────────────────────────────

    def grade(
        self,
        action_history: List[Dict],
        prediction_made_day: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Grade a completed episode. Returns final_score in [0.0, 1.0].
        """
        fraud  = set(self.ground_truth["fraud_ring_entities"])
        decoys = set(self.ground_truth.get("decoy_entities", []))

        flagged, frozen, linked, budget_used, audits_used, monitor_count, pred_day = \
            self._parse_history(action_history)

        if prediction_made_day is not None and pred_day is None:
            pred_day = prediction_made_day

        # ── All 8 components ─────────────────────────────────────────────────
        t_score  = self._score_timing(pred_day)
        d_score  = self._score_detection(flagged, frozen, fraud)
        l_score  = self._score_linking(linked, fraud)
        b_score  = self._score_belief_calibration(frozen, decoys)
        i_score  = self._score_info_efficiency(audits_used)
        c_score  = self._score_contradiction(frozen, decoys)
        bu_score = self._score_budget(budget_used)

        decoy_count   = len(frozen & decoys)
        penalty_unit  = abs(float(self.grader_config.get("decoy_penalty", 0.0)))
        decoy_penalty = penalty_unit * decoy_count

        # ── Difficulty-weighted sum ───────────────────────────────────────────
        if self.difficulty == "easy":
            raw = (
                0.50 * t_score
                + 0.40 * d_score
                + 0.10 * bu_score
            )
        elif self.difficulty == "medium":
            raw = (
                0.40 * t_score
                + 0.40 * l_score
                + 0.10 * d_score
                + 0.10 * bu_score
                - decoy_penalty
            )
        else:  # hard — full 8-component
            raw = (
                0.32 * t_score
                + 0.28 * l_score
                + 0.12 * d_score
                + 0.08 * b_score
                + 0.05 * i_score
                + 0.05 * c_score
                + 0.10 * bu_score
                - decoy_penalty
            )

        final_score = round(float(max(0.0, min(1.0, raw))), 4)

        timing_score = t_score
        linking_score = l_score
        detection_score = d_score
        belief_score = b_score
        info_bonus = i_score
        contradiction_bonus = c_score
        budget_bonus = bu_score
        prediction_made_day = pred_day
        decoy_entities = decoys

        score_breakdown_readable = (
            f"KAVACH-X Score Breakdown\n"
            f"========================\n"
            f"Prediction Timing  (weight 0.32): {timing_score:.3f}  → contribution {0.32 * timing_score:.3f}\n"
            f"Entity Linking     (weight 0.28): {linking_score:.3f}  → contribution {0.28 * linking_score:.3f}\n"
            f"Entity Detection   (weight 0.12): {detection_score:.3f}  → contribution {0.12 * detection_score:.3f}\n"
            f"Belief Calibration (weight 0.08): {belief_score:.3f}  → contribution {0.08 * belief_score:.3f}\n"
            f"Info Efficiency    (weight 0.05): {info_bonus:.3f}  → contribution {0.05 * info_bonus:.3f}\n"
            f"Contradiction Hdl  (weight 0.05): {contradiction_bonus:.3f}  → contribution {0.05 * contradiction_bonus:.3f}\n"
            f"Budget Efficiency  (weight 0.10): {budget_bonus:.3f}  → contribution {0.10 * budget_bonus:.3f}\n"
            f"Decoy Penalty               :  -{decoy_penalty:.3f}\n"
            f"------------------------\n"
            f"FINAL SCORE: {max(0.0, min(1.0, raw)):.4f}\n"
            f"Budget used: {budget_used} | Audits used: {audits_used} | "
            f"Prediction day: {prediction_made_day} | Decoys frozen: {len(frozen & decoy_entities)}"
        )

        return {
            "final_score": final_score,
            "components": {
                "prediction_timing":        round(t_score,  4),
                "entity_detection":         round(d_score,  4),
                "entity_linking":           round(l_score,  4),
                "belief_calibration":       round(b_score,  4),
                "information_efficiency":   round(i_score,  4),
                "contradiction_handling":   round(c_score,  4),
                "budget_efficiency_bonus":  round(bu_score, 4),
                "decoy_penalty":            round(-decoy_penalty, 4),
            },
            "stats": {
                "budget_used":              budget_used,
                "prediction_day":           pred_day,
                "fraud_entities_flagged":   len((flagged | frozen) & fraud),
                "fraud_entities_total":     len(fraud),
                "decoys_frozen":            decoy_count,
                "links_made":               len(linked),
                "audits_used":              audits_used,
                "difficulty":               self.difficulty,
            },
            "score_breakdown_readable": score_breakdown_readable,
        }
