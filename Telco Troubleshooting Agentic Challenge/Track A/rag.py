"""RAG module: retrieve similar training cases to guide LLM reasoning.

Builds an in-memory index from train.json at startup (extracted LAYER 0 features).
At inference time, finds the top-K most similar cases and formats them as few-shot
context for the LLM system prompt.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rule_engine import extract_features, DiagnosisResult

logger = logging.getLogger(__name__)

# ── Feature weights for similarity (higher = more discriminative) ─────────────
_WEIGHTS: Dict[str, float] = {
    "min_rsrp":    2.0,
    "max_tilt":    2.0,
    "total_tilt":  1.5,
    "handovers":   1.5,
    "max_speed":   1.0,
    "avg_rb":      1.0,
    "num_neighbors": 1.0,
    "max_cce_fail":  1.5,
    "serving_dl_prb": 1.0,
}

_NORMALIZERS: Dict[str, float] = {
    "min_rsrp":    30.0,
    "max_tilt":    20.0,
    "total_tilt":  50.0,
    "handovers":   5.0,
    "max_speed":   50.0,
    "avg_rb":      100.0,
    "num_neighbors": 5.0,
    "max_cce_fail":  1.0,
    "serving_dl_prb": 50.0,
}

# Boolean signal features also used for similarity
_BOOL_FEATURES = [
    "has_rrc", "has_a2", "has_a5", "has_a3_diff_cells",
    "has_a3_same_cell", "has_ho_attempt",
]
_BOOL_WEIGHT = 3.0  # signaling pattern is the strongest discriminator


def _compute_similarity(f1: Dict, f2: Dict) -> float:
    """Weighted similarity in [0, 1] between two feature dicts."""
    total_w = 0.0
    score = 0.0

    # Numeric features
    for key, weight in _WEIGHTS.items():
        v1 = f1.get(key)
        v2 = f2.get(key)
        if v1 is None or v2 is None:
            continue
        diff = abs(float(v1) - float(v2)) / _NORMALIZERS.get(key, 1.0)
        score += weight * max(0.0, 1.0 - diff)
        total_w += weight

    # Boolean signaling features (exact match)
    for key in _BOOL_FEATURES:
        v1 = f1.get(key)
        v2 = f2.get(key)
        if v1 is None or v2 is None:
            continue
        score += _BOOL_WEIGHT * (1.0 if v1 == v2 else 0.0)
        total_w += _BOOL_WEIGHT

    return score / total_w if total_w > 0 else 0.0


class RagIndex:
    """In-memory index of training cases."""

    def __init__(self, cases: List[Dict]) -> None:
        # Each case: {scenario_id, answer, features, tag}
        self._cases = cases

    def __len__(self) -> int:
        return len(self._cases)

    def find_similar(self, features: Dict, k: int = 3) -> List[Dict]:
        """Return top-k most similar cases, with diversity across answer types."""
        if not features or not self._cases:
            return []

        scored = [
            (c, _compute_similarity(features, c["features"]))
            for c in self._cases
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Diverse selection: cover different answer sets
        result: List[Dict] = []
        seen_answers: set = set()

        for case, sim in scored:
            if len(result) >= k:
                break
            ans = case["answer"]
            if ans not in seen_answers or len(result) < 2:
                result.append({**case, "similarity": round(sim, 3)})
                seen_answers.add(ans)

        return result


def build_rag_index(train_path: str = "data/Phase_1/train.json") -> Optional[RagIndex]:
    """Build RAG index from training data. Returns None if file not found."""
    path = Path(train_path)
    if not path.exists():
        logger.warning(f"[RAG] Training file not found: {train_path}")
        return None

    logger.info(f"[RAG] Building index from {train_path} ...")
    cases: List[Dict] = []

    with open(path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    for s in scenarios:
        answer = s.get("answer", "")
        if not answer:
            continue
        data_raw = s.get("data", {})
        if hasattr(data_raw, "model_dump"):
            data = data_raw.model_dump()
        elif hasattr(data_raw, "dict"):
            data = data_raw.dict()
        else:
            data = data_raw if isinstance(data_raw, dict) else {}

        try:
            features = extract_features(data)
        except Exception:
            features = {}

        cases.append({
            "scenario_id": s.get("scenario_id", ""),
            "answer": answer,
            "tag": s.get("tag", "single-answer"),
            "features": features,
        })

    index = RagIndex(cases)
    logger.info(f"[RAG] Index built: {len(index)} cases")
    return index


def format_rag_context(similar_cases: List[Dict]) -> str:
    """Format similar cases as few-shot context string."""
    if not similar_cases:
        return ""

    lines = ["[Similar Training Cases — use as reference, not as definitive answers]"]
    for i, case in enumerate(similar_cases, 1):
        f = case.get("features", {})
        sig = (
            f"rrc={f.get('has_rrc')}, a2={f.get('has_a2')}, a5={f.get('has_a5')}, "
            f"a3_diff={f.get('has_a3_diff_cells')}, a3_same={f.get('has_a3_same_cell')}, "
            f"ho={f.get('has_ho_attempt')}"
        )
        num = (
            f"rsrp={f.get('min_rsrp')}, tilt_max={f.get('max_tilt')}, "
            f"handovers={f.get('handovers')}, cce_fail={f.get('max_cce_fail')}, "
            f"dl_prb={f.get('serving_dl_prb')}"
        )
        ans = case.get("answer", "?")
        sim = case.get("similarity", 0)
        lines.append(f"  Case {i} (sim={sim}): signal=[{sig}] metrics=[{num}] => Answer: {ans}")

    return "\n".join(lines)
