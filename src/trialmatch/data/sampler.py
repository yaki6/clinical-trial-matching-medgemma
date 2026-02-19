"""Stratified sampling of criterion annotations for Phase 0.

Samples from CriterionAnnotation list, stratified by expert_label
(MET / NOT_MET / UNKNOWN), with deterministic seed.
"""

from __future__ import annotations

import random

from trialmatch.models.schema import CriterionAnnotation, CriterionVerdict, Phase0Sample


def stratified_sample(
    annotations: list[CriterionAnnotation],
    n_pairs: int,
    seed: int = 42,
    distribution: dict[CriterionVerdict, float] | None = None,
) -> Phase0Sample:
    """Sample n_pairs with stratified label distribution.

    Default distribution: 40% MET, 40% NOT_MET, 20% UNKNOWN.
    """
    if distribution is None:
        distribution = {
            CriterionVerdict.MET: 0.4,
            CriterionVerdict.NOT_MET: 0.4,
            CriterionVerdict.UNKNOWN: 0.2,
        }

    buckets: dict[CriterionVerdict, list[CriterionAnnotation]] = {}
    for a in annotations:
        buckets.setdefault(a.expert_label, []).append(a)

    rng = random.Random(seed)
    selected: list[CriterionAnnotation] = []

    for label, ratio in sorted(distribution.items(), key=lambda x: x[0].value):
        pool = buckets.get(label, [])
        n_target = round(n_pairs * ratio)
        n_take = min(n_target, len(pool))
        if n_take > 0:
            selected.extend(rng.sample(pool, n_take))

    # If rounding left us short, fill from largest pool
    while len(selected) < n_pairs:
        remaining = [a for a in annotations if a not in selected]
        if not remaining:
            break
        selected.append(rng.choice(remaining))

    return Phase0Sample(pairs=selected[:n_pairs])
