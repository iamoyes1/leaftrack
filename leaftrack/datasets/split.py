"""Group-aware train / val / test splitting.

Time-lapse frames of the same plant are highly correlated, and the same plant
may also be seen by several cameras. Splitting by frame would leak near-
duplicate images between train and test, so splits are made over *groups*
(one group = one physical plant) and every frame of a group lands in the same
split.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping

SPLITS: tuple[str, str, str] = ("train", "val", "test")


def split_groups(
    groups: Iterable[Hashable],
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    strata: Mapping[Hashable, Hashable] | None = None,
) -> dict[str, list[Hashable]]:
    """Assign whole groups to ``train`` / ``val`` / ``test``.

    The assignment is deterministic for a given ``seed``. When ``strata`` is
    given (e.g. ``group -> camera rig``), each stratum is split separately so
    every split contains groups from every stratum where possible. Every
    split with a non-zero ratio receives at least one group per stratum as
    long as the stratum has enough groups.

    Args:
        groups: unique group identifiers (duplicates are ignored).
        ratios: train / val / test fractions; must be non-negative and sum to 1.
        seed: random seed for the shuffle.
        strata: optional mapping from group to stratum label.

    Returns:
        Mapping from split name to a sorted list of groups.

    Raises:
        ValueError: if the ratios are invalid.
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must be three non-negative numbers summing to 1, got {ratios}")

    unique = sorted(set(groups), key=str)
    by_stratum: dict[Hashable, list[Hashable]] = defaultdict(list)
    for g in unique:
        by_stratum[strata[g] if strata is not None else None].append(g)

    rng = random.Random(seed)
    result: dict[str, list[Hashable]] = {s: [] for s in SPLITS}
    for stratum in sorted(by_stratum, key=str):
        members = by_stratum[stratum]
        rng.shuffle(members)
        counts = _allocate(len(members), ratios)
        start = 0
        for name, count in zip(SPLITS, counts, strict=True):
            result[name].extend(members[start : start + count])
            start += count

    return {name: sorted(members, key=str) for name, members in result.items()}


def _allocate(n: int, ratios: tuple[float, float, float]) -> list[int]:
    """Turn ratios into integer counts that sum to ``n``.

    Non-zero splits get one item each first (when ``n`` allows, val/test
    before train so evaluation is never empty), then the remainder is
    distributed by largest fractional share.
    """
    counts = [0, 0, 0]
    for idx in (1, 2, 0):
        if ratios[idx] > 0 and sum(counts) < n:
            counts[idx] = 1
    remaining = n - sum(counts)
    if remaining > 0:
        ideal = [r * n - c for r, c in zip(ratios, counts, strict=True)]
        base = [max(0, int(v)) for v in ideal]
        while sum(base) > remaining:
            base[base.index(max(base))] -= 1
        for k in range(3):
            counts[k] += base[k]
        remaining -= sum(base)
        order = sorted(range(3), key=lambda k: -(ideal[k] - base[k]))
        for k in order[:remaining]:
            counts[k] += 1
    return counts


def assign_items(
    items: Iterable[str],
    group_of: Mapping[str, Hashable],
    group_split: Mapping[str, Iterable[Hashable]],
) -> dict[str, list[str]]:
    """Expand a group-level split to item level (e.g. image stems).

    Raises:
        KeyError: if an item's group is not present in any split.
    """
    lookup = {g: name for name, gs in group_split.items() for g in gs}
    out: dict[str, list[str]] = {name: [] for name in group_split}
    for item in items:
        out[lookup[group_of[item]]].append(item)
    return {name: sorted(v) for name, v in out.items()}
