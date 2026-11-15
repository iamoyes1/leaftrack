"""Tests for group-aware splitting (no plant may appear in two splits)."""

import pytest

from leaftrack.datasets.split import SPLITS, _allocate, assign_items, split_groups

KOMATSUNA_LIKE = {f"multi_p{i:02d}": "multi" for i in range(5)} | {
    f"rgbd_p{i:02d}": "rgbd" for i in range(5)
}


def test_every_group_assigned_exactly_once() -> None:
    result = split_groups(KOMATSUNA_LIKE, seed=1, strata=KOMATSUNA_LIKE)
    assigned = [g for name in SPLITS for g in result[name]]
    assert sorted(assigned) == sorted(KOMATSUNA_LIKE)
    assert len(assigned) == len(set(assigned))


def test_split_is_deterministic_and_seed_dependent() -> None:
    a = split_groups(KOMATSUNA_LIKE, seed=7, strata=KOMATSUNA_LIKE)
    b = split_groups(KOMATSUNA_LIKE, seed=7, strata=KOMATSUNA_LIKE)
    assert a == b
    others = [split_groups(KOMATSUNA_LIKE, seed=s, strata=KOMATSUNA_LIKE) for s in range(8, 16)]
    assert any(o != a for o in others)


def test_stratification_puts_every_rig_in_every_split() -> None:
    result = split_groups(KOMATSUNA_LIKE, ratios=(0.6, 0.2, 0.2), seed=0, strata=KOMATSUNA_LIKE)
    for name in SPLITS:
        assert {KOMATSUNA_LIKE[g] for g in result[name]} == {"multi", "rgbd"}
    assert [len(result[n]) for n in SPLITS] == [6, 2, 2]


@pytest.mark.parametrize("ratios", [(0.5, 0.5, 0.5), (1.2, -0.1, -0.1), (0.5, 0.5)])
def test_invalid_ratios_raise(ratios: tuple) -> None:
    with pytest.raises(ValueError):
        split_groups(["a", "b"], ratios=ratios)


@pytest.mark.parametrize("n", range(1, 25))
def test_allocate_sums_to_n_and_fills_eval_splits_first(n: int) -> None:
    counts = _allocate(n, (0.7, 0.15, 0.15))
    assert sum(counts) == n
    if n >= 3:
        assert all(c >= 1 for c in counts)


def test_assign_items_keeps_all_frames_of_a_plant_together() -> None:
    # 3 cameras x 4 frames per plant -> 12 items per group
    group_of = {f"{g}_c{c}_t{t}": g for g in KOMATSUNA_LIKE for c in range(3) for t in range(4)}
    group_split = split_groups(KOMATSUNA_LIKE, seed=3, strata=KOMATSUNA_LIKE)
    items = assign_items(group_of, group_of, group_split)
    assert sum(len(v) for v in items.values()) == len(group_of)
    for name in SPLITS:
        groups_here = {group_of[i] for i in items[name]}
        assert groups_here == set(group_split[name])
    seen = [group_of[i] for name in SPLITS for i in items[name]]
    for g in KOMATSUNA_LIKE:
        splits_with_g = {n for n in SPLITS if any(group_of[i] == g for i in items[n])}
        assert len(splits_with_g) == 1, f"{g} leaked across splits"
    assert len(seen) == len(group_of)


def test_assign_items_raises_for_unknown_group() -> None:
    with pytest.raises(KeyError):
        assign_items(["x"], {"x": "missing"}, {"train": ["a"], "val": [], "test": []})
