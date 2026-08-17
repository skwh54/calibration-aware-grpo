import numpy as np
import pytest

from calibration_aware_grpo.objectives import (
    build_calibration_advantages,
    build_overconfidence_advantages_for_mode,
    build_overconfidence_wrong_advantages,
    compose_advantages,
    compute_calibration_raw_rewards,
    compute_group_relative_advantages,
    resolve_calibration_lambda,
    validate_mode,
)

MAJORITY_MASK = np.array([True, True, True, True, False, False, False])


def test_raw_signed_rewards_correct_majority_only():
    rewards = compute_calibration_raw_rewards(
        correctness=1.0,
        majority_confidence=4 / 7,
        majority_mask=MAJORITY_MASK,
        mode="raw_signed",
    )
    np.testing.assert_allclose(rewards[:4], np.full(4, 3 / 7), rtol=1e-6)
    np.testing.assert_array_equal(rewards[4:], np.zeros(3))


def test_raw_signed_rewards_wrong_majority_are_negative():
    rewards = compute_calibration_raw_rewards(
        correctness=0.0,
        majority_confidence=4 / 7,
        majority_mask=MAJORITY_MASK,
        mode="raw_signed_oc",
    )
    np.testing.assert_allclose(rewards[:4], np.full(4, -4 / 7), rtol=1e-6)
    np.testing.assert_array_equal(rewards[4:], np.zeros(3))


def test_raw_rlcr_matches_cluster_level_formula():
    rewards = compute_calibration_raw_rewards(
        correctness=1.0,
        majority_confidence=4 / 7,
        majority_mask=MAJORITY_MASK,
        mode="raw_rlcr",
    )
    np.testing.assert_allclose(rewards[:4], np.full(4, 40 / 49), rtol=1e-6)


def test_asymmetric_mode_penalizes_correct_majority_minority_samples():
    rewards = compute_calibration_raw_rewards(
        correctness=1.0,
        majority_confidence=4 / 7,
        majority_mask=MAJORITY_MASK,
        mode="raw_signed_asym",
        minority_negative_weight=0.25,
    )
    np.testing.assert_allclose(rewards[:4], np.full(4, 3 / 7), rtol=1e-6)
    np.testing.assert_allclose(rewards[4:], np.full(3, -1 / 7), rtol=1e-6)


def test_non_asymmetric_mode_rejects_minority_weight():
    with pytest.raises(ValueError, match="only valid for asymmetric"):
        compute_calibration_raw_rewards(
            correctness=1.0,
            majority_confidence=0.75,
            majority_mask=[True, True, True, False],
            mode="raw_signed",
            minority_negative_weight=0.25,
        )


def test_group_relative_advantages_are_group_local():
    advantages = compute_group_relative_advantages([1.0, 0.0, 2.0, 2.0], group_size=2)
    np.testing.assert_allclose(advantages[:2], [0.706999, -0.706999], atol=1e-5)
    np.testing.assert_array_equal(advantages[2:], [0.0, 0.0])


def test_group_relative_advantages_reject_nonfinite_rewards():
    with pytest.raises(ValueError, match="finite"):
        compute_group_relative_advantages([1.0, np.nan], group_size=2)


@pytest.mark.parametrize("group_size", [True, 2.0])
def test_objective_group_size_must_be_an_integer(group_size):
    with pytest.raises(TypeError, match="group_size"):
        compute_group_relative_advantages([1.0, 0.0], group_size=group_size)


def test_raw_advantage_clipping_preserves_sign():
    advantages = build_calibration_advantages(
        rewards_calibration=[-2.0, -0.5, 0.5, 2.0],
        group_size=4,
        mode="raw_signed",
        clip=1.0,
    )
    np.testing.assert_array_equal(advantages, [-1.0, -0.5, 0.5, 1.0])


def test_group_norm_uses_group_relative_estimator():
    advantages = build_calibration_advantages(
        rewards_calibration=[1.0, 0.0],
        group_size=2,
        mode="group_norm_signed",
    )
    np.testing.assert_allclose(advantages, [0.706999, -0.706999], atol=1e-5)


def test_overconfidence_penalty_targets_positive_mass_in_wrong_majority():
    advantages = build_overconfidence_wrong_advantages(
        advantages_self_certainty=[3.0, 1.0, -1.0, 0.0, 4.0, 2.0, 1.0],
        majority_correctness=[0.0],
        majority_mask=MAJORITY_MASK,
        group_size=7,
        clip=2.0,
    )
    np.testing.assert_array_equal(advantages, [-2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_overconfidence_rejects_nonfinite_self_certainty(value):
    with pytest.raises(ValueError, match="self_certainty.*finite"):
        build_overconfidence_wrong_advantages(
            advantages_self_certainty=[value, 0.0],
            majority_correctness=[0.0],
            majority_mask=[True, False],
            group_size=2,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_overconfidence_rejects_invalid_correctness(value):
    with pytest.raises(ValueError, match="majority_correctness"):
        build_overconfidence_wrong_advantages(
            advantages_self_certainty=[1.0, 0.0],
            majority_correctness=[value],
            majority_mask=[True, False],
            group_size=2,
        )


def test_overconfidence_accepts_consistent_empty_batch():
    advantages = build_overconfidence_wrong_advantages(
        advantages_self_certainty=[],
        majority_correctness=[],
        majority_mask=np.array([], dtype=bool),
        group_size=2,
    )
    np.testing.assert_array_equal(advantages, np.array([], dtype=np.float32))


@pytest.mark.parametrize(
    "mask",
    [
        [1, 0],
        [False, False],
        [True],
    ],
)
def test_overconfidence_rejects_invalid_mask(mask):
    with pytest.raises((TypeError, ValueError), match="majority_mask"):
        build_overconfidence_wrong_advantages(
            advantages_self_certainty=[1.0, 0.0],
            majority_correctness=[0.0],
            majority_mask=mask,
            group_size=2,
        )


def test_overconfidence_penalty_is_zero_for_correct_majority():
    advantages = build_overconfidence_wrong_advantages(
        advantages_self_certainty=np.ones(7),
        majority_correctness=[1.0],
        majority_mask=MAJORITY_MASK,
        group_size=7,
    )
    np.testing.assert_array_equal(advantages, np.zeros(7))


def test_non_oc_mode_returns_zero_overconfidence_branch():
    advantages = build_overconfidence_advantages_for_mode(
        mode="raw_signed",
        advantages_self_certainty=np.ones(7),
        majority_correctness=[0.0],
        majority_mask=MAJORITY_MASK,
        group_size=7,
    )
    np.testing.assert_array_equal(advantages, np.zeros(7))


def test_compose_advantages_keeps_branches_explicit():
    combined = compose_advantages(
        self_certainty=[1.0, 2.0],
        calibration=[2.0, -2.0],
        overconfidence=[-1.0, 0.0],
        calibration_weight=0.25,
        overconfidence_weight=0.5,
    )
    np.testing.assert_allclose(combined, [1.0, 1.5])


def test_compose_advantages_rejects_float32_overflow():
    with pytest.raises(OverflowError, match="finite float32"):
        compose_advantages(
            self_certainty=[1e39],
            calibration=[0.0],
            overconfidence=[0.0],
            calibration_weight=0.0,
            overconfidence_weight=0.0,
        )


@pytest.mark.parametrize(
    "mask",
    [
        [1, 1, 0, 0],
        [True, np.nan, False, False],
        [],
    ],
)
def test_calibration_rewards_require_nonempty_boolean_mask(mask):
    with pytest.raises((TypeError, ValueError), match="majority_mask"):
        compute_calibration_raw_rewards(
            correctness=1.0,
            majority_confidence=0.5,
            majority_mask=mask,
            mode="raw_signed",
        )


def test_calibration_confidence_must_match_mask_fraction():
    with pytest.raises(ValueError, match="selected mask fraction"):
        compute_calibration_raw_rewards(
            correctness=1.0,
            majority_confidence=0.75,
            majority_mask=[True, True, False, False],
            mode="raw_signed",
        )


def test_validate_mode_lists_supported_modes():
    with pytest.raises(ValueError, match="raw_signed_asym_oc"):
        validate_mode("not-a-mode")


def test_explicit_warmup_steps_take_precedence():
    assert resolve_calibration_lambda(
        current_step=5,
        target_lambda=0.2,
        max_steps=100,
        warmup_fraction=0.5,
        warmup_steps=10,
    ) == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("step", "expected"),
    [(-1, 0.0), (0, 0.0), (5, 0.1), (10, 0.2), (20, 0.2)],
)
def test_fraction_based_calibration_warmup(step, expected):
    assert resolve_calibration_lambda(
        current_step=step,
        target_lambda=0.2,
        max_steps=100,
        warmup_fraction=0.1,
    ) == pytest.approx(expected)


def test_zero_warmup_returns_target_immediately():
    assert (
        resolve_calibration_lambda(
            current_step=0,
            target_lambda=0.2,
            warmup_steps=0,
        )
        == 0.2
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_lambda": -0.1},
        {"target_lambda": np.nan},
        {"target_lambda": 0.1, "warmup_fraction": 1.1},
        {"target_lambda": 0.1, "warmup_steps": -1},
        {"target_lambda": 0.1, "warmup_fraction": 0.1, "max_steps": 0},
    ],
)
def test_calibration_schedule_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        resolve_calibration_lambda(current_step=0, **kwargs)


def test_calibration_schedule_requires_integral_steps():
    with pytest.raises(TypeError, match="current_step"):
        resolve_calibration_lambda(
            current_step=0.5,
            target_lambda=0.1,
            warmup_steps=10,
        )


def test_fraction_warmup_requires_max_steps():
    with pytest.raises(ValueError, match="max_steps is required"):
        resolve_calibration_lambda(
            current_step=1,
            target_lambda=0.1,
            warmup_fraction=0.1,
        )
