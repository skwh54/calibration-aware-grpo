import numpy as np
import pytest

from calibration_aware_grpo.majority import build_calibration_batch


def _boxed(values):
    return [rf"reasoning \boxed{{{value}}}" for value in values]


def test_batches_verification_across_groups_and_builds_rewards():
    completions = _boxed([1, 1, 1, 1, 2, 3, 4]) + _boxed([2, 2, 2, 2, 1, 3, 4])
    calls = []

    def verifier(*, gold_answers, completion_texts):
        calls.append((list(gold_answers), list(completion_texts)))
        return [1.0, 0.0]

    batch = build_calibration_batch(
        completions=completions,
        gold_answers=["1", "1"],
        group_size=7,
        verifier=verifier,
        mode="raw_signed",
    )

    assert len(calls) == 1
    assert calls[0][0] == ["1", "1"]
    assert calls[0][1] == [
        r"reasoning \boxed{1}",
        r"reasoning \boxed{2}",
    ]
    assert batch.group_count == 2
    np.testing.assert_allclose(batch.calibration_rewards[:4], np.full(4, 3 / 7))
    np.testing.assert_array_equal(batch.calibration_rewards[4:7], np.zeros(3))
    np.testing.assert_allclose(batch.calibration_rewards[7:11], np.full(4, -4 / 7))
    np.testing.assert_array_equal(batch.calibration_rewards[11:], np.zeros(3))
    assert batch.groups[0].correctness == 1.0
    assert batch.groups[1].correctness == 0.0
    assert batch.groups[0].brier_score == pytest.approx((4 / 7 - 1) ** 2)
    assert batch.groups[1].brier_score == pytest.approx((4 / 7) ** 2)


def test_numeric_equivalence_forms_one_majority_cluster():
    completions = [
        r"\boxed{0.5}",
        r"\boxed{1/2}",
        r"\boxed{\frac{2}{4}}",
        r"\boxed{3/6}",
        r"\boxed{1}",
        r"\boxed{2}",
        r"\boxed{3}",
    ]
    batch = build_calibration_batch(
        completions=completions,
        gold_answers=["1/2"],
        group_size=7,
        verifier=lambda **_: [1.0],
    )
    assert batch.groups[0].answer_key == "num::1/2"
    assert batch.groups[0].cluster_size == 4
    assert batch.groups[0].confidence == pytest.approx(4 / 7)


def test_unique_plurality_below_half_is_still_selected():
    batch = build_calibration_batch(
        completions=_boxed([1, 1, 1, 2, 2, 3, 4]),
        gold_answers=["1"],
        group_size=7,
        verifier=lambda **_: [1.0],
    )
    assert batch.groups[0].answer_key == "num::1"
    assert batch.groups[0].cluster_size == 3
    assert batch.groups[0].confidence == pytest.approx(3 / 7)


def test_tied_majority_skips_verifier():
    called = False

    def verifier(**_):
        nonlocal called
        called = True
        return []

    batch = build_calibration_batch(
        completions=_boxed([1, 1, 2, 2]),
        gold_answers=["1"],
        group_size=4,
        verifier=verifier,
    )
    assert not called
    assert batch.groups[0].skip_reason == "tied_majority"
    np.testing.assert_array_equal(batch.calibration_rewards, np.zeros(4))
    np.testing.assert_array_equal(batch.majority_mask, np.zeros(4, dtype=bool))


def test_below_confidence_threshold_skips_verifier_but_keeps_diagnostics():
    called = False

    def verifier(**_):
        nonlocal called
        called = True
        return []

    batch = build_calibration_batch(
        completions=_boxed([1, 1, 2, 3]),
        gold_answers=["1"],
        group_size=4,
        verifier=verifier,
        min_majority_confidence=0.75,
    )
    assert not called
    result = batch.groups[0]
    assert result.skip_reason == "below_min_majority_confidence"
    assert result.confidence == 0.5
    assert result.correctness is None
    np.testing.assert_array_equal(batch.majority_mask, [True, True, False, False])


def test_unparsable_outputs_do_not_create_false_majority():
    batch = build_calibration_batch(
        completions=["no answer", "no answer", "unknown", "unknown"],
        gold_answers=["1"],
        group_size=4,
        verifier=lambda **_: pytest.fail("verifier should not run"),
    )
    assert batch.groups[0].skip_reason == "tied_majority"


@pytest.mark.parametrize(
    ("scores", "received"),
    [([], 0), ([1.0, 1.0], 2)],
)
def test_rejects_verifier_cardinality_mismatch(scores, received):
    with pytest.raises(
        RuntimeError,
        match=rf"verifier returned an unexpected number of scores: expected=1, got={received}",
    ):
        build_calibration_batch(
            completions=_boxed([1, 1, 1, 1, 2, 3, 4]),
            gold_answers=["1"],
            group_size=7,
            verifier=lambda **_: scores,
        )


def test_rejects_non_probability_verifier_score():
    with pytest.raises(ValueError, match="must be finite and in"):
        build_calibration_batch(
            completions=_boxed([1, 1, 1, 1, 2, 3, 4]),
            gold_answers=["1"],
            group_size=7,
            verifier=lambda **_: [np.nan],
        )


def test_asymmetric_mode_changes_only_correct_majority_minority():
    batch = build_calibration_batch(
        completions=_boxed([1, 1, 1, 1, 2, 3, 4]),
        gold_answers=["1"],
        group_size=7,
        verifier=lambda **_: [1.0],
        mode="raw_signed_asym",
        minority_negative_weight=0.25,
    )
    np.testing.assert_allclose(batch.calibration_rewards[4:], np.full(3, -1 / 7))


def test_requires_one_gold_answer_per_group():
    with pytest.raises(ValueError, match="one answer per group"):
        build_calibration_batch(
            completions=_boxed([1, 1, 2, 2]),
            gold_answers=[],
            group_size=4,
            verifier=lambda **_: [],
        )


@pytest.mark.parametrize("group_size", [True, 4.0])
def test_group_size_must_be_an_integer(group_size):
    with pytest.raises(TypeError, match="group_size"):
        build_calibration_batch(
            completions=_boxed([1, 1, 2, 2]),
            gold_answers=["1"],
            group_size=group_size,
            verifier=lambda **_: [],
        )


def test_invalid_minority_weight_fails_before_group_skips():
    with pytest.raises(ValueError, match="only valid for asymmetric"):
        build_calibration_batch(
            completions=_boxed([1, 1, 2, 2]),
            gold_answers=["1"],
            group_size=4,
            verifier=lambda **_: [],
            mode="raw_signed",
            minority_negative_weight=0.25,
        )


def test_verifier_order_ignores_interleaved_skipped_group():
    completions = _boxed([1, 1, 1, 2]) + _boxed([3, 3, 4, 4]) + _boxed([2, 2, 2, 5])
    observed = {}

    def verifier(*, gold_answers, completion_texts):
        observed["gold_answers"] = list(gold_answers)
        observed["completion_texts"] = list(completion_texts)
        return [1.0, 1.0]

    batch = build_calibration_batch(
        completions=completions,
        gold_answers=["1", "3", "2"],
        group_size=4,
        verifier=verifier,
    )
    assert observed == {
        "gold_answers": ["1", "2"],
        "completion_texts": [
            r"reasoning \boxed{1}",
            r"reasoning \boxed{2}",
        ],
    }
    assert [group.skip_reason for group in batch.groups] == [
        None,
        "tied_majority",
        None,
    ]


def test_empty_batch_is_valid_and_does_not_call_verifier():
    batch = build_calibration_batch(
        completions=[],
        gold_answers=[],
        group_size=7,
        verifier=lambda **_: pytest.fail("verifier should not run"),
    )
    assert batch.group_count == 0
    np.testing.assert_array_equal(batch.calibration_rewards, np.array([]))
    np.testing.assert_array_equal(batch.majority_mask, np.array([], dtype=bool))
