from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from calibration_aware_grpo.jax_tunix import (  # noqa: E402
    compute_dual_objective_pg_loss,
    compute_self_certainty_from_logits,
    mean_token_scores,
    prepare_objective_branches,
    prepare_objective_from_policy,
)


def exact_verifier(*, gold_answers, completion_texts):
    def last_boxed(text: str) -> str:
        return text.rsplit(r"\boxed{", 1)[1].split("}", 1)[0]

    return [
        float(last_boxed(text) == gold)
        for gold, text in zip(gold_answers, completion_texts)
    ]


def logits_from_strengths(strengths: list[float], token_count: int = 2):
    rows = []
    for strength in strengths:
        token_logits = [[strength, 0.0, -strength]] * token_count
        rows.append(token_logits)
    return jnp.asarray(rows, dtype=jnp.float32)


def completion_fixture(majority_answer: str = "1") -> list[str]:
    return [
        rf"reasoning {index} \boxed{{{answer}}}"
        for index, answer in enumerate([majority_answer] * 4 + ["2", "3", "4"])
    ]


def test_self_certainty_matches_integrated_trainer_formula():
    logits = jnp.asarray([[[2.0, 0.0, -1.0]]], dtype=jnp.float32)
    actual = compute_self_certainty_from_logits(logits)
    expected = jax.nn.logsumexp(logits, axis=-1) - jnp.mean(logits, axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)


def test_masked_mean_ignores_padding_and_zero_length_rows():
    scores = jnp.asarray([[1.0, 3.0, 99.0], [5.0, 6.0, 7.0]])
    mask = jnp.asarray([[1, 1, 0], [0, 0, 0]])
    actual = mean_token_scores(scores, mask)
    np.testing.assert_allclose(actual, np.asarray([2.0, 0.0]), rtol=0.0, atol=1e-7)


def test_objective_branches_join_policy_and_public_kernel():
    logits = logits_from_strengths([0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6])
    mask = jnp.ones((7, 2), dtype=jnp.int32)
    branches = prepare_objective_branches(
        completion_logits=logits,
        completion_mask=mask,
        completion_texts=completion_fixture("1"),
        gold_answers=["1"],
        group_size=7,
        verifier=exact_verifier,
        mode="raw_signed",
        calibration_lambda=0.25,
    )

    assert branches.calibration_batch.groups[0].correctness == 1.0
    assert branches.calibration_batch.groups[0].confidence == pytest.approx(4 / 7)
    np.testing.assert_array_equal(
        branches.calibration_batch.majority_mask,
        np.asarray([True, True, True, True, False, False, False]),
    )
    expected_combined = np.asarray(
        branches.advantages_self_certainty
    ) + 0.25 * np.asarray(branches.advantages_calibration)
    np.testing.assert_allclose(
        branches.combined_advantages,
        expected_combined,
        rtol=0.0,
        atol=1e-6,
    )
    assert set(branches.as_train_example_fields()) == {
        "advantages",
        "advantages_sc",
        "advantages_cal",
        "advantages_oc",
        "calibration_lambda",
        "overconfidence_lambda",
    }


def test_wrong_majority_enables_overconfidence_branch():
    logits = logits_from_strengths([2.0, 1.8, 1.6, 1.4, 0.1, 0.2, 0.3])
    branches = prepare_objective_branches(
        completion_logits=logits,
        completion_mask=jnp.ones((7, 2), dtype=jnp.int32),
        completion_texts=completion_fixture("9"),
        gold_answers=["1"],
        group_size=7,
        verifier=exact_verifier,
        mode="raw_signed_oc",
        calibration_lambda=0.25,
        overconfidence_lambda=0.5,
        overconfidence_clip=2.0,
    )

    oc = np.asarray(branches.advantages_overconfidence)
    assert np.any(oc[:4] < 0)
    np.testing.assert_array_equal(oc[4:], np.zeros(3, dtype=np.float32))
    assert branches.calibration_batch.groups[0].correctness == 0.0


def test_policy_callback_is_teacher_forced_boundary():
    calls = []
    logits = logits_from_strengths([0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6])

    def recompute_policy(**kwargs):
        calls.append(kwargs)
        return logits

    branches = prepare_objective_from_policy(
        recompute_policy=recompute_policy,
        prompt_tokens=jnp.ones((7, 3), dtype=jnp.int32),
        prompt_mask=jnp.ones((7, 3), dtype=jnp.int32),
        completion_tokens=jnp.ones((7, 2), dtype=jnp.int32),
        completion_mask=jnp.ones((7, 2), dtype=jnp.int32),
        completion_texts=completion_fixture("1"),
        gold_answers=["1"],
        group_size=7,
        verifier=exact_verifier,
        mode="raw_signed",
        calibration_lambda=0.25,
    )

    assert len(calls) == 1
    assert calls[0]["completion_tokens"].shape == (7, 2)
    assert branches.combined_advantages.shape == (7,)


def test_dual_objective_loss_matches_branchwise_clipped_formula():
    logits = logits_from_strengths([0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6])
    mask = jnp.asarray([[1, 1], [1, 1], [1, 1], [1, 0], [1, 1], [1, 1], [1, 1]])
    branches = prepare_objective_branches(
        completion_logits=logits,
        completion_mask=mask,
        completion_texts=completion_fixture("1"),
        gold_answers=["1"],
        group_size=7,
        verifier=exact_verifier,
        mode="raw_signed",
        calibration_lambda=0.25,
    )
    coef_1 = jnp.full((7, 2), 1.1, dtype=jnp.float32)
    coef_2 = jnp.full((7, 2), 0.9, dtype=jnp.float32)
    output = compute_dual_objective_pg_loss(
        coef_1=coef_1,
        coef_2=coef_2,
        completion_mask=mask,
        branches=branches,
    )

    sc = np.asarray(branches.advantages_self_certainty)[:, None]
    cal = np.asarray(branches.advantages_calibration)[:, None]
    expected_per_token = -np.minimum(
        np.asarray(coef_1) * sc, np.asarray(coef_2) * sc
    ) - 0.25 * np.minimum(np.asarray(coef_1) * cal, np.asarray(coef_2) * cal)
    expected_loss = np.sum(expected_per_token * np.asarray(mask)) / np.sum(mask)
    np.testing.assert_allclose(output.per_token_loss, expected_per_token, atol=1e-6)
    np.testing.assert_allclose(output.loss, expected_loss, atol=1e-6)


def test_shape_mismatches_fail_closed():
    with pytest.raises(ValueError, match="completion_logits must have shape"):
        compute_self_certainty_from_logits(jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="identical shapes"):
        mean_token_scores(jnp.ones((2, 3)), jnp.ones((2, 2)))
