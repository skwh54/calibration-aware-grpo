"""JAX helpers for calibration-aware GRPO objectives.

Teacher-forced policy logits enter through a callback. This module computes
self-certainty, assembles the grouped objective branches, and evaluates the
clipped policy-gradient loss. Model construction and checkpoint management
remain the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .majority import CalibrationBatch, Verifier, build_calibration_batch
from .objectives import (
    build_calibration_advantages,
    build_overconfidence_advantages_for_mode,
    compute_group_relative_advantages,
)

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised by import-error users
    raise ImportError(
        "calibration_aware_grpo.jax_tunix requires the 'jax' optional extra"
    ) from exc

Array = Any


class PolicyRecompute(Protocol):
    """Teacher-forced actor callback used after rollout generation.

    Implementations should evaluate the same policy weights that generated the
    completions and return completion-token logits with shape
    ``[samples, completion_tokens, vocabulary]``.
    """

    def __call__(
        self,
        *,
        prompt_tokens: Array,
        prompt_mask: Array,
        completion_tokens: Array,
        completion_mask: Array,
    ) -> Array: ...


@dataclass(frozen=True)
class JaxObjectiveBranches:
    """Per-sample rewards and advantages for a policy update."""

    rewards_self_certainty: Array
    rewards_calibration: Array
    advantages_self_certainty: Array
    advantages_calibration: Array
    advantages_overconfidence: Array
    combined_rewards: Array
    combined_advantages: Array
    calibration_lambda: Array
    overconfidence_lambda: Array
    calibration_batch: CalibrationBatch

    def as_train_example_fields(self) -> dict[str, Array]:
        """Return fields suitable for a Tunix ``TrainExample``."""

        return {
            "advantages": self.combined_advantages,
            "advantages_sc": self.advantages_self_certainty,
            "advantages_cal": self.advantages_calibration,
            "advantages_oc": self.advantages_overconfidence,
            "calibration_lambda": self.calibration_lambda,
            "overconfidence_lambda": self.overconfidence_lambda,
        }


@dataclass(frozen=True)
class DualObjectiveLoss:
    """Outputs from the calibration-aware policy loss."""

    loss: Array
    per_token_loss: Array
    pg_clipfrac_self_certainty: Array
    pg_clipfrac_calibration: Array
    pg_clipfrac_overconfidence: Array


def compute_self_certainty_from_logits(completion_logits: Array) -> Array:
    """Compute the Intuitor self-certainty score for every completion token."""

    logits = jnp.asarray(completion_logits, dtype=jnp.float32)
    if logits.ndim != 3:
        raise ValueError(
            "completion_logits must have shape [samples, tokens, vocabulary]; "
            f"got {logits.shape}"
        )
    if logits.shape[-1] < 1:
        raise ValueError("completion_logits vocabulary axis must be non-empty")
    return jax.nn.logsumexp(logits, axis=-1) - jnp.mean(logits, axis=-1)


def mean_token_scores(token_scores: Array, completion_mask: Array) -> Array:
    """Average token scores over the non-padding completion positions."""

    scores = jnp.asarray(token_scores, dtype=jnp.float32)
    mask = jnp.asarray(completion_mask, dtype=jnp.float32)
    if scores.shape != mask.shape:
        raise ValueError(
            "token_scores and completion_mask must have identical shapes; "
            f"got {scores.shape} and {mask.shape}"
        )
    if scores.ndim != 2:
        raise ValueError(
            f"token_scores must have shape [samples, tokens]; got {scores.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(scores))):
        raise ValueError("token_scores must contain only finite values")
    if not bool(jnp.all((mask == 0) | (mask == 1))):
        raise ValueError("completion_mask must contain only 0/1 values")

    lengths = jnp.sum(mask, axis=-1)
    safe_lengths = jnp.maximum(lengths, 1.0)
    sequence_scores = jnp.sum(scores * mask, axis=-1) / safe_lengths
    return jnp.where(lengths > 0, sequence_scores, 0.0)


def _group_correctness(batch: CalibrationBatch) -> np.ndarray:
    return np.asarray(
        [
            0.0 if group.correctness is None else group.correctness
            for group in batch.groups
        ],
        dtype=np.float32,
    )


def prepare_objective_branches(
    *,
    completion_logits: Array,
    completion_mask: Array,
    completion_texts: Sequence[str],
    gold_answers: Sequence[str],
    group_size: int,
    verifier: Verifier,
    mode: str,
    calibration_lambda: float,
    overconfidence_lambda: float = 0.0,
    min_majority_confidence: float = 0.0,
    calibration_clip: float | None = None,
    overconfidence_clip: float = 2.0,
    minority_negative_weight: float = 0.0,
) -> JaxObjectiveBranches:
    """Build the three reward/advantage branches used by the TPU learner.

    ``completion_logits`` must come from teacher-forced recomputation under the
    rollout policy. The verifier receives only one representative completion per
    eligible unique-plurality cluster through the NumPy implementation.
    """

    if not np.isfinite(calibration_lambda) or calibration_lambda < 0:
        raise ValueError("calibration_lambda must be finite and non-negative")
    if not np.isfinite(overconfidence_lambda) or overconfidence_lambda < 0:
        raise ValueError("overconfidence_lambda must be finite and non-negative")

    token_self_certainty = compute_self_certainty_from_logits(completion_logits)
    sequence_self_certainty = mean_token_scores(
        token_self_certainty,
        completion_mask,
    )
    sample_count = int(sequence_self_certainty.shape[0])
    if len(completion_texts) != sample_count:
        raise ValueError(
            "completion_texts must match the recomputed policy batch; "
            f"got {len(completion_texts)} and {sample_count}"
        )
    expected_groups = sample_count // group_size if group_size > 0 else 0
    if group_size < 2 or sample_count % group_size != 0:
        raise ValueError(
            "sample count must be divisible by group_size >= 2; "
            f"got {sample_count} and {group_size}"
        )
    if len(gold_answers) != expected_groups:
        raise ValueError(
            "gold_answers must contain one answer per rollout group; "
            f"got {len(gold_answers)} and {expected_groups}"
        )

    calibration_batch = build_calibration_batch(
        completions=list(completion_texts),
        gold_answers=list(gold_answers),
        group_size=group_size,
        verifier=verifier,
        min_majority_confidence=min_majority_confidence,
        mode=mode,
        minority_negative_weight=minority_negative_weight,
    )
    rewards_self_certainty_np = np.asarray(
        sequence_self_certainty,
        dtype=np.float32,
    )
    advantages_self_certainty_np = compute_group_relative_advantages(
        rewards_self_certainty_np,
        group_size=group_size,
    )
    advantages_calibration_np = build_calibration_advantages(
        rewards_calibration=calibration_batch.calibration_rewards,
        group_size=group_size,
        mode=mode,
        clip=calibration_clip,
    )
    if np.any(calibration_batch.majority_mask):
        advantages_overconfidence_np = build_overconfidence_advantages_for_mode(
            mode=mode,
            advantages_self_certainty=advantages_self_certainty_np,
            majority_correctness=_group_correctness(calibration_batch),
            majority_mask=calibration_batch.majority_mask,
            group_size=group_size,
            clip=overconfidence_clip,
        )
    else:
        advantages_overconfidence_np = np.zeros_like(
            advantages_self_certainty_np,
            dtype=np.float32,
        )

    rewards_self_certainty = jnp.asarray(
        rewards_self_certainty_np,
        dtype=jnp.float32,
    )
    rewards_calibration = jnp.asarray(
        calibration_batch.calibration_rewards,
        dtype=jnp.float32,
    )
    advantages_self_certainty = jnp.asarray(
        advantages_self_certainty_np,
        dtype=jnp.float32,
    )
    advantages_calibration = jnp.asarray(
        advantages_calibration_np,
        dtype=jnp.float32,
    )
    advantages_overconfidence = jnp.asarray(
        advantages_overconfidence_np,
        dtype=jnp.float32,
    )
    calibration_weight = jnp.asarray(calibration_lambda, dtype=jnp.float32)
    overconfidence_weight = jnp.asarray(overconfidence_lambda, dtype=jnp.float32)

    return JaxObjectiveBranches(
        rewards_self_certainty=rewards_self_certainty,
        rewards_calibration=rewards_calibration,
        advantages_self_certainty=advantages_self_certainty,
        advantages_calibration=advantages_calibration,
        advantages_overconfidence=advantages_overconfidence,
        combined_rewards=(
            rewards_self_certainty
            + calibration_weight * rewards_calibration
            + overconfidence_weight * advantages_overconfidence
        ),
        combined_advantages=(
            advantages_self_certainty
            + calibration_weight * advantages_calibration
            + overconfidence_weight * advantages_overconfidence
        ),
        calibration_lambda=calibration_weight,
        overconfidence_lambda=overconfidence_weight,
        calibration_batch=calibration_batch,
    )


def prepare_objective_from_policy(
    *,
    recompute_policy: PolicyRecompute,
    prompt_tokens: Array,
    prompt_mask: Array,
    completion_tokens: Array,
    completion_mask: Array,
    completion_texts: Sequence[str],
    gold_answers: Sequence[str],
    group_size: int,
    verifier: Verifier,
    mode: str,
    calibration_lambda: float,
    overconfidence_lambda: float = 0.0,
    min_majority_confidence: float = 0.0,
    calibration_clip: float | None = None,
    overconfidence_clip: float = 2.0,
    minority_negative_weight: float = 0.0,
) -> JaxObjectiveBranches:
    """Recompute rollout-policy logits, then build the TPU objective branches."""

    completion_logits = recompute_policy(
        prompt_tokens=prompt_tokens,
        prompt_mask=prompt_mask,
        completion_tokens=completion_tokens,
        completion_mask=completion_mask,
    )
    return prepare_objective_branches(
        completion_logits=completion_logits,
        completion_mask=completion_mask,
        completion_texts=completion_texts,
        gold_answers=gold_answers,
        group_size=group_size,
        verifier=verifier,
        mode=mode,
        calibration_lambda=calibration_lambda,
        overconfidence_lambda=overconfidence_lambda,
        min_majority_confidence=min_majority_confidence,
        calibration_clip=calibration_clip,
        overconfidence_clip=overconfidence_clip,
        minority_negative_weight=minority_negative_weight,
    )


def _aggregate_token_mean(per_token_loss: Array, completion_mask: Array) -> Array:
    mask = jnp.asarray(completion_mask, dtype=jnp.float32)
    return jnp.sum(per_token_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def compute_dual_objective_pg_loss(
    *,
    coef_1: Array,
    coef_2: Array,
    completion_mask: Array,
    branches: JaxObjectiveBranches,
) -> DualObjectiveLoss:
    """Compute the clipped three-branch policy-gradient loss used on TPU.

    The coefficients are the two GRPO/PPO clipped-ratio candidates produced by
    the surrounding Tunix learner. This public function implements token-mean
    aggregation; callers using another Tunix loss aggregation mode must provide
    the corresponding reduction.
    """

    first = jnp.asarray(coef_1, dtype=jnp.float32)
    second = jnp.asarray(coef_2, dtype=jnp.float32)
    mask = jnp.asarray(completion_mask, dtype=jnp.float32)
    if first.shape != second.shape or first.shape != mask.shape:
        raise ValueError(
            "coef_1, coef_2, and completion_mask must have identical shapes; "
            f"got {first.shape}, {second.shape}, and {mask.shape}"
        )

    sc = jnp.asarray(branches.advantages_self_certainty, dtype=jnp.float32)
    cal = jnp.asarray(branches.advantages_calibration, dtype=jnp.float32)
    oc = jnp.asarray(branches.advantages_overconfidence, dtype=jnp.float32)
    expected_samples = first.shape[0]
    if sc.shape != (expected_samples,) or cal.shape != (expected_samples,):
        raise ValueError("self-certainty and calibration advantages must be per-sample")
    if oc.shape != (expected_samples,):
        raise ValueError("overconfidence advantages must be per-sample")

    def branch_loss(advantages: Array) -> tuple[Array, Array]:
        first_term = first * jnp.expand_dims(advantages, axis=1)
        second_term = second * jnp.expand_dims(advantages, axis=1)
        loss = -jnp.minimum(first_term, second_term)
        clip_fraction = jnp.sum(((-second_term) > (-first_term)) * mask) / jnp.maximum(
            jnp.sum(mask),
            1.0,
        )
        return loss, clip_fraction

    sc_loss, sc_clipfrac = branch_loss(sc)
    cal_loss, cal_clipfrac = branch_loss(cal)
    oc_loss, oc_clipfrac = branch_loss(oc)
    per_token_loss = (
        sc_loss
        + branches.calibration_lambda * cal_loss
        + branches.overconfidence_lambda * oc_loss
    )
    return DualObjectiveLoss(
        loss=_aggregate_token_mean(per_token_loss, mask),
        per_token_loss=per_token_loss,
        pg_clipfrac_self_certainty=sc_clipfrac,
        pg_clipfrac_calibration=cal_clipfrac,
        pg_clipfrac_overconfidence=oc_clipfrac,
    )
