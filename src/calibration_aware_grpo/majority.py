"""Majority grouping and batched verifier orchestration."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol, TypeAlias

import numpy as np

from .answers import canonicalize_answer_key
from .objectives import (
    ASYMMETRIC_MINORITY_MODES,
    compute_calibration_raw_rewards,
    validate_mode,
)

SkipReason: TypeAlias = Literal[
    "tied_majority",
    "below_min_majority_confidence",
]


class Verifier(Protocol):
    """Keyword-only batch verifier contract."""

    def __call__(
        self,
        *,
        gold_answers: Sequence[str],
        completion_texts: Sequence[str],
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class MajorityGroupResult:
    """Diagnostics and labels for one rollout group."""

    start: int
    stop: int
    answer_key: str | None
    majority_mask: np.ndarray
    confidence: float | None
    correctness: float | None
    brier_score: float | None
    cluster_size: int
    representative_completion: str | None
    skip_reason: SkipReason | None


@dataclass(frozen=True)
class CalibrationBatch:
    """Per-sample calibration rewards plus per-group diagnostics."""

    calibration_rewards: np.ndarray
    majority_mask: np.ndarray
    groups: tuple[MajorityGroupResult, ...]

    @property
    def group_count(self) -> int:
        return len(self.groups)


def _validate_minimum_confidence(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(
            f"min_majority_confidence must be finite and in [0, 1]; got {value!r}"
        )
    return result


def _validate_verifier_score(value: float, group_index: int) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(
            f"verifier score for group {group_index} must be finite and in [0, 1]; "
            f"got {value!r}"
        )
    return result


def build_calibration_batch(
    *,
    completions: Sequence[str],
    gold_answers: Sequence[str],
    group_size: int,
    verifier: Verifier,
    min_majority_confidence: float = 0.0,
    mode: str = "raw_signed",
    minority_negative_weight: float = 0.0,
) -> CalibrationBatch:
    """Group completions, verify representatives once, and construct rewards.

    ``gold_answers`` contains one answer per rollout group. Eligible plurality
    representatives are sent to ``verifier`` in one batched call. The verifier
    receives keyword arguments ``gold_answers`` and ``completion_texts`` and
    must return one score in ``[0, 1]`` for every request.

    The first completion in the winning canonical-answer cluster is the
    representative. The verifier must grade final-answer correctness
    invariantly across completions with that same canonical answer; a
    reasoning- or formatting-sensitive verifier is not compatible with this
    cluster-level label. Confidence exactly equal to
    ``min_majority_confidence`` remains eligible.
    """

    checked_mode = validate_mode(mode)
    minimum_confidence = _validate_minimum_confidence(min_majority_confidence)
    if isinstance(group_size, (bool, np.bool_)) or not isinstance(
        group_size, (int, np.integer)
    ):
        raise TypeError(f"group_size must be an integer; got {group_size!r}")
    group_size = int(group_size)
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    minority_weight = float(minority_negative_weight)
    if not np.isfinite(minority_weight) or minority_weight < 0.0:
        raise ValueError("minority_negative_weight must be finite and non-negative")
    if checked_mode not in ASYMMETRIC_MINORITY_MODES and minority_weight != 0.0:
        raise ValueError(
            "minority_negative_weight is only valid for asymmetric minority modes"
        )
    if not callable(verifier):
        raise TypeError("verifier must be callable")

    completion_texts = [str(completion) for completion in completions]
    if len(completion_texts) % group_size != 0:
        raise ValueError(
            f"completions length must be divisible by group_size; got "
            f"{len(completion_texts)} and {group_size}"
        )
    group_count = len(completion_texts) // group_size
    if len(gold_answers) != group_count:
        raise ValueError(
            f"gold_answers must contain one answer per group; got "
            f"{len(gold_answers)}, expected {group_count}"
        )

    rewards = np.zeros(len(completion_texts), dtype=np.float32)
    sample_majority_mask = np.zeros(len(completion_texts), dtype=bool)
    group_results: list[MajorityGroupResult] = []
    pending: list[tuple[int, str, str]] = []

    for group_index in range(group_count):
        start = group_index * group_size
        stop = start + group_size
        group_texts = completion_texts[start:stop]
        keys = [
            canonicalize_answer_key(text, unparsable_id=start + local_index)[0]
            for local_index, text in enumerate(group_texts)
        ]
        counts = Counter(keys)
        maximum_count = max(counts.values())
        winning_keys = [key for key, count in counts.items() if count == maximum_count]

        if len(winning_keys) != 1:
            group_results.append(
                MajorityGroupResult(
                    start=start,
                    stop=stop,
                    answer_key=None,
                    majority_mask=np.zeros(group_size, dtype=bool),
                    confidence=None,
                    correctness=None,
                    brier_score=None,
                    cluster_size=maximum_count,
                    representative_completion=None,
                    skip_reason="tied_majority",
                )
            )
            continue

        majority_key = winning_keys[0]
        group_mask = np.asarray([key == majority_key for key in keys], dtype=bool)
        sample_majority_mask[start:stop] = group_mask
        representative_index = int(np.flatnonzero(group_mask)[0])
        representative = group_texts[representative_index]
        confidence = maximum_count / group_size

        if confidence < minimum_confidence:
            group_results.append(
                MajorityGroupResult(
                    start=start,
                    stop=stop,
                    answer_key=majority_key,
                    majority_mask=group_mask,
                    confidence=confidence,
                    correctness=None,
                    brier_score=None,
                    cluster_size=maximum_count,
                    representative_completion=representative,
                    skip_reason="below_min_majority_confidence",
                )
            )
            continue

        result_index = len(group_results)
        group_results.append(
            MajorityGroupResult(
                start=start,
                stop=stop,
                answer_key=majority_key,
                majority_mask=group_mask,
                confidence=confidence,
                correctness=None,
                brier_score=None,
                cluster_size=maximum_count,
                representative_completion=representative,
                skip_reason=None,
            )
        )
        pending.append((result_index, str(gold_answers[group_index]), representative))

    if pending:
        verifier_scores = list(
            verifier(
                gold_answers=[gold_answer for _, gold_answer, _ in pending],
                completion_texts=[representative for _, _, representative in pending],
            )
        )
    else:
        verifier_scores = []
    if len(verifier_scores) != len(pending):
        raise RuntimeError(
            "verifier returned an unexpected number of scores: "
            f"expected={len(pending)}, got={len(verifier_scores)}"
        )

    for pending_index, (result_index, _gold_answer, _representative) in enumerate(
        pending
    ):
        result = group_results[result_index]
        correctness = _validate_verifier_score(
            verifier_scores[pending_index], result_index
        )
        assert result.confidence is not None
        group_rewards = compute_calibration_raw_rewards(
            correctness=correctness,
            majority_confidence=result.confidence,
            majority_mask=result.majority_mask,
            mode=checked_mode,
            minority_negative_weight=minority_weight,
        )
        rewards[result.start : result.stop] = group_rewards
        group_results[result_index] = replace(
            result,
            correctness=correctness,
            brier_score=(result.confidence - correctness) ** 2,
        )

    return CalibrationBatch(
        calibration_rewards=rewards,
        majority_mask=sample_majority_mask,
        groups=tuple(group_results),
    )
