"""NumPy reference implementation of calibration-aware GRPO signals."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import numpy as np

CalibrationMode: TypeAlias = Literal[
    "group_norm_signed",
    "raw_signed",
    "raw_rlcr",
    "raw_signed_oc",
    "raw_signed_asym",
    "raw_signed_asym_oc",
]
CALIBRATION_MODES: tuple[CalibrationMode, ...] = (
    "group_norm_signed",
    "raw_signed",
    "raw_rlcr",
    "raw_signed_oc",
    "raw_signed_asym",
    "raw_signed_asym_oc",
)
RAW_MODES = frozenset(CALIBRATION_MODES[1:])
OVERCONFIDENCE_MODES = frozenset({"raw_signed_oc", "raw_signed_asym_oc"})
ASYMMETRIC_MINORITY_MODES = frozenset({"raw_signed_asym", "raw_signed_asym_oc"})


def validate_mode(mode: str) -> CalibrationMode:
    """Validate and normalize a calibration mode name."""

    normalized = str(mode).strip()
    if normalized not in CALIBRATION_MODES:
        expected = ", ".join(CALIBRATION_MODES)
        raise ValueError(
            f"unsupported calibration mode {normalized!r}; expected {expected}"
        )
    return normalized  # type: ignore[return-value]


def _validate_probability(value: float, field_name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]; got {value!r}")
    return result


def _validate_nonnegative(value: float, field_name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative; got {value!r}")
    return result


def _validate_integer(value: int, field_name: str, *, allow_negative: bool) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer; got {value!r}")
    result = int(value)
    if not allow_negative and result < 0:
        raise ValueError(f"{field_name} must be non-negative; got {value!r}")
    return result


def _validate_boolean_mask(
    values: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype != np.bool_:
        raise TypeError(f"{field_name} must contain booleans, not {raw.dtype}")
    mask = raw.reshape(-1)
    if mask.size == 0 and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    return mask


def compute_group_relative_advantages(
    rewards: Any,
    *,
    group_size: int,
) -> np.ndarray:
    """Compute per-group sample-standardized GRPO advantages.

    Groups with constant rewards map to zero. The ``1e-4`` denominator epsilon
    and sample standard deviation (``ddof=1``) match the research trainer.
    """

    group_size = _validate_integer(group_size, "group_size", allow_negative=False)
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    rewards_array = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if rewards_array.size == 0:
        return rewards_array.astype(np.float32)
    if rewards_array.size % group_size != 0:
        raise ValueError(
            f"rewards length must be divisible by group_size; got "
            f"{rewards_array.size} and {group_size}"
        )
    if not np.all(np.isfinite(rewards_array)):
        raise ValueError("rewards must contain only finite values")

    grouped = rewards_array.reshape(-1, group_size)
    means = grouped.mean(axis=-1, keepdims=True)
    standard_deviations = grouped.std(axis=-1, ddof=1, keepdims=True)
    advantages = (grouped - means) / (standard_deviations + 1e-4)
    return advantages.reshape(-1).astype(np.float32)


def compute_calibration_raw_rewards(
    *,
    correctness: float,
    majority_confidence: float,
    majority_mask: Any,
    mode: str,
    minority_negative_weight: float = 0.0,
) -> np.ndarray:
    """Construct raw calibration rewards for one rollout group."""

    checked_mode = validate_mode(mode)
    y = _validate_probability(correctness, "correctness")
    p = _validate_probability(majority_confidence, "majority_confidence")
    minority_weight = _validate_nonnegative(
        minority_negative_weight, "minority_negative_weight"
    )
    if checked_mode not in ASYMMETRIC_MINORITY_MODES and minority_weight != 0.0:
        raise ValueError(
            "minority_negative_weight is only valid for asymmetric minority modes"
        )

    mask = _validate_boolean_mask(majority_mask, "majority_mask")
    if not np.any(mask):
        raise ValueError("majority_mask must select at least one sample")
    observed_confidence = float(mask.mean())
    if not np.isclose(p, observed_confidence, rtol=0.0, atol=1e-6):
        raise ValueError(
            "majority_confidence must equal the selected mask fraction; "
            f"got {p} and {observed_confidence}"
        )
    if checked_mode == "raw_rlcr":
        majority_value = y - (y - p) ** 2
    else:
        majority_value = y - p

    rewards = majority_value * mask.astype(np.float32)
    if checked_mode in ASYMMETRIC_MINORITY_MODES:
        minority_value = -minority_weight * y * p
        rewards = rewards + minority_value * (~mask).astype(np.float32)
    return np.asarray(rewards, dtype=np.float32)


def build_calibration_advantages(
    *,
    rewards_calibration: Any,
    group_size: int,
    mode: str,
    clip: float | None = None,
) -> np.ndarray:
    """Convert calibration rewards to final calibration advantages."""

    checked_mode = validate_mode(mode)
    group_size = _validate_integer(group_size, "group_size", allow_negative=False)
    rewards = np.asarray(rewards_calibration, dtype=np.float32).reshape(-1)
    if rewards.size == 0:
        return rewards.copy()
    if group_size <= 0 or rewards.size % group_size != 0:
        raise ValueError(
            f"rewards_calibration length must be divisible by group_size; got "
            f"{rewards.size} and {group_size}"
        )
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards_calibration must contain only finite values")

    if checked_mode == "group_norm_signed":
        return compute_group_relative_advantages(
            rewards,
            group_size=group_size,
        )

    advantages = rewards.copy()
    if clip is not None:
        clip_value = _validate_nonnegative(abs(float(clip)), "clip")
        advantages = np.clip(advantages, -clip_value, clip_value)
    return advantages.astype(np.float32, copy=False)


def _expand_group_values(
    values: Any,
    *,
    sample_count: int,
    group_size: int,
    field_name: str,
) -> np.ndarray:
    group_size = _validate_integer(group_size, "group_size", allow_negative=False)
    if group_size <= 0 or sample_count % group_size != 0:
        raise ValueError(
            f"sample_count must be divisible by group_size; got "
            f"{sample_count} and {group_size}"
        )
    values_array = np.asarray(values, dtype=np.float32).reshape(-1)
    if values_array.size == sample_count:
        return values_array
    group_count = sample_count // group_size
    if values_array.size != group_count:
        raise ValueError(
            f"{field_name} must contain one value per group or sample; got "
            f"{values_array.size}, expected {group_count} or {sample_count}"
        )
    return np.repeat(values_array, group_size).astype(np.float32, copy=False)


def build_overconfidence_wrong_advantages(
    *,
    advantages_self_certainty: Any,
    majority_correctness: Any,
    majority_mask: Any,
    group_size: int,
    clip: float = 2.0,
) -> np.ndarray:
    """Penalize positive self-certainty advantages in wrong majorities."""

    group_size = _validate_integer(group_size, "group_size", allow_negative=False)
    self_certainty = np.asarray(advantages_self_certainty, dtype=np.float32).reshape(-1)
    if group_size <= 0 or self_certainty.size % group_size != 0:
        raise ValueError(
            "advantages_self_certainty length must be divisible by group_size; "
            f"got {self_certainty.size} and {group_size}"
        )

    if not np.all(np.isfinite(self_certainty)):
        raise ValueError("advantages_self_certainty must contain only finite values")
    mask = _validate_boolean_mask(
        majority_mask,
        "majority_mask",
        allow_empty=True,
    )
    if mask.shape != self_certainty.shape:
        raise ValueError(
            "majority_mask must match advantages_self_certainty; got "
            f"{mask.shape} and {self_certainty.shape}"
        )
    if self_certainty.size == 0:
        return self_certainty.copy()
    if not np.any(mask):
        raise ValueError("majority_mask must select at least one sample")
    correctness = _expand_group_values(
        majority_correctness,
        sample_count=self_certainty.size,
        group_size=group_size,
        field_name="majority_correctness",
    )
    if not np.all(np.isfinite(correctness)):
        raise ValueError("majority_correctness must contain only finite values")
    if np.any((correctness < 0.0) | (correctness > 1.0)):
        raise ValueError("majority_correctness values must be in [0, 1]")
    clip_value = _validate_nonnegative(clip, "clip")
    positive_mass = np.clip(np.maximum(self_certainty, 0.0), 0.0, clip_value)
    wrong_majority = mask & (correctness < 0.5)
    return np.asarray(
        -wrong_majority.astype(np.float32) * positive_mass, dtype=np.float32
    )


def build_overconfidence_advantages_for_mode(
    *,
    mode: str,
    advantages_self_certainty: Any,
    majority_correctness: Any,
    majority_mask: Any,
    group_size: int,
    clip: float = 2.0,
) -> np.ndarray:
    """Return OC advantages only for modes that explicitly enable them."""

    checked_mode = validate_mode(mode)
    group_size = _validate_integer(group_size, "group_size", allow_negative=False)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    self_certainty = np.asarray(advantages_self_certainty, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(self_certainty)):
        raise ValueError("advantages_self_certainty must contain only finite values")
    if checked_mode not in OVERCONFIDENCE_MODES:
        return np.zeros_like(self_certainty, dtype=np.float32)
    return build_overconfidence_wrong_advantages(
        advantages_self_certainty=self_certainty,
        majority_correctness=majority_correctness,
        majority_mask=majority_mask,
        group_size=group_size,
        clip=clip,
    )


def resolve_calibration_lambda(
    *,
    current_step: int,
    target_lambda: float,
    max_steps: int | None = None,
    warmup_fraction: float | None = None,
    warmup_steps: int | None = None,
) -> float:
    """Resolve a linearly warmed calibration or OC branch weight.

    Explicit ``warmup_steps`` takes precedence over the fraction-based
    schedule. Negative current steps are clamped to zero.
    """

    step = _validate_integer(current_step, "current_step", allow_negative=True)
    target = _validate_nonnegative(target_lambda, "target_lambda")
    if warmup_steps is not None:
        resolved_warmup_steps = _validate_integer(
            warmup_steps, "warmup_steps", allow_negative=False
        )
        if target == 0.0:
            return 0.0
        if resolved_warmup_steps == 0:
            return target
        progress = min(max(step, 0), resolved_warmup_steps)
        return float(target * progress / resolved_warmup_steps)

    fraction = _validate_probability(
        0.0 if warmup_fraction is None else warmup_fraction,
        "warmup_fraction",
    )
    if target == 0.0:
        return 0.0
    if fraction == 0.0:
        return target
    if max_steps is None:
        raise ValueError("max_steps is required when warmup_fraction is positive")
    resolved_max_steps = _validate_integer(max_steps, "max_steps", allow_negative=False)
    if resolved_max_steps == 0:
        raise ValueError("max_steps must be positive when warmup_fraction is positive")
    resolved_warmup_steps = max(1, int(resolved_max_steps * fraction))
    progress = min(max(step, 0), resolved_warmup_steps)
    return float(target * progress / resolved_warmup_steps)


def compose_advantages(
    *,
    self_certainty: Any,
    calibration: Any,
    overconfidence: Any,
    calibration_weight: float,
    overconfidence_weight: float,
) -> np.ndarray:
    """Combine separately computed advantage branches."""

    branches = [
        np.asarray(self_certainty, dtype=np.float64).reshape(-1),
        np.asarray(calibration, dtype=np.float64).reshape(-1),
        np.asarray(overconfidence, dtype=np.float64).reshape(-1),
    ]
    if len({branch.shape for branch in branches}) != 1:
        raise ValueError("all advantage branches must have the same shape")
    if not all(np.all(np.isfinite(branch)) for branch in branches):
        raise ValueError("advantage branches must contain only finite values")
    calibration_lambda = _validate_nonnegative(calibration_weight, "calibration_weight")
    overconfidence_lambda = _validate_nonnegative(
        overconfidence_weight, "overconfidence_weight"
    )
    combined = np.asarray(
        branches[0]
        + calibration_lambda * branches[1]
        + overconfidence_lambda * branches[2],
        dtype=np.float64,
    )
    float32_limit = np.finfo(np.float32).max
    if not np.all(np.isfinite(combined)) or np.any(np.abs(combined) > float32_limit):
        raise OverflowError("composed advantages are not finite float32 values")
    return combined.astype(np.float32)
