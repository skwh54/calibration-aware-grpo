"""CPU reference kernel for calibration-aware grouped policy optimization."""

from .answers import (
    BoxedAnswerSpan,
    canonicalize_answer_key,
    canonicalize_numeric_answer,
    extract_last_boxed_answer,
    find_last_boxed_answer_span,
)
from .majority import CalibrationBatch, MajorityGroupResult, build_calibration_batch
from .objectives import (
    ASYMMETRIC_MINORITY_MODES,
    CALIBRATION_MODES,
    OVERCONFIDENCE_MODES,
    build_calibration_advantages,
    build_overconfidence_advantages_for_mode,
    build_overconfidence_wrong_advantages,
    compose_advantages,
    compute_calibration_raw_rewards,
    compute_group_relative_advantages,
    resolve_calibration_lambda,
    validate_mode,
)

__all__ = [
    "ASYMMETRIC_MINORITY_MODES",
    "CALIBRATION_MODES",
    "OVERCONFIDENCE_MODES",
    "BoxedAnswerSpan",
    "CalibrationBatch",
    "MajorityGroupResult",
    "build_calibration_advantages",
    "build_calibration_batch",
    "build_overconfidence_advantages_for_mode",
    "build_overconfidence_wrong_advantages",
    "canonicalize_answer_key",
    "canonicalize_numeric_answer",
    "compose_advantages",
    "compute_calibration_raw_rewards",
    "compute_group_relative_advantages",
    "extract_last_boxed_answer",
    "find_last_boxed_answer_span",
    "resolve_calibration_lambda",
    "validate_mode",
]
