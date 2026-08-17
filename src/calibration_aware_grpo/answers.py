"""Conservative answer extraction and canonicalization helpers."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import NamedTuple

NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
SIMPLE_DECIMAL_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
SIMPLE_SLASH_FRACTION_RE = re.compile(
    rf"(?P<numerator>{SIMPLE_DECIMAL_RE.pattern})/"
    rf"(?P<denominator>{SIMPLE_DECIMAL_RE.pattern})"
)
SIMPLE_LATEX_FRACTION_RE = re.compile(
    rf"(?P<sign>[-+]?)\\(?:frac|dfrac|tfrac)"
    rf"\{{(?P<numerator>{SIMPLE_DECIMAL_RE.pattern})\}}"
    rf"\{{(?P<denominator>{SIMPLE_DECIMAL_RE.pattern})\}}"
)
_MAX_DECIMAL_DIGITS = 50
_MAX_DECIMAL_EXPONENT_ABS = 50
_MAX_COMPLETION_CHARS = 100_000
_MAX_BOX_MARKERS = 64


class BoxedAnswerSpan(NamedTuple):
    """A brace-balanced ``\\boxed{...}`` answer and its source span."""

    answer: str
    start: int
    end: int


def find_last_boxed_answer_span(text: object) -> BoxedAnswerSpan | None:
    """Return the last well-formed, non-empty ``\\boxed{...}`` answer.

    The scanner is brace-aware, so nested LaTeX such as
    ``\\boxed{\\frac{14}{3}}`` is handled without a symbolic parser. If the
    final marker is malformed, scanning continues from the preceding marker.
    """

    raw = str(text)
    marker = r"\boxed{"
    if len(raw) > _MAX_COMPLETION_CHARS or raw.count(marker) > _MAX_BOX_MARKERS:
        return None

    def is_escaped(index: int) -> bool:
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and raw[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        return backslashes % 2 == 1

    search_end = len(raw)
    while True:
        start = raw.rfind(marker, 0, search_end)
        if start < 0:
            return None

        chars: list[str] = []
        depth = 1
        index = start + len(marker)
        while index < len(raw):
            char = raw[index]
            if char == "{" and not is_escaped(index):
                depth += 1
            elif char == "}" and not is_escaped(index):
                depth -= 1
                if depth == 0:
                    answer = "".join(chars).strip()
                    if answer:
                        return BoxedAnswerSpan(answer, start, index + 1)
                    break
            chars.append(char)
            index += 1

        search_end = start


def extract_last_boxed_answer(text: object) -> str | None:
    """Return only the answer from :func:`find_last_boxed_answer_span`."""

    span = find_last_boxed_answer_span(text)
    return None if span is None else span.answer


def normalize_math_text(text: str) -> str:
    """Remove superficial LaTeX wrappers and whitespace."""

    normalized = str(text).strip().replace("$", "")
    normalized = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", normalized)
    return re.sub(r"\s+", "", normalized)


def _parse_decimal_fraction(text: str) -> Fraction | None:
    candidate = str(text).strip()
    if not candidate or not SIMPLE_DECIMAL_RE.fullmatch(candidate):
        return None
    mantissa, separator, exponent = candidate.replace(",", "").partition("e")
    if not separator:
        mantissa, separator, exponent = candidate.replace(",", "").partition("E")
    if sum(character.isdigit() for character in mantissa) > _MAX_DECIMAL_DIGITS:
        return None
    if separator:
        exponent_digits = exponent.lstrip("+-")
        if len(exponent_digits) > 3:
            return None
        if abs(int(exponent)) > _MAX_DECIMAL_EXPONENT_ABS:
            return None
    try:
        value = Decimal(candidate.replace(",", ""))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return Fraction(value)


def canonicalize_numeric_answer(text: str) -> str | None:
    """Return one key for equivalent simple numeric answers.

    Supported forms are finite decimals, signed integers, slash fractions, and
    simple LaTeX ``frac``/``dfrac``/``tfrac`` expressions. General symbolic
    equivalence is deliberately out of scope.
    """

    normalized = normalize_math_text(text).replace("−", "-")
    if not normalized:
        return None

    value: Fraction | None = None
    latex_match = SIMPLE_LATEX_FRACTION_RE.fullmatch(normalized)
    if latex_match:
        numerator = _parse_decimal_fraction(latex_match.group("numerator"))
        denominator = _parse_decimal_fraction(latex_match.group("denominator"))
        if numerator is None or denominator is None or denominator == 0:
            return None
        value = numerator / denominator
        if latex_match.group("sign") == "-":
            value = -value
    else:
        slash_match = SIMPLE_SLASH_FRACTION_RE.fullmatch(normalized)
        if slash_match:
            numerator = _parse_decimal_fraction(slash_match.group("numerator"))
            denominator = _parse_decimal_fraction(slash_match.group("denominator"))
            if numerator is None or denominator is None or denominator == 0:
                return None
            value = numerator / denominator
        else:
            value = _parse_decimal_fraction(normalized)

    if value is None:
        return None
    if value.denominator == 1:
        return f"num::{value.numerator}"
    return f"num::{value.numerator}/{value.denominator}"


def canonicalize_answer_key(
    completion_text: str,
    *,
    unparsable_id: str | int,
) -> tuple[str, bool]:
    """Map one completion to a conservative equality key.

    Unparsable completions receive a caller-provided unique identifier so that
    unrelated free-form outputs cannot accidentally form a majority cluster.
    """

    text = str(completion_text)
    if len(text) > _MAX_COMPLETION_CHARS or text.count(r"\boxed{") > _MAX_BOX_MARKERS:
        return f"__unparsable__::{unparsable_id}", False
    boxed_answer = extract_last_boxed_answer(text)
    if boxed_answer is not None:
        numeric_key = canonicalize_numeric_answer(boxed_answer)
        if numeric_key is not None:
            return numeric_key, True
        normalized = normalize_math_text(boxed_answer)
        if normalized:
            return f"boxed::{normalized}", True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    trailing_text = lines[-1] if lines else text
    numeric_key = canonicalize_numeric_answer(trailing_text)
    if numeric_key is not None:
        return numeric_key, True

    if "\\" not in trailing_text:
        numeric_matches = [
            match.group(0).replace(",", "")
            for match in NUMBER_RE.finditer(trailing_text.replace("−", "-"))
        ]
        if len(numeric_matches) == 1:
            numeric_key = canonicalize_numeric_answer(numeric_matches[0])
            if numeric_key is not None:
                return numeric_key, True

    return f"__unparsable__::{unparsable_id}", False
