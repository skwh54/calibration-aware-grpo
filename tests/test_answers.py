import pytest

from calibration_aware_grpo.answers import (
    canonicalize_answer_key,
    canonicalize_numeric_answer,
    extract_last_boxed_answer,
    find_last_boxed_answer_span,
)


def test_extracts_nested_latex_from_last_box():
    text = r"first \boxed{2}, final \boxed{\frac{14}{3}}"
    span = find_last_boxed_answer_span(text)
    assert span is not None
    assert span.answer == r"\frac{14}{3}"
    assert text[span.start : span.end] == r"\boxed{\frac{14}{3}}"


def test_falls_back_when_last_box_is_malformed():
    text = r"valid \boxed{7}; broken \boxed{8"
    assert extract_last_boxed_answer(text) == "7"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("1", "num::1"),
        ("1.0", "num::1"),
        ("0.5", "num::1/2"),
        ("1/2", "num::1/2"),
        (r"\frac{2}{4}", "num::1/2"),
        (r"-\dfrac{3}{6}", "num::-1/2"),
        ("1,000", "num::1000"),
        ("1e-2", "num::1/100"),
    ],
)
def test_numeric_equivalence_classes(answer, expected):
    assert canonicalize_numeric_answer(answer) == expected


@pytest.mark.parametrize("answer", ["1/0", r"\frac{1}{0}", "nan", "x + 1"])
def test_rejects_unsupported_numeric_forms(answer):
    assert canonicalize_numeric_answer(answer) is None


def test_numeric_canonicalization_enforces_resource_bounds():
    assert canonicalize_numeric_answer("9" * 50) is not None
    assert canonicalize_numeric_answer("9" * 51) is None
    assert canonicalize_numeric_answer("1e50") is not None
    assert canonicalize_numeric_answer("1e51") is None


def test_boxed_symbolic_answer_uses_normalized_key():
    key, parsable = canonicalize_answer_key(
        r"Therefore \boxed{\left x + y \right}", unparsable_id=0
    )
    assert parsable
    assert key == r"boxed::x+y"


def test_trailing_single_number_is_recovered():
    key, parsable = canonicalize_answer_key(
        "The result is approximately 42.", unparsable_id=0
    )
    assert parsable
    assert key == "num::42"


def test_multiple_unboxed_numbers_are_not_guessed():
    key, parsable = canonicalize_answer_key(
        "The candidates are 1 and 2.", unparsable_id="sample-3"
    )
    assert not parsable
    assert key == "__unparsable__::sample-3"


@pytest.mark.parametrize(
    "text",
    [
        "The roots are 2 and 2.",
        "Result: 2/2",
    ],
)
def test_repeated_or_embedded_numbers_are_not_collapsed(text):
    key, parsable = canonicalize_answer_key(text, unparsable_id=4)
    assert not parsable
    assert key == "__unparsable__::4"


def test_latex_control_words_do_not_collide_during_normalization():
    left, _ = canonicalize_answer_key(r"\boxed{\leftarrow}", unparsable_id=0)
    right, _ = canonicalize_answer_key(r"\boxed{\rightarrow}", unparsable_id=1)
    assert left == r"boxed::\leftarrow"
    assert right == r"boxed::\rightarrow"
    assert left != right


def test_escaped_close_brace_does_not_end_box_early():
    assert extract_last_boxed_answer(r"\boxed{x\}}") == r"x\}"


@pytest.mark.parametrize(
    "text",
    [
        "x" * 100_001,
        r"\boxed{" * 65 + "1" + "}" * 65,
    ],
    ids=["too_long", "too_many_box_markers"],
)
def test_over_budget_completion_is_unparsable(text):
    key, parsable = canonicalize_answer_key(text, unparsable_id="large")
    assert not parsable
    assert key == "__unparsable__::large"


def test_unparsable_ids_prevent_accidental_clusters():
    first, _ = canonicalize_answer_key("no answer", unparsable_id=0)
    second, _ = canonicalize_answer_key("no answer", unparsable_id=1)
    assert first != second
