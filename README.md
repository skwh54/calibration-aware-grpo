# Calibration-Aware GRPO

A small, accelerator-independent reference implementation of calibration signals
for grouped policy optimization. The package isolates the objective kernel used
in a larger Qwen/JAX/Tunix research system so that its assumptions and failure
modes can be tested on CPU without a TPU stack.

## Scope

Included:

- brace-aware extraction of the last `\boxed{...}` answer;
- conservative numeric answer canonicalization;
- answer-frequency unique-plurality grouping;
- batched verification of one representative per eligible group;
- six calibration reward/advantage modes;
- an explicit overconfident-wrong penalty branch;
- a linear warmup schedule for calibration and OC branch weights;
- composition of self-certainty, calibration, and overconfidence advantages;
- invariant-focused NumPy tests.

Not included:

- a model, dataset, rollout engine, or full GRPO trainer;
- JAX, Tunix, vLLM, checkpoint, sharding, or TPU integration;
- broad symbolic equivalence or an automatic mathematical verifier;
- claims that CPU tests validate accelerator-specific training behavior.

The verifier remains an explicit dependency supplied by the caller. This keeps
semantic correctness labels separate from answer-frequency statistics. It must
grade final-answer correctness invariantly across completions sharing the same
canonical answer. Reasoning- or formatting-sensitive scoring cannot safely be
propagated from one representative to the whole cluster.

## Objective modes

For a rollout group, let `p` be the unique winning answer's frequency, `y` the
verified winning-answer correctness in `[0, 1]`, and `M_i` indicate whether
sample `i` belongs to that cluster. The grouping rule is a unique plurality,
not a requirement that `p > 0.5`; tied groups are skipped.

| Mode | Calibration signal | Intended role |
| --- | --- | --- |
| `group_norm_signed` | group-normalized `(y - p) M_i` | Historical relative-objective reproduction |
| `raw_signed` | `(y - p) M_i` | Conservative checked-majority correction |
| `raw_rlcr` | `[y - (y - p)^2] M_i` | RLCR-style cluster-level ablation |
| `raw_signed_oc` | `raw_signed` plus OC branch | Direct overconfident-wrong ablation |
| `raw_signed_asym` | `(y - p) M_i - rho y p (1-M_i)` | Minority-penalty ablation when the majority is correct |
| `raw_signed_asym_oc` | asymmetric signal plus OC branch | Aggressive combined ablation |

The overconfidence branch is

```text
A_oc_i = -1[majority wrong] M_i clip(ReLU(A_sc_i), 0, tau)
```

where `A_sc_i` is a prompt-local self-certainty advantage. Neither `raw_signed`
nor the OC branch is presented as a proper scoring rule; they are auditable
training signals whose empirical behavior must be measured.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Only NumPy is required at runtime.

Development gates:

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest -q
python -m build
```

## Validation

The initial CPU artifact passes **87 focused tests** covering nested answer
extraction, numeric equivalence, unique-plurality grouping, verifier batching
and failure handling, all six calibration modes, support invariants, OC gating,
branch composition, and lambda warmup.

A separate local parity check compared this package against private trainer
commit `8b79758` over 700 randomized valid-input groups, 4,200 mode cases, 84
schedule cases, and five answer-canonicalization cases. All compared outputs
matched within `1e-6`, providing randomized evidence of parity for those
cases. The standalone kernel deliberately rejects ambiguous or non-finite
boundary inputs that the integrated trainer historically tolerated. This does
not establish exhaustive equivalence and does not validate TPU execution,
distributed training, checkpointing, or rollout engines.

## Minimal example

```python
from calibration_aware_grpo import (
    build_calibration_advantages,
    build_calibration_batch,
)

completions = [
    r"work ... \boxed{1}",
    r"work ... \boxed{1}",
    r"work ... \boxed{1}",
    r"work ... \boxed{1}",
    r"work ... \boxed{2}",
    r"work ... \boxed{3}",
    r"work ... \boxed{4}",
]

# A production caller should replace this with a real semantic verifier.
def verifier(*, gold_answers, completion_texts):
    assert gold_answers == ["1"]
    assert len(completion_texts) == 1
    return [1.0]

batch = build_calibration_batch(
    completions=completions,
    gold_answers=["1"],
    group_size=7,
    verifier=verifier,
    mode="raw_signed",
)
advantages = build_calibration_advantages(
    rewards_calibration=batch.calibration_rewards,
    group_size=7,
    mode="raw_signed",
)
```

## Evaluation contract

Matched experiments should hold the model checkpoint, dataset revision, prompt
template, train/evaluation splits, rollout count, optimizer settings, maximum
lengths, checkpoint cadence, and decoding configuration fixed. At minimum,
report exact-answer accuracy, ECE, Brier score, reliability diagrams,
correctness-conditioned self-certainty, majority correctness/confidence,
parser/tie rates, and non-finite values.

Raw non-asymmetric modes must never produce non-zero calibration advantages
outside the unique winning cluster; invariant tests enforce this property.
Verifier result cardinality must exactly match the request batch, and the
implementation fails closed on a mismatch.

## Provenance and licensing

This repository is a standalone extraction of calibration logic developed for a
private Qwen GRPO/RLIF research implementation by Jaehyeon Shin. The extracted
source lines are attributable to the same author in that repository's Git
history. No Google Tunix source, TPU trainer, model weights, checkpoints,
credentials, or private experiment outputs are vendored here.

The code in this repository is licensed under Apache License 2.0. External
models, datasets, verifiers, and training frameworks retain their own licenses.
