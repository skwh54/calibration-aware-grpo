# Calibration-Aware GRPO

This package contains calibration terms for grouped policy optimization and the
JAX/Tunix code that connects them to policy recomputation. It provides:

- a NumPy reference implementation for answer grouping, calibration rewards,
  and advantages;
- JAX functions for self-certainty, objective assembly, and the clipped policy
  loss; and
- a Flax NNX/Tunix adapter for teacher-forced completion logits.

It is not a complete trainer. Model construction, rollout generation, optimizer
state, checkpointing, and datasets are supplied by the surrounding training
system.

## Objective modes

For each rollout group, let `p` be the frequency of the unique most common
answer, `y` its verifier score in `[0, 1]`, and `M_i` indicate that sample `i`
has that answer. Groups with a tie for most common answer are skipped.

| Mode | Calibration signal |
| --- | --- |
| `group_norm_signed` | group-normalized `(y - p) M_i` |
| `raw_signed` | `(y - p) M_i` |
| `raw_rlcr` | `[y - (y - p)^2] M_i` |
| `raw_signed_oc` | `raw_signed` plus the overconfidence branch |
| `raw_signed_asym` | `(y - p) M_i - rho y p (1-M_i)` |
| `raw_signed_asym_oc` | asymmetric signal plus the overconfidence branch |

The overconfidence branch is

```text
A_oc_i = -1[majority wrong] M_i clip(ReLU(A_sc_i), 0, tau)
```

where `A_sc_i` is the prompt-local self-certainty advantage. These are training
signals rather than proper scoring rules; their effect must be evaluated
empirically.

## Installation

The reference implementation requires Python 3.11 or newer and NumPy.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Install only the JAX functions with:

```bash
python -m pip install -e '.[jax]'
```

Install the Flax/Tunix adapter with the recorded dependency versions:

```bash
python -m pip install -e '.[tunix]'
```

## Reference API

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

The verifier receives one representative completion for each eligible group.
Its score is reused only for completions with the same canonical answer, so it
should grade final-answer correctness rather than formatting or reasoning
style.

## JAX and Tunix

`prepare_objective_from_policy` recomputes completion logits and builds the
self-certainty, calibration, and overconfidence branches:

```python
from calibration_aware_grpo.jax_tunix import (
    compute_dual_objective_pg_loss,
    prepare_objective_from_policy,
)

branches = prepare_objective_from_policy(
    recompute_policy=recompute_actor_logits,
    prompt_tokens=prompt_tokens,
    prompt_mask=prompt_mask,
    completion_tokens=completion_tokens,
    completion_mask=completion_mask,
    completion_texts=completion_texts,
    gold_answers=gold_answers,
    group_size=7,
    verifier=verifier,
    mode="raw_signed_oc",
    calibration_lambda=0.25,
    overconfidence_lambda=0.1,
)

loss = compute_dual_objective_pg_loss(
    coef_1=unclipped_ratio,
    coef_2=clipped_ratio,
    completion_mask=completion_mask,
    branches=branches,
)
```

The recomputation callback returns logits with shape
`[sample, completion_token, vocabulary]`. The branch fields correspond to
`advantages_sc`, `advantages_cal`, `advantages_oc`, `calibration_lambda`, and
`overconfidence_lambda` in the source trainer. The loss helper uses token-mean
aggregation.

`TunixLearnerPolicyRecompute` supplies the callback from a Tunix learner:

```python
from calibration_aware_grpo.tunix_runtime import TunixLearnerPolicyRecompute

recompute_actor_logits = TunixLearnerPolicyRecompute.from_learner(learner)
```

For vanilla rollout, it uses the rollout model and synchronizes parameters when
CPU offload is enabled. For vLLM, it uses the actor model and rejects async or
CPU-offloaded configurations because recomputation might use different weights
from generation. Prompt and completion masks are passed directly; padding token
IDs are not used to infer attention.

## Validation

The repository has 99 tests covering answer extraction and normalization,
unique-plurality grouping, verifier batching, all six objective modes,
overconfidence penalties, schedules, JAX branch assembly, clipped losses,
masking, next-token alignment, microbatching, model selection, mesh entry, and
offload synchronization.

The pinned JAX 0.8.1, Flax 0.11.1, and Tunix stack was installed in a clean
environment. A small Flax NNX model completed split/merge, Tunix mask and
position construction, teacher-forced logit recomputation, and selective
log-softmax on CPU.

A separate parity check compared the reference functions with the source
trainer over 700 randomized groups, 4,200 objective-mode cases, 84 schedule
cases, and five answer-normalization cases. Compared values matched within
`1e-6`. This is not an exhaustive equivalence proof.

The published checks do not cover a complete training run, accelerator
collectives, optimizer state, or checkpoint recovery.

Development checks:

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest -q
python -m build
```

## Provenance and license

Jaehyeon Shin extracted and packaged this code from a private Qwen GRPO/RLIF
research implementation that he also developed. No Tunix source, Qwen model
code, model weights, checkpoints, credentials, private data, or experiment
outputs are included.

The repository is licensed under Apache License 2.0. External models, datasets,
verifiers, and training frameworks retain their own licenses.
