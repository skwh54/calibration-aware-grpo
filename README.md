# Calibration-Aware GRPO

Calibration-aware reward terms for Group Relative Policy Optimization (GRPO),
with a NumPy reference implementation and optional JAX/Tunix integration.

The package treats answer frequency within a rollout group as an empirical
confidence estimate, then compares it with a verifier score. It focuses on the
objective logic rather than end-to-end training: model setup, rollout
generation, data loading, optimization, and checkpointing are out of scope.

## What's included

- extraction and normalization of boxed numeric answers
- unique-plurality grouping with batched verifier calls
- six calibration reward modes
- JAX helpers for self-certainty, objective assembly, and clipped policy loss
- a Flax NNX/Tunix adapter for teacher-forced policy recomputation

## Objective modes

For a rollout group, `p` is the frequency of the unique most common answer,
`y` is its verifier score in `[0, 1]`, and `M_i` indicates whether sample `i`
contains that answer. A group is skipped when its most common answer is tied.

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

where `A_sc_i` is the prompt-local self-certainty advantage. These are
optimization signals, not proper scoring rules.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional dependencies:

```bash
python -m pip install -e '.[jax]'
python -m pip install -e '.[tunix]'
```

## Basic usage

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

The verifier receives one representative completion from each eligible group.
Its score is shared only by completions with the same normalized answer, so the
verifier should judge final-answer correctness rather than formatting or
reasoning style.

## JAX and Tunix

`prepare_objective_from_policy` recomputes completion logits and assembles the
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
`[sample, completion_token, vocabulary]`. The loss helper uses token-mean
aggregation.

`TunixLearnerPolicyRecompute` creates the callback from a Tunix learner:

```python
from calibration_aware_grpo.tunix_runtime import TunixLearnerPolicyRecompute

recompute_actor_logits = TunixLearnerPolicyRecompute.from_learner(learner)
```

Vanilla rollout uses the rollout model and synchronizes parameters when CPU
offload is enabled. The vLLM path uses the actor model and rejects async or
CPU-offloaded configurations, where recomputation could use different weights
from generation. Prompt and completion masks are passed through directly.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check src tests
ruff format --check src tests
mypy src
pytest -q
python -m build
```

## License

Apache-2.0. See [LICENSE](LICENSE).
