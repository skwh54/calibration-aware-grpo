# Calibration-Aware GRPO

This repository contains two layers extracted from a larger Qwen/JAX/Tunix
research system:

1. an accelerator-independent NumPy reference kernel for calibration-aware
   grouped policy optimization; and
2. the JAX objective wiring used at the TPU learner boundary, including
   teacher-forced policy recomputation, token-level self-certainty, branch
   assembly, and the clipped multi-branch policy-gradient loss; and
3. an optional concrete Flax NNX/Tunix runtime bridge that selects the
   rollout-matching model, enters the correct mesh, handles microbatching and
   offload synchronization, and returns teacher-forced completion logits.

The full Qwen2.5-1.5B experiments ran on TPU v6e-8 with eight JAX devices. This
package does not claim to be a standalone trainer: model construction, rollout
generation, sharding, optimizer state, and checkpoint management remain Tunix
framework responsibilities.

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
- JAX self-certainty computation from teacher-forced completion logits;
- a callback boundary for recomputing those logits under the rollout policy;
- a concrete Tunix actor-recomputation adapter with vanilla/vLLM safety checks;
- JAX assembly of self-certainty, calibration, and overconfidence branches;
- the clipped three-branch policy-gradient loss used by the TPU learner;
- invariant-focused NumPy tests.

Not included:

- a model, dataset, rollout engine, optimizer, or complete GRPO trainer;
- Tunix `RLCluster`, Qwen model, vLLM, checkpoint, or sharding setup;
- broad symbolic equivalence or an automatic mathematical verifier;
- claims that CPU/JAX tests validate accelerator-specific collectives,
  checkpointing, or device placement.

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

Only NumPy is required for the reference kernel. Install the JAX integration
without the development tools with:

```bash
python -m pip install -e '.[jax]'
```

Install the concrete runtime bridge against the Tunix revision used by the
recorded TPU trainer with:

```bash
python -m pip install -e '.[tunix]'
```

Development gates:

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest -q
python -m build
```

## JAX/Tunix integration

`calibration_aware_grpo.jax_tunix` exposes the objective boundary used by the
integrated learner:

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

The callback is the framework seam. In the TPU trainer it teacher-forces the
generated completions through the same policy weights used for rollout and
returns `[sample, completion_token, vocabulary]` logits. The returned branch
fields map directly to the extended Tunix train example:
`advantages_sc`, `advantages_cal`, `advantages_oc`,
`calibration_lambda`, and `overconfidence_lambda`. The public loss helper uses
token-mean aggregation. A trainer configured for a different Tunix aggregation
mode must supply the corresponding reduction.

`calibration_aware_grpo.tunix_runtime.TunixLearnerPolicyRecompute` supplies a
concrete implementation of that callback:

```python
from calibration_aware_grpo.jax_tunix import prepare_objective_from_policy
from calibration_aware_grpo.tunix_runtime import TunixLearnerPolicyRecompute

recompute_actor_logits = TunixLearnerPolicyRecompute.from_learner(learner)
branches = prepare_objective_from_policy(
    recompute_policy=recompute_actor_logits,
    # prompt/completion tensors and objective arguments as above
)
```

For the vanilla rollout engine, the adapter recomputes under the rollout model
and resynchronizes parameters around CPU offload. For vLLM it uses the actor
model and rejects asynchronous or CPU-offloaded layouts where the recomputed
weights may not match generation. Explicit prompt and completion masks are
preserved; padding IDs are never used to reconstruct attention.

## Validation

The initial CPU artifact passes **87 focused tests** covering nested answer
extraction, numeric equivalence, unique-plurality grouping, verifier batching
and failure handling, all six calibration modes, support invariants, OC gating,
branch composition, and lambda warmup.

The JAX suite separately checks the self-certainty formula, masked sequence
reduction, the teacher-forced recomputation boundary, public-kernel branch
assembly, overconfident-wrong gating, clipped branch losses, and fail-closed
shape validation on CPU.

The concrete Tunix bridge is tested through the same narrow operations used by
Flax NNX and Tunix: prompt/completion assembly, next-token logit alignment,
microbatch concatenation, rollout-engine model selection, mesh entry, and
offload synchronization. The unit suite injects fake models and runtime
operations so it remains runnable without a TPU or installing Tunix.
The pinned optional stack was also installed in a clean environment and the
real NNX split/merge, Tunix position/mask helpers, and selective log-softmax
path completed on a tiny CPU model.

A separate local parity check compared this package against private trainer
commit `8b79758` over 700 randomized valid-input groups, 4,200 mode cases, 84
schedule cases, and five answer-canonicalization cases. All compared outputs
matched within `1e-6`, providing randomized evidence of parity for those
cases. The standalone kernel deliberately rejects ambiguous or non-finite
boundary inputs that the integrated trainer historically tolerated. This does
not establish exhaustive equivalence.

The source integration was extracted from the same trainer that completed
recorded TPU v6e-8 calibration-aware smoke runs. A TPU run of the script below
is still required before claiming that this packaged integration has been
validated on TPU.

## TPU runtime smoke

`scripts/tpu_runtime_smoke.py` checks the package on a real JAX accelerator. It
uses a small Flax NNX language model and the pinned Tunix mask, sharding,
selective-log-softmax, and model split/merge operations. It then runs
teacher-forced completion recomputation, builds the self-certainty,
calibration, and overconfidence branches, differentiates the clipped policy
loss, and applies one update.

On a TPU VM:

```bash
python -m pip install --upgrade pip
python -m pip install "jax[tpu]==0.8.1" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -e '.[tunix]'
python scripts/tpu_runtime_smoke.py --output /tmp/tpu-runtime-smoke.json
```

The script requires a TPU backend unless `--allow-non-tpu` is supplied for a
local preflight. It is small enough for a `v6e-1`; the recorded eight-device
training topology is not required for this runtime check.

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

This repository is a standalone extraction of calibration logic and objective
wiring developed for a private Qwen GRPO/RLIF research implementation by
Jaehyeon Shin. The extracted source lines are attributable to the same author
in that repository's Git history. No Google Tunix source, Qwen model code,
model weights, checkpoints, credentials, or private experiment outputs are
vendored here.

The code in this repository is licensed under Apache License 2.0. External
models, datasets, verifiers, and training frameworks retain their own licenses.
