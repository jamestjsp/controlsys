# LPV, LTV, Sparse, and Large-Scale Model Scope

This note records the scope decision for MATLAB parity families that do not fit
the current dense LTI-first architecture without explicit design work.

## Decision Summary

| Family | Scope | Rationale |
| --- | --- | --- |
| LPV models | Later | Requires parameter-grid semantics, interpolation policy, array-aware validation, and scheduling metadata. The new `ModelArray` tracer is the right precursor, but LPV should not share the dense `System` type directly. |
| LTV models | Later | Requires time-varying state update callbacks or sampled trajectories, simulation-only semantics, and separate guarantees from LTI frequency-domain APIs. |
| Sparse first-order state-space | Later | Valuable for large models, but current algorithms assume dense `mat.Dense`, dense LAPACK, and dense transfer/frequency conversion. |
| Sparse second-order models | Out of scope for now | Needs a second-order mechanical model abstraction and sparse solvers before public API exposure. |
| Large-scale reduction workflows | Later | Should follow sparse first-order support and must not overload existing dense `Balred`, `Modred`, or `Stabsep` semantics. |

## API Boundaries

Dense-only APIs should remain dense-only until sparse implementations have
separate types and benchmark gates:

- `System`, `New`, `NewDescriptor`, `TransferFunction`, `FreqResponse`, `Balred`, `Modred`, `Stabsep`, `Systune`, and Riccati/synthesis helpers.
- Dense delay and internal-delay APIs should not accept sparse models without a separate delay representation review.

Future extension points:

- `LPVModel` built on explicit scheduling variables and gridded `ModelArray` storage.
- `LTVModel` with sampled or callback-based state matrices and simulation-first APIs.
- `SparseSystem` using sparse matrices and sparse-compatible analysis/reduction methods.
- `LargeScaleReductionOptions` separate from dense reduction options.

## Required Performance Gates Before Implementation

No LPV, LTV, sparse, or large-scale implementation PR should start until these
targets are accepted in a follow-up PRD:

- LPV: evaluate a 10x10 scheduling grid of 20-state, 2x2 dense models with less
  than 2x memory overhead versus storing the underlying model array, and with
  interpolation overhead below 25 percent versus direct model selection.
- LTV: simulate 10,000 time steps of a 20-state, 2-input, 2-output sampled LTV
  model with less than 30 percent overhead versus a direct hand-written loop.
- Sparse first-order: frequency response of a 10,000-state sparse model with
  under 1 GB peak RSS and at least 5x lower memory than dense materialization.
- Sparse second-order: no implementation until second-order model equations,
  units/physical semantics, and sparse solver requirements are specified.
- Large-scale reduction: reduce a 5,000-state sparse stable model without dense
  matrix materialization and with explicit error diagnostics.

## Follow-Up Work

Create separate PRDs for LPV and sparse first-order support after representative
fixtures and benchmarks exist. Keep LTV and large-scale reduction as research
items until the sparse model boundary is proven.
