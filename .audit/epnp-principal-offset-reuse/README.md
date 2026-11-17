# EPnP principal-point offset reuse

Actual upstream PR: https://github.com/pypose/pypose/pull/545

Production head: `ba24afb6b87c6240b950a3b77a18aa4e7edb4094`.
Baseline: `298e9f0862cb228b0e8851b2616e682ccdca6a32`.
Validation head: `2393ca545b88592b9f39783892e270205b127401`.
Exact-source CPU validation: https://github.com/Afloat16/pypose/actions/runs/44038162521

Both Torch 2.5.1 and 2.6.0: 26 independent baseline-oracle tests pass;
426 complete candidate-suite tests pass, 18 skip. Sixteen public comparisons
per version verify identical null bases and poses, compatible point/pixel/
intrinsics gradients, and the exact six-buffer saved-storage reduction.
At 65,536 points the reduction is 1,572,864 bytes in float32 or 3,145,728
bytes in float64. This counts distinct storage referenced by live autograd
saved tensors, not process peak memory. No wall-clock speedup is claimed.

The new pose tests and controlled public comparisons explicitly use the
existing `gelsd` beta solver. Production solver selection is unchanged.
The original default QR solver can return differing near-zero beta
coefficients for identical input. A signed-zero coefficient can trigger
the existing `sign(0)` path and an all-zero, non-finite candidate. The first
full local test therefore has 1 failure / 425 passes / 18 skips; it is
preserved, along with the original observations and subsequent isolated
oracle runs. Projective matrices, Gram matrices, eigenpairs and null bases
remain bit-identical in repeated traces. No general default-solver
bit-identical-pose claim is made.

`complete-independent-review.json.gz` includes the independent raw traces.
The `repeated-stage-trace-solver-alias` trace correctly hooks the imported
solver alias; the earlier `repeated-stage-trace` does not hook that alias,
and its unobserved `distinct_lstsq` field is not solver-constancy evidence.
Three noncontiguous nullspace gradient cases encounter the same complex
eigenvector-phase restriction on both versions, not successful gradients.

`complete-pnp-source-history.json.gz` preserves all 82 distinct sources
from 135 commits reachable from 343 upstream PR heads and 69 existing fork
branches, including renamed and intermediate versions. All retain eight
repeated principal-point subtractions. The initial failed lazy-fetch scan
is not counted; the complete `--no-renames` scan is retained in local controls.

`complete-local-controls.json.gz` includes complete scripts and raw logs,
including noisy timing explorations and the rejected scale-reassociation
prototype. The production patch only reuses `du` and `dv`.

The manifest records each archived artifact's SHA-256 and Git blob SHA.
Official upstream CI statuses are snapshots and are separate from the
successful exact-source fork validation.
