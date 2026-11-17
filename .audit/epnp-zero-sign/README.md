# EPnP zero-sign magnitude reconstruction

Actual upstream PR: https://github.com/pypose/pypose/pull/547
Prior issue: https://github.com/pypose/pypose/issues/546

Production: `3706c43d4a345baf8206499465801c4ac7051a71`.
Baseline: `298e9f0862cb228b0e8851b2616e682ccdca6a32`.
Validation: `afe339f8b74724539ae4e32a749797c4d8e6c8fb`.
Run: https://github.com/Afloat16/pypose/actions/runs/44040766108

Both CPU Torch 2.5.1 and 2.6.0 verify 76 expected failures / 12 passing
controls with the original source, and 488 complete-suite passes / 18 skips
with the exact candidate. The 88 new tests cover signed zero, all affected
direction positions, two dtypes and multiple batch axes, nonzero controls,
genuine zero magnitudes, and complete camera poses.

On each version, 64 nonzero solver-output controls preserve all candidate
values and coefficient gradients bit for bit. Eight ordinary public calls
preserve poses bit for bit and match point/pixel/intrinsics gradients.
Eight public zero-anchor replays fail on the original and recover the
known camera pose after the fix. Both complete raw job logs are preserved.

The camera regression explicitly injects a signed-zero coefficient observed
in earlier default-QR traces; its remaining solve uses the existing gelsd
driver for reproducibility. It is not a natural failure-frequency claim.
Production solver choice is unchanged. The +1 zero-direction convention
preserves a separately estimated nonzero magnitude, not a guarantee of a
unique optimal direction. Genuine all-zero estimates remain degenerate;
the existing sqrt(abs(S)) derivative at zero is not repaired or claimed
finite.

The first 44-case independent prototype review retained historical slot
mapping. Its corrected report explicitly says this is not a mathematical
oracle for the full-four coefficient ordering. Final production tests
compare that approximation's unordered squared-magnitude set; separate
baseline compatibility controls preserve the original nonzero outputs.

The order audit found a separate reversal in the Torch migration. All 14
legacy NumPy-containing sources already use the correct order, as confirmed
by explicit arithmetic replay. That restoration is excluded under the
task's strict novelty rule, is not in this patch, and is not counted as a
new improvement. The zero-sign convention remains unhandled across all 82
distinct historical PnP sources; the complete source archive is included.

The local archive retains original and final tests, raw controls, prototypes,
the corrected independent review, and the exclusion evidence. The initial
local test version included a historical full-four ordering expectation;
it was replaced before the production commit by the order-independent
magnitude check. The final exact-source fork CI is the submission gate.
