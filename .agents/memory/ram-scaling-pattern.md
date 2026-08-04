---
name: RAM-proportional constant scaling
description: How to make hardcoded RAM-based thresholds scale with machine size without changing today's behavior.
---

When a hardcoded byte threshold (buffer caps, full-load vs. chunked-processing
cutoffs, flush points) was tuned for one specific machine size, replace the
literal with `os.totalmem() * multiplier`, choosing the multiplier so it
reproduces the exact original literal at the machine size it was tuned for
(e.g. `4GB / 8GB = 0.5`).

**Why:** This gives zero regression on the current machine (bit-identical
threshold) while automatically scaling up on a bigger machine (e.g. a 16 GB
deployment) — no need to hardcode a second number or add machine-size
detection logic. Don't hardcode a new target size; derive the multiplier from
the machine the original literal was tuned for.

**How to apply:** Any time you find a raw byte/GB constant gating a
memory-related code path (queue caps, load-strategy switches, flush
thresholds), check what machine size it implies and convert it to this
pattern before assuming it needs manual retuning for a different machine.
