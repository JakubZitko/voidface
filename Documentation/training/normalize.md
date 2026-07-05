# Per-target loss normalization

Voidface's ensemble targets naturally produce losses on wildly
different scales. A quick census across the R5.5 reference config:

| Family     | Loss  | Typical magnitude |
| ---------- | ----- | ----------------- |
| detectors  | face-present suppression      | 0.01 – 0.1 |
| recognizers| 1 + cos(f(x), f(x+δ))         | 0.5 – 2.0  |
| VAEs       | \|\|E(x+δ) - z_gray\|\|²      | 20 – 200   |
| CLIP       | 1 - cos(g(x), g(x+δ))         | 0.1 – 1.0  |
| restorers  | LPIPS(restorer(x), r(x+δ))    | 0.05 – 0.5 |

At equal per-family weight the VAE terms dominate the backpropagated
gradient by three orders of magnitude — and the detectors, which the
CEO critic flagged as the choke point of the attack pipeline, get
essentially no signal.

Per-family normalization equalizes these scales.

## Mechanism

Every :class:`CompositeLoss` instance maintains an EMA of the
per-target loss magnitude:

    ema_i(t+1) = decay · ema_i(t) + (1 - decay) · |loss_i(t)|

The backpropagated per-target loss is then

    loss_i_backprop = loss_i(t) / max(ema_i(t), 1e-6)

which is roughly O(1) regardless of the target's intrinsic scale.
Reported per-target scalars in :class:`LossBreakdown` are the RAW
pre-normalization magnitudes so users see the actual loss values
in logs.

## Enabling

In the training TOML:

    [loss]
    normalize_per_target = true
    normalization_ema_decay = 0.99

Or in Python:

    weights = LossWeights(
        targets={...},
        normalize_per_target=True,
        normalization_ema_decay=0.99,
    )

## When to enable

- ✅ Full ensemble training with mixed families (R5.5 reference).
- ✅ Anytime you have a family-scale disparity above ~10x.
- ⚠️ Single-target training — normalization is a no-op with one
  target but the EMA overhead is real (small; O(number of targets)
  scalars per step).
- ⚠️ Very short runs (< 1000 steps). The EMA takes ~1/(1-decay) steps
  to warm up; at decay=0.99 that's ~100 steps of un-normalized
  training. Reduce `normalization_ema_decay` (e.g. 0.9) for short
  experiments.

## Tuning

- **Decay:** 0.99 = warm-up over ~100 steps, follows slow drift. 0.9 =
  ~10 step warm-up, responsive to shift. 0.999 = ~1000 step warm-up,
  extremely stable. Start at 0.99.
- **Per-target weights:** With normalization enabled, per-target
  weights become **priorities** rather than **scale calibrations**.
  A recognizer at 0.5 vs a VAE at 0.3 means "we want 5 units of
  recognizer signal for every 3 units of VAE signal" regardless of
  their intrinsic scales.

## Diagnosing training-time drift

If a single target dominates the backpropagated gradient DESPITE
normalization, three common causes:

1. The EMA has not warmed up — first ~100 steps at decay=0.99 look
   like un-normalized training. Log more frequently at start.
2. The loss for that target is nearly constant (e.g. detector
   suppression already achieved) so the EMA is very small and the
   normalization inflates the residual. Expected.
3. The target's loss is negative (e.g. bilevel LPIPS enters with a
   negative sign). The `abs()` in the EMA update handles this, but
   the reported per_target value can be negative — don't compare
   raw magnitudes across sign-different targets.
