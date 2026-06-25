"""Input shaping for the network — step 3 of feature analysis.

This is a DIFFERENT step from the regime-division of `boba.research.screening` (dividing a feature by
its yardstick `σ_ev`/`λ_ev` so it means the same thing in any market). Here we reshape the *already
regime-divided* feature for the neural network's input layer — roughly centred, unit-scale, and with no
wild outliers — and pick the LIGHTEST transform that achieves it.

  `shaping_report(feature)` -> a `ShapingReport` holding:
    - the raw distribution stats (`std`, `skew`, `excess_kurt`, `max_abs`);
    - the candidate transforms `{name -> transformed vector}` (on the finite subset);
    - per-candidate diagnostics `{name -> {excess_kurt, max_abs}}`;
    - `recommended` — the lightest candidate whose `max_abs <= outlier_bar`.

The four candidates, in increasing "weight" (how much of the feature's own shape they discard):

  1. z-score        `(f − mean) / std`                       — a plain affine rescale, keeps the shape.
  2. robust + clip  `clip(rz, −clip, +clip)`, `rz=(f−median)/(1.4826·MAD)` — robust centre/scale, then
                                                               hard-clip the tails (caps the outliers).
  3. arcsinh        `arcsinh(rz)`                             — a smooth, sign-preserving tail-compressor.
  4. rank-Gaussian  `norm.ppf((rank − 0.5) / n)`             — discards everything but the ordering.

The conclusion (and the recommendation here) is to take the lightest transform that meets the "no wild
outliers" bar — `max_abs <= outlier_bar`. The z-score keeps the shape but does nothing to a fat tail, so
a feature with a big spike fails the bar under it and the recommendation falls through to robust+clip
(whose `max_abs` is exactly `clip` by construction). The heavier arcsinh / rank-Gaussian transforms
flatten the tails further but throw information away, so they are chosen only when even the clip can't tame
the distribution (which, with a finite `clip`, never happens on `max_abs` — they exist for the report /
the QQ-plot, and as the fall-through when a caller raises the bar above `clip`).

Plotting (the histogram + QQ-vs-N(0,1)) stays in the notebook; this engine returns only the numbers.

All statistics come from `scipy.stats` (`skew`, `kurtosis`, `rankdata`, `norm`) — the same primitives the
monolith used — computed on the finite subset of the feature (non-finite rows are masked out first, as the
screening gates do). Feature-agnostic: `feature` is any 1-D float array.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew

# Lightest -> heaviest. The recommendation walks this order and takes the FIRST candidate that meets the
# outlier bar, so a lighter transform is always preferred when it already passes.
LIGHTNESS_ORDER: tuple[str, ...] = ("z-score", "robust+clip", "arcsinh", "rank-Gaussian")


@dataclass
class ShapingReport:
    """The numbers behind the input-shaping choice for one feature (the plot stays in the notebook).

    raw_std / raw_skew / raw_excess_kurt / raw_max_abs
                  -- the raw feature's spread, asymmetry, tailedness (0 = normal), and worst |value|,
                     all on the finite subset. `raw_max_abs` is what the z-score's `max_abs` reflects
                     once centred/scaled — a big spike here is what fails the z-score at the outlier bar.
    candidates    -- `{name -> transformed vector}` on the finite subset, one per `LIGHTNESS_ORDER` name.
    diagnostics   -- `{name -> {"excess_kurt": ..., "max_abs": ...}}`, the per-candidate printout.
    recommended   -- the LIGHTEST candidate name whose `max_abs <= outlier_bar`; falls through the
                     lightness order. `None` only if no candidate meets the bar (cannot happen while a
                     finite `clip <= outlier_bar`, since robust+clip's `max_abs == clip`).
    clip          -- the symmetric clip used for robust+clip (`±clip`).
    outlier_bar   -- the "no wild outliers" threshold the recommendation is taken against.
    n_finite      -- number of finite rows the report was computed on.
    """

    raw_std: float
    raw_skew: float
    raw_excess_kurt: float
    raw_max_abs: float
    candidates: dict[str, np.ndarray]
    diagnostics: dict[str, dict[str, float]]
    recommended: str | None
    clip: float
    outlier_bar: float
    n_finite: int

    def __str__(self) -> str:
        lines = [
            f"shaping report  (n={self.n_finite:,}, bar |·| <= {self.outlier_bar:g})",
            f"  raw   std={self.raw_std:.3g}  skew={self.raw_skew:+.2f}  "
            f"excess_kurt={self.raw_excess_kurt:.1f}  max|·|={self.raw_max_abs:.1f}  (0 = normal)",
        ]
        for name in LIGHTNESS_ORDER:
            d = self.diagnostics[name]
            mark = "  <- recommended" if name == self.recommended else ""
            lines.append(f"  {name:14} excess_kurt={d['excess_kurt']:>7.1f}   "
                         f"max|·|={d['max_abs']:>6.1f}{mark}")
        return "\n".join(lines)


def shaping_report(feature: np.ndarray, *, clip: float = 4.0, outlier_bar: float = 5.0) -> ShapingReport:
    """Build the four candidate input-shaping transforms for `feature` and recommend the lightest that
    clears the "no wild outliers" bar.

    Masks `feature` to its finite subset first (non-finite rows carry no shape), then computes — exactly
    as the monolith's §8 does — the raw `std`/`skew`/`excess_kurt`/`max_abs`, the four candidate vectors,
    and each candidate's `excess_kurt` + `max_abs`. `recommended` is the first name in `LIGHTNESS_ORDER`
    whose `max_abs <= outlier_bar` (z-score < robust+clip < arcsinh < rank-Gaussian) — so a near-normal
    feature gets a plain z-score, while one with a fat tail / big spike falls through to robust+clip.

      z-score        (f − mean) / std
      robust+clip    clip(rz, −clip, +clip),  rz = (f − median) / (1.4826 · MAD)
      arcsinh        arcsinh(rz)
      rank-Gaussian  norm.ppf((rank − 0.5) / n)   (average-rank ties)

    `clip` is the symmetric robust-z clip (`±clip`); `outlier_bar` is the recommendation threshold
    (default `5.0` σ — a clean gaussian's worst |z| sits just under it, so a tidy feature still earns the
    light z-score). The `kurtosis` is Fisher (excess: 0 = normal); ties use average ranks (matching the
    screening gates) so a feature with mass at one value doesn't get a spurious rank spread.
    """
    feature = np.asarray(feature, dtype=float)
    f = feature[np.isfinite(feature)]
    n = f.size
    if n == 0:
        raise ValueError("feature has no finite values to shape")

    mean, std = float(f.mean()), float(f.std())
    median = float(np.median(f))
    mad = 1.4826 * float(np.median(np.abs(f - median)))
    # robust z-score; guard a degenerate (constant / all-equal) feature so we return zeros, not nan/inf.
    rz = (f - median) / mad if mad > 0 else np.zeros_like(f)
    z = (f - mean) / std if std > 0 else np.zeros_like(f)

    candidates: dict[str, np.ndarray] = {
        "z-score": z,
        "robust+clip": np.clip(rz, -clip, clip),
        "arcsinh": np.arcsinh(rz),
        "rank-Gaussian": norm.ppf((rankdata(f, method="average") - 0.5) / n),
    }

    diagnostics: dict[str, dict[str, float]] = {
        name: {"excess_kurt": float(kurtosis(v)), "max_abs": float(np.abs(v).max())}
        for name, v in candidates.items()
    }

    recommended: str | None = next(
        (name for name in LIGHTNESS_ORDER if diagnostics[name]["max_abs"] <= outlier_bar), None
    )

    return ShapingReport(
        raw_std=std,
        raw_skew=float(skew(f)),
        raw_excess_kurt=float(kurtosis(f)),
        raw_max_abs=float(np.abs(f).max()),
        candidates=candidates,
        diagnostics=diagnostics,
        recommended=recommended,
        clip=float(clip),
        outlier_bar=float(outlier_bar),
        n_finite=n,
    )
