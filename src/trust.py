"""Sample-size trust and shrinkage helpers.

NBA 30-day windows usually contain at most about 15 games. These helpers encode
that reality directly: samples below 5 are treated cautiously, samples at 10+
are trusted heavily, and samples near 15 get close to full trust.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .config import (
    HIGH_TRUST_SAMPLE_THRESHOLD,
    HIGH_TRUST_WEIGHT,
    LOW_TRUST_SAMPLE_THRESHOLD,
    LOW_TRUST_WEIGHT,
    MAX_MATCHUP_MULTIPLIER,
    MAX_SAMPLE_THRESHOLD,
    MAX_TRUST_WEIGHT,
    MIN_MATCHUP_MULTIPLIER,
)


@dataclass(frozen=True)
class MultiplierBounds:
    """Metric-specific safety ranges applied before sample-size shrinkage."""

    fga_lower: float | None = None
    fga_upper: float | None = None
    reb_chances_lower: float | None = None
    reb_chances_upper: float | None = None

    def __post_init__(self) -> None:
        for metric in ("fga", "reb_chances"):
            lower, upper = self.for_metric(metric)
            if lower is not None:
                if not np.isfinite(lower) or lower <= 0 or lower > 1:
                    raise ValueError(
                        f"{metric} lower multiplier bound must be finite and in (0, 1]."
                    )
            if upper is not None:
                if not np.isfinite(upper) or upper < 1:
                    raise ValueError(
                        f"{metric} upper multiplier bound must be finite and at least 1."
                    )
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(
                    f"{metric} lower multiplier bound cannot exceed its upper bound."
                )

    @classmethod
    def fixed(cls, lower: float, upper: float) -> "MultiplierBounds":
        """Use one safety range for both opportunity metrics."""
        return cls(
            fga_lower=lower,
            fga_upper=upper,
            reb_chances_lower=lower,
            reb_chances_upper=upper,
        )

    @classmethod
    def from_value(
        cls,
        value: "MultiplierBounds | Mapping[str, Any] | None",
    ) -> "MultiplierBounds":
        """Normalize a dataclass, flat JSON mapping, or ``None``."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("multiplier_bounds must be a MultiplierBounds, mapping, or None.")

        payload = dict(value)
        if "lower" in payload or "upper" in payload:
            lower = payload.pop("lower", None)
            upper = payload.pop("upper", None)
            payload = {
                "fga_lower": lower,
                "fga_upper": upper,
                "reb_chances_lower": lower,
                "reb_chances_upper": upper,
                **payload,
            }
        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - fields)
        if unknown:
            raise ValueError(f"Unknown multiplier-bound fields: {unknown}")
        return cls(**payload)

    def for_metric(self, metric: str) -> tuple[float | None, float | None]:
        if metric == "fga":
            return self.fga_lower, self.fga_upper
        if metric == "reb_chances":
            return self.reb_chances_lower, self.reb_chances_upper
        raise ValueError("Multiplier bounds support only 'fga' and 'reb_chances'.")

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)

    @property
    def is_unbounded(self) -> bool:
        return all(value is None for value in asdict(self).values())


DEFAULT_MULTIPLIER_BOUNDS = MultiplierBounds.fixed(
    MIN_MATCHUP_MULTIPLIER,
    MAX_MATCHUP_MULTIPLIER,
)


def bound_multiplier(
    raw_multiplier: float,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
) -> float:
    """Return a finite raw ratio clipped to the requested safety range."""
    try:
        value = float(raw_multiplier)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(value):
        return np.nan
    if lower_bound is not None:
        value = max(value, float(lower_bound))
    if upper_bound is not None:
        value = min(value, float(upper_bound))
    return value


def _smoothstep(x: float) -> float:
    """Smooth 0-to-1 curve with flat ends."""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def sample_trust_weight(
    n_samples: int,
    low_sample_threshold: int = LOW_TRUST_SAMPLE_THRESHOLD,
    high_sample_threshold: int = HIGH_TRUST_SAMPLE_THRESHOLD,
    max_sample_threshold: int = MAX_SAMPLE_THRESHOLD,
    low_trust_weight: float = LOW_TRUST_WEIGHT,
    high_trust_weight: float = HIGH_TRUST_WEIGHT,
    max_trust_weight: float = MAX_TRUST_WEIGHT,
) -> float:
    """Return how much to trust a matchup sample, from 0 to ~1.

    Defaults:
    - 0 samples: 0.00 trust
    - 5 samples: 0.35 trust
    - 10 samples: 0.80 trust
    - 15+ samples: 0.95 trust
    """
    n = max(int(n_samples), 0)
    if n == 0:
        return 0.0

    if n <= low_sample_threshold:
        progress = n / low_sample_threshold
        return low_trust_weight * _smoothstep(progress)

    if n <= high_sample_threshold:
        progress = (n - low_sample_threshold) / (high_sample_threshold - low_sample_threshold)
        return low_trust_weight + (high_trust_weight - low_trust_weight) * _smoothstep(progress)

    if n <= max_sample_threshold:
        progress = (n - high_sample_threshold) / (max_sample_threshold - high_sample_threshold)
        return high_trust_weight + (max_trust_weight - high_trust_weight) * _smoothstep(progress)

    return max_trust_weight


def effective_multiplier(
    raw_multiplier: float,
    n_samples: int,
    neutral_multiplier: float = 1.0,
    lower_bound: float | None = MIN_MATCHUP_MULTIPLIER,
    upper_bound: float | None = MAX_MATCHUP_MULTIPLIER,
) -> float:
    """Bound a raw ratio, then shrink it toward neutral based on sample trust."""
    bounded = bound_multiplier(
        raw_multiplier,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    if not np.isfinite(bounded):
        return neutral_multiplier

    trust = sample_trust_weight(n_samples)
    return trust * bounded + (1 - trust) * neutral_multiplier


def trust_label(n_samples: int) -> str:
    """Human-readable trust bucket for reports/debugging."""
    if n_samples <= 0:
        return "none"
    if n_samples < LOW_TRUST_SAMPLE_THRESHOLD:
        return "low"
    if n_samples < HIGH_TRUST_SAMPLE_THRESHOLD:
        return "medium"
    if n_samples < MAX_SAMPLE_THRESHOLD:
        return "high"
    return "max"
