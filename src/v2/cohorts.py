"""Shared research and training-cohort selection rules."""

from __future__ import annotations

import pandas as pd

from .regular_minutes_experiment import annotate_regular_minutes


TRAINING_COHORTS = ("all", "regular")


def select_training_cohort(
    frame: pd.DataFrame,
    *,
    cohort: str,
    absolute_tolerance: float = 3.0,
    relative_tolerance: float = 0.15,
) -> pd.DataFrame:
    """Return all rows or rows whose actual minutes stayed near expectation."""
    if cohort not in TRAINING_COHORTS:
        raise ValueError(
            f"Unknown training cohort {cohort!r}; choose from {TRAINING_COHORTS}."
        )
    if cohort == "all":
        return frame.copy()
    annotated = annotate_regular_minutes(
        frame,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    selected = annotated[annotated["regular_minutes_flag"]].copy()
    if selected.empty:
        raise ValueError("The regular-minutes cohort contains no training rows.")
    return selected
