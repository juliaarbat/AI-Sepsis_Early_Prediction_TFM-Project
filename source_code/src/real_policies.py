from __future__ import annotations

import pandas as pd


REAL_START_DATE_DEFAULT = "2026-01-01"

REAL_READMITTED_2026 = "real_readmitted_2026"
REAL_NEW_2026 = "real_new_2026"
REAL_ALL_2026 = "real_all_2026"

REAL_POLICIES = (
    {
        "key": REAL_READMITTED_2026,
        "label": "Real 2026 - readmitted patients",
        "short_label": "Readmitted",
        "description": (
            "Patients with history before real_start_date and activity from "
            "real_start_date onward. The full patient is reserved to avoid "
            "sharing with development, and evaluation is filtered to rows "
            "from real_start_date onward."
        ),
    },
    {
        "key": REAL_NEW_2026,
        "label": "Real 2026 - new patients",
        "short_label": "New",
        "description": (
            "Patients whose first observed activity starts at real_start_date or later. "
            "This measures generalization to patients without previous history in the real period."
        ),
    },
    {
        "key": REAL_ALL_2026,
        "label": "Real 2026 - all patients",
        "short_label": "All",
        "description": (
            "Combination of readmitted patients and new patients with activity from "
            "real_start_date onward. Equivalent to reserving all patients with "
            "activity in the real period."
        ),
    },
)

REAL_POLICY_ALIASES = {
    REAL_READMITTED_2026: REAL_READMITTED_2026,
    REAL_NEW_2026: REAL_NEW_2026,
    REAL_ALL_2026: REAL_ALL_2026,
}


def normalize_real_policy(policy: str) -> str:
    """Normalize canonical real-policy names."""
    value = str(policy).lower().strip()
    if value not in REAL_POLICY_ALIASES:
        raise ValueError(
            "real_overlap_policy must be one of: "
            f"{', '.join(real_policy_keys())}."
        )
    return REAL_POLICY_ALIASES[value]


def real_policy_keys() -> tuple[str, ...]:
    """Return the canonical policy keys in execution order."""
    return tuple(str(policy["key"]) for policy in REAL_POLICIES)


def real_policy_labels(short: bool = False) -> dict[str, str]:
    """Return user-facing labels keyed by canonical policy."""
    field = "short_label" if short else "label"
    return {str(policy["key"]): str(policy[field]) for policy in REAL_POLICIES}


def select_real_units(
    units: pd.DataFrame,
    policy: str,
    real_start: pd.Timestamp | None,
    start_date_col: str,
    end_date_col: str,
) -> pd.Series:
    """Return the boolean mask of units reserved for the real split."""
    if real_start is None:
        return pd.Series(False, index=units.index)

    policy = normalize_real_policy(policy)
    start_date = pd.to_datetime(units[start_date_col], errors="coerce")
    end_date = pd.to_datetime(units[end_date_col], errors="coerce")
    has_previous_history = start_date < real_start
    has_real_activity = end_date >= real_start
    starts_in_real = start_date >= real_start

    if policy == REAL_READMITTED_2026:
        mask = has_previous_history & has_real_activity
    elif policy == REAL_NEW_2026:
        mask = starts_in_real
    elif policy == REAL_ALL_2026:
        mask = has_real_activity
    else:  # pragma: no cover - normalize_real_policy prevents this branch.
        raise ValueError(f"Unknown real policy: {policy}")
    return mask.fillna(False)


def select_excluded_real_units(
    units: pd.DataFrame,
    policy: str,
    real_start: pd.Timestamp | None,
    start_date_col: str,
    end_date_col: str,
) -> pd.Series:
    """Return units that should be excluded from both development and real."""
    if real_start is None:
        return pd.Series(False, index=units.index)

    policy = normalize_real_policy(policy)
    start_date = pd.to_datetime(units[start_date_col], errors="coerce")
    end_date = pd.to_datetime(units[end_date_col], errors="coerce")

    if policy == REAL_READMITTED_2026:
        # In this analysis the real split contains readmissions only; new patients
        # from 2026 are fully excluded to avoid contaminating development.
        mask = start_date >= real_start
    else:
        mask = pd.Series(False, index=units.index)
    return (mask & end_date.notna()).fillna(False)


