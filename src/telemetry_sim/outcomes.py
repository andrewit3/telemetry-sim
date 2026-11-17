from __future__ import annotations
from datetime import timedelta

import numpy as np
import pandas as pd

from .timing import VisitRecord


def annotate_session_outcomes(
    events: list[dict],
    price_sensitivity: float,
    rng: np.random.Generator,
) -> list[dict]:
    converted = any(e["page"] == "CONFIRMATION" for e in events)
    session_depth = len(events)
    revenue: float | None = None

    if converted:
        # Revenue increases with low price_sensitivity (willing to pay more)
        mu = 3.0 - 2.0 * price_sensitivity
        revenue = float(rng.lognormal(mu, 0.5))

    for e in events:
        e["converted"] = converted
        e["revenue"] = revenue
        e["session_depth"] = session_depth

    return events


def compute_return_visits(
    true_events: pd.DataFrame,
    visit_records: list[VisitRecord],
    lookforward_days: int = 7,
) -> pd.DataFrame:
    # Build user → sorted list of session start times
    user_sessions: dict[str, list[datetime]] = {}
    for v in visit_records:
        user_sessions.setdefault(v.user_id, []).append(v.scheduled_start)
    for uid in user_sessions:
        user_sessions[uid].sort()

    window = timedelta(days=lookforward_days)
    session_return: dict[str, bool] = {}
    for v in visit_records:
        future = [
            s for s in user_sessions[v.user_id]
            if s > v.scheduled_start and s <= v.scheduled_start + window
        ]
        session_return[v.session_id] = len(future) > 0

    df = true_events.copy()
    df["return_visit"] = df["session_id"].map(session_return).astype(bool)
    return df
