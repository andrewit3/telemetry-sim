from __future__ import annotations
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from .config import TimingConfig

SIM_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
_HOURS = np.arange(24)


@dataclass
class VisitRecord:
    user_id: str
    user_idx: int
    session_id: str
    scheduled_start: datetime


def generate_visit_schedule(
    user_df: pd.DataFrame,
    config: TimingConfig,
    user_rngs: list[np.random.Generator],
    sim_start: datetime = SIM_START,
) -> list[VisitRecord]:
    hourly_probs = np.array(config.hourly_weights, dtype=float)
    hourly_probs /= hourly_probs.sum()

    records: list[VisitRecord] = []

    for _, row in user_df.iterrows():
        idx = int(row["user_idx"])
        rng = user_rngs[idx]
        user_id: str = row["user_id"]
        engagement = float(row["engagement"])
        last_visit_day: int | None = None

        base_prob = min(config.base_visit_prob * (1.0 + engagement), 1.0)

        for day in range(config.n_days):
            visit_prob = base_prob
            if last_visit_day is not None:
                days_since = day - last_visit_day
                visit_prob = visit_prob * float(np.exp(-config.recency_decay * days_since))

            if rng.random() < visit_prob:
                hour = int(rng.choice(_HOURS, p=hourly_probs))
                minute = int(rng.integers(0, 60))
                second = int(rng.integers(0, 60))
                start_dt = sim_start + timedelta(
                    days=day, hours=hour, minutes=minute, seconds=second
                )
                session_id = str(uuid.UUID(bytes=rng.bytes(16)))
                records.append(VisitRecord(
                    user_id=user_id,
                    user_idx=idx,
                    session_id=session_id,
                    scheduled_start=start_dt,
                ))
                last_visit_day = day

    records.sort(key=lambda r: r.scheduled_start)
    return records
