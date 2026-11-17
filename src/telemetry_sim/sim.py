from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimConfig
from .users import generate_user_population
from .timing import generate_visit_schedule
from .navigation import simulate_session
from .outcomes import annotate_session_outcomes, compute_return_visits
from .damage import apply_damage
from .io import write_parquet

_TRUE_EVENTS_COLUMNS = [
    "event_id", "user_id", "session_id", "true_timestamp",
    "page", "event_type", "session_depth", "converted", "revenue", "return_visit",
]


@dataclass
class SimResult:
    true_users: pd.DataFrame
    true_events: pd.DataFrame
    observed_events: pd.DataFrame
    config: SimConfig


def run_simulation(config: SimConfig | None = None) -> SimResult:
    if config is None:
        config = SimConfig()

    # Independent child RNGs: one per user + one for the damage layer
    ss = np.random.SeedSequence(config.seed)
    child_sequences = ss.spawn(config.users.n_users + 1)
    user_rngs: list[np.random.Generator] = [
        np.random.default_rng(s) for s in child_sequences[: config.users.n_users]
    ]
    damage_rng = np.random.default_rng(child_sequences[-1])

    # 1. User population (advances each user_rng by 4 draws: 3 factors + user_id bytes)
    true_users = generate_user_population(config.users, user_rngs)

    # 2. Visit schedule (continues from each user_rng's current state)
    visit_records = generate_visit_schedule(true_users, config.timing, user_rngs)

    # 3. Session simulation + outcome annotation
    all_events: list[dict] = []
    user_lookup = true_users.set_index("user_id")

    for visit in visit_records:
        row = user_lookup.loc[visit.user_id]
        user_rng = user_rngs[visit.user_idx]

        raw_events = simulate_session(
            user_id=visit.user_id,
            session_id=visit.session_id,
            start_time=visit.scheduled_start,
            engagement=float(row["engagement"]),
            price_sensitivity=float(row["price_sensitivity"]),
            brand_affinity=float(row["brand_affinity"]),
            rng=user_rng,
        )
        raw_events = annotate_session_outcomes(
            raw_events,
            price_sensitivity=float(row["price_sensitivity"]),
            rng=user_rng,
        )
        all_events.extend(raw_events)

    if all_events:
        true_events = pd.DataFrame(all_events)
        # 4. Return-visit annotation (requires full visit schedule)
        true_events = compute_return_visits(true_events, visit_records)
    else:
        true_events = pd.DataFrame(columns=_TRUE_EVENTS_COLUMNS)

    # 5. Damage layer (uses separate damage_rng for independence from user streams)
    observed_events = apply_damage(true_events, true_users, config.damage, damage_rng,
                                   expose_user_id=config.expose_user_id)

    result = SimResult(
        true_users=true_users,
        true_events=true_events,
        observed_events=observed_events,
        config=config,
    )

    if config.output_dir is not None:
        write_parquet(result, config.output_dir)

    return result
