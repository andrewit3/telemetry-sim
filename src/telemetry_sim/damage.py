from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DamageConfig


def apply_damage(
    true_events: pd.DataFrame,
    true_users: pd.DataFrame,
    config: DamageConfig,
    rng: np.random.Generator,
    expose_user_id: bool = False,
) -> pd.DataFrame:
    base_cols = ["event_id", "cookie_id", "session_id", "observed_timestamp", "page", "event_type"]
    if expose_user_id:
        base_cols = ["event_id", "cookie_id", "user_id", "session_id", "observed_timestamp", "page", "event_type"]
    if true_events.empty:
        return pd.DataFrame(columns=base_cols)

    # Merge engagement so we can compute per-event drop probability
    df = true_events.merge(
        true_users[["user_id", "engagement"]], on="user_id", how="left"
    )

    # 1. Drop — correlated with engagement (high engagement → fewer drops)
    drop_probs = np.clip(
        config.drop_rate * (1.0 - config.drop_corr_engagement * df["engagement"].to_numpy()),
        0.0, 1.0,
    )
    keep_mask = rng.random(len(df)) >= drop_probs
    df = df[keep_mask].reset_index(drop=True)

    # 2. Duplicate — retry logic; duplicates share the original event_id
    if len(df) > 0:
        dup_mask = rng.random(len(df)) < config.duplicate_rate
        if dup_mask.any():
            dups = df[dup_mask].copy()
            offsets = rng.uniform(0.0, 5.0, int(dup_mask.sum()))
            dups["true_timestamp"] = dups["true_timestamp"] + pd.to_timedelta(offsets, unit="s")
            df = pd.concat([df, dups], ignore_index=True)

    # 3. Timestamp noise (applied before reorder so duplicates get independent noise)
    noise = rng.normal(0.0, config.timestamp_noise_seconds, len(df))
    df["observed_timestamp"] = df["true_timestamp"] + pd.to_timedelta(noise, unit="s")

    # 4. Local reorder within sessions (simulates batched send)
    df = _apply_local_shuffle(df, config.reorder_window, rng)

    # Build observed table: rename user_id → cookie_id; optionally keep user_id too
    df = df.rename(columns={"user_id": "cookie_id"})
    if expose_user_id:
        df["user_id"] = df["cookie_id"]
        keep = ["event_id", "cookie_id", "user_id", "session_id", "observed_timestamp", "page", "event_type"]
    else:
        keep = ["event_id", "cookie_id", "session_id", "observed_timestamp", "page", "event_type"]

    observed = (
        df[keep]
        .sort_values("observed_timestamp")
        .reset_index(drop=True)
    )
    return observed


def _apply_local_shuffle(
    df: pd.DataFrame,
    window: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Within each session, sort by true_timestamp, then shuffle within non-overlapping windows."""
    parts: list[pd.DataFrame] = []
    for _, session_df in df.groupby("session_id", sort=False):
        session_df = session_df.sort_values("true_timestamp").reset_index(drop=True)
        n = len(session_df)
        new_order: list[int] = []
        for start in range(0, n, window):
            end = min(start + window, n)
            chunk = list(range(start, end))
            new_order.extend(rng.permutation(chunk).tolist())
        parts.append(session_df.iloc[new_order].reset_index(drop=True))

    if not parts:
        return df.iloc[0:0].reset_index(drop=True)
    return pd.concat(parts, ignore_index=True)
