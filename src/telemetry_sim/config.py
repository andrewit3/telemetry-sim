from __future__ import annotations
from dataclasses import dataclass, field

# Bimodal hourly arrival weights: morning peak ~10am, evening peak ~8pm (len=24)
_DEFAULT_HOURLY_WEIGHTS: tuple[float, ...] = (
    0.5, 0.3, 0.2, 0.2, 0.2, 0.4,   # 0–5am
    1.0, 2.0, 3.5, 4.5, 5.0, 4.5,   # 6–11am
    4.0, 3.5, 3.0, 3.0, 3.5, 4.0,   # 12–5pm
    5.0, 5.5, 5.0, 4.0, 3.0, 2.0,   # 6–11pm
)


@dataclass(frozen=True)
class UserConfig:
    n_users: int = 5_000
    bot_fraction: float = 0.05
    # (mean, std) for each latent factor; drawn from clipped normal → [0, 1]
    engagement_dist: tuple[float, float] = (0.5, 0.2)
    price_sensitivity_dist: tuple[float, float] = (0.5, 0.2)
    brand_affinity_dist: tuple[float, float] = (0.3, 0.2)


@dataclass(frozen=True)
class TimingConfig:
    n_days: int = 30
    hourly_weights: tuple[float, ...] = _DEFAULT_HOURLY_WEIGHTS
    base_visit_prob: float = 0.15
    # visit_prob *= exp(-recency_decay * days_since_last_visit)
    recency_decay: float = 0.1


@dataclass(frozen=True)
class DamageConfig:
    drop_rate: float = 0.05
    # higher engagement → less dropping: drop_prob = drop_rate * (1 - corr * engagement)
    drop_corr_engagement: float = 0.3
    duplicate_rate: float = 0.02
    timestamp_noise_seconds: float = 30.0
    reorder_window: int = 3


@dataclass(frozen=True)
class SimConfig:
    seed: int = 42
    users: UserConfig = field(default_factory=UserConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    damage: DamageConfig = field(default_factory=DamageConfig)
    # None → return DataFrames only; path → also write Parquet files
    output_dir: str | None = None
    # True → include user_id column in observed_events (removes identity-resolution challenge)
    expose_user_id: bool = False
