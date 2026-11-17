from .config import SimConfig, UserConfig, TimingConfig, DamageConfig
from .sim import SimResult, run_simulation

__all__ = [
    "SimConfig",
    "UserConfig",
    "TimingConfig",
    "DamageConfig",
    "SimResult",
    "run_simulation",
]


def main() -> None:
    config = SimConfig(
        users=UserConfig(n_users=1_000),
        timing=TimingConfig(n_days=30),
    )
    result = run_simulation(config)
    sessions = result.true_events.drop_duplicates("session_id")
    conversion_rate = sessions["converted"].mean()
    return_rate = sessions["return_visit"].mean()
    print(f"Users:            {len(result.true_users):,}")
    print(f"True events:      {len(result.true_events):,}")
    print(f"Observed events:  {len(result.observed_events):,}")
    print(f"Sessions:         {len(sessions):,}")
    print(f"Conversion rate:  {conversion_rate:.2%}")
    print(f"7-day return rate:{return_rate:.2%}")
