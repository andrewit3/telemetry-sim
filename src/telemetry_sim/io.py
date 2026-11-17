from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sim import SimResult


def write_parquet(result: SimResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.true_users.to_parquet(out / "true_users.parquet", index=False)
    result.true_events.to_parquet(out / "true_events.parquet", index=False)
    result.observed_events.to_parquet(out / "observed_events.parquet", index=False)
