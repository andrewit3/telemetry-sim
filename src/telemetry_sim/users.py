from __future__ import annotations
import uuid

import numpy as np
import pandas as pd

from .config import UserConfig


def generate_user_population(
    config: UserConfig,
    user_rngs: list[np.random.Generator],
) -> pd.DataFrame:
    rows = []
    for i, rng in enumerate(user_rngs):
        is_bot = bool(rng.random() < config.bot_fraction)

        if is_bot:
            engagement = float(np.clip(rng.normal(0.9, 0.05), 0.0, 1.0))
            price_sensitivity = float(np.clip(rng.normal(*config.price_sensitivity_dist), 0.0, 1.0))
            brand_affinity = float(np.clip(rng.normal(0.1, 0.05), 0.0, 1.0))
            bot_type: str | None = "mimicking"
        else:
            engagement = float(np.clip(rng.normal(*config.engagement_dist), 0.0, 1.0))
            price_sensitivity = float(np.clip(rng.normal(*config.price_sensitivity_dist), 0.0, 1.0))
            brand_affinity = float(np.clip(rng.normal(*config.brand_affinity_dist), 0.0, 1.0))
            bot_type = None

        user_id = str(uuid.UUID(bytes=rng.bytes(16)))

        rows.append({
            "user_id": user_id,
            "user_idx": i,
            "engagement": engagement,
            "price_sensitivity": price_sensitivity,
            "brand_affinity": brand_affinity,
            "is_bot": is_bot,
            "bot_type": bot_type,
        })

    return pd.DataFrame(rows)
