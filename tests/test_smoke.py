import pandas as pd
import pytest

from telemetry_sim import run_simulation, SimConfig, UserConfig, TimingConfig, DamageConfig


SMALL = SimConfig(
    seed=0,
    users=UserConfig(n_users=50, bot_fraction=0.1),
    timing=TimingConfig(n_days=7),
    damage=DamageConfig(drop_rate=0.1, duplicate_rate=0.05),
)


def test_output_shapes():
    r = run_simulation(SMALL)
    assert len(r.true_users) == 50
    assert len(r.true_events) > 0
    assert len(r.observed_events) > 0


def test_true_users_schema():
    r = run_simulation(SMALL)
    assert set(r.true_users.columns) >= {
        "user_id", "engagement", "price_sensitivity", "brand_affinity", "is_bot", "bot_type"
    }
    assert r.true_users["engagement"].between(0, 1).all()
    assert r.true_users["price_sensitivity"].between(0, 1).all()
    assert r.true_users["brand_affinity"].between(0, 1).all()


def test_true_events_schema():
    r = run_simulation(SMALL)
    assert set(r.true_events.columns) >= {
        "event_id", "user_id", "session_id", "true_timestamp",
        "page", "event_type", "session_depth", "converted", "revenue", "return_visit"
    }
    valid_types = {"pageview", "add_to_cart", "purchase", "bounce"}
    assert r.true_events["event_type"].isin(valid_types).all()


def test_observed_events_schema():
    r = run_simulation(SMALL)
    assert list(r.observed_events.columns) == [
        "event_id", "cookie_id", "session_id", "observed_timestamp", "page", "event_type"
    ]
    # observed must NOT contain user_id or latent factors
    assert "user_id" not in r.observed_events.columns
    assert "converted" not in r.observed_events.columns


def test_revenue_aligned_with_conversion():
    r = run_simulation(SMALL)
    converted = r.true_events[r.true_events["converted"]]
    not_converted = r.true_events[~r.true_events["converted"]]
    if len(converted) > 0:
        assert converted["revenue"].notna().all()
    assert not_converted["revenue"].isna().all()


def test_reproducibility():
    cfg = SimConfig(seed=42, users=UserConfig(n_users=100), timing=TimingConfig(n_days=5))
    r1 = run_simulation(cfg)
    r2 = run_simulation(cfg)
    pd.testing.assert_frame_equal(r1.true_users, r2.true_users)
    pd.testing.assert_frame_equal(r1.true_events, r2.true_events)
    pd.testing.assert_frame_equal(r1.observed_events, r2.observed_events)


def test_bots_are_behaviorally_distinct():
    cfg = SimConfig(seed=1, users=UserConfig(n_users=500, bot_fraction=0.3), timing=TimingConfig(n_days=5))
    r = run_simulation(cfg)
    bots = r.true_users[r.true_users["is_bot"]]
    humans = r.true_users[~r.true_users["is_bot"]]
    if len(bots) > 10 and len(humans) > 10:
        assert bots["engagement"].mean() > humans["engagement"].mean()
        assert bots["brand_affinity"].mean() < humans["brand_affinity"].mean()


def test_drop_reduces_observed_count():
    base_cfg = SimConfig(
        seed=5,
        users=UserConfig(n_users=100),
        timing=TimingConfig(n_days=10),
        damage=DamageConfig(drop_rate=0.0, duplicate_rate=0.0,
                            timestamp_noise_seconds=0.0, reorder_window=1),
    )
    damaged_cfg = SimConfig(
        seed=5,
        users=UserConfig(n_users=100),
        timing=TimingConfig(n_days=10),
        damage=DamageConfig(drop_rate=0.4, duplicate_rate=0.0),
    )
    r_clean = run_simulation(base_cfg)
    r_damaged = run_simulation(damaged_cfg)
    assert len(r_damaged.observed_events) < len(r_clean.observed_events)


def test_duplicate_increases_observed_count():
    base_cfg = SimConfig(
        seed=7,
        users=UserConfig(n_users=100),
        timing=TimingConfig(n_days=10),
        damage=DamageConfig(drop_rate=0.0, duplicate_rate=0.0),
    )
    dup_cfg = SimConfig(
        seed=7,
        users=UserConfig(n_users=100),
        timing=TimingConfig(n_days=10),
        damage=DamageConfig(drop_rate=0.0, duplicate_rate=0.3),
    )
    r_base = run_simulation(base_cfg)
    r_dup = run_simulation(dup_cfg)
    assert len(r_dup.observed_events) > len(r_base.observed_events)


def test_high_engagement_users_visit_more():
    r = run_simulation(SimConfig(seed=99, users=UserConfig(n_users=300), timing=TimingConfig(n_days=30)))
    visit_counts = r.true_events.groupby("user_id")["session_id"].nunique().reset_index()
    visit_counts.columns = ["user_id", "n_sessions"]
    merged = visit_counts.merge(r.true_users[["user_id", "engagement"]], on="user_id")
    corr = merged["n_sessions"].corr(merged["engagement"])
    assert corr > 0.1, f"Expected positive engagement-visits correlation, got {corr:.3f}"


def test_expose_user_id():
    cfg = SimConfig(seed=0, users=UserConfig(n_users=50), timing=TimingConfig(n_days=5),
                    expose_user_id=True)
    r = run_simulation(cfg)
    assert "user_id" in r.observed_events.columns
    assert "cookie_id" in r.observed_events.columns
    assert (r.observed_events["user_id"] == r.observed_events["cookie_id"]).all()

    cfg_hidden = SimConfig(seed=0, users=UserConfig(n_users=50), timing=TimingConfig(n_days=5),
                           expose_user_id=False)
    r_hidden = run_simulation(cfg_hidden)
    assert "user_id" not in r_hidden.observed_events.columns


def test_high_brand_affinity_increases_conversion():
    r = run_simulation(SimConfig(seed=77, users=UserConfig(n_users=500), timing=TimingConfig(n_days=30)))
    sessions = r.true_events.drop_duplicates("session_id")[["session_id", "user_id", "converted"]]
    merged = sessions.merge(r.true_users[["user_id", "brand_affinity"]], on="user_id")
    # Sessions from high brand-affinity users should convert more
    high = merged[merged["brand_affinity"] > 0.6]["converted"].mean()
    low = merged[merged["brand_affinity"] < 0.3]["converted"].mean()
    assert high > low, f"high-brand conversion {high:.3f} not > low-brand {low:.3f}"
