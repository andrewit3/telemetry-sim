# telemetry-sim — Design Specification

## Purpose

A Python library that simulates website telemetry with a fully observable ground-truth data-generating process (DGP). The primary audience is causal inference (CI) researchers who need labeled datasets where the true confounding structure is known, so they can benchmark estimators against ground truth.

**Core promise:** every random draw is traceable from a latent user trait through the navigation model, through the damage layer, to the observed log. The gap between the two output tables *is* the research problem.

---

## v1 Scope

| Feature | v1 | Phase 2 |
|---|---|---|
| Continuous latent user factors | ✓ | |
| E-commerce Markov chain | ✓ | Programmatic graph builder |
| Realistic visit timing | ✓ | |
| All four outcome types | ✓ | |
| Dropped / duplicated / noisy-timestamp events | ✓ | |
| Bots (mimicking) | ✓ | |
| Parquet output | ✓ | |
| Per-user RNG seeding | ✓ | |
| Causal DAG spec + confounder injection API | | ✓ |
| Treatment assignment | | ✓ |
| Cookie resets / identity fragmentation | | ✓ |
| Programmatic page graph builder | | ✓ |
| YAML config serialization | | ✓ |

---

## Data Model

### Ground-truth tables (never observed by the analyst)

**`true_users`** — one row per user

| column | type | description |
|---|---|---|
| `user_id` | str (UUID) | stable identity |
| `engagement` | float [0,1] | propensity to browse, return, go deep |
| `price_sensitivity` | float [0,1] | low = willing to pay; high = price-driven |
| `brand_affinity` | float [0,1] | directly increases conversion probability |
| `is_bot` | bool | drawn from `bot_fraction` config |
| `bot_type` | str \| None | `"mimicking"` or None |

**`true_events`** — one row per raw event as it was fired

| column | type | description |
|---|---|---|
| `event_id` | str (UUID) | |
| `user_id` | str | links to `true_users` |
| `session_id` | str | stable per visit |
| `true_timestamp` | datetime (UTC) | exact firing time |
| `page` | str | state in the Markov chain |
| `event_type` | str | `pageview`, `add_to_cart`, `purchase`, `bounce` |
| `session_depth` | int | page index within session |
| `converted` | bool | did purchase happen this session |
| `revenue` | float \| None | order value if converted |

### Observed tables (what the analyst sees)

**`observed_events`** — derived from `true_events` after damage

| column | type | description |
|---|---|---|
| `event_id` | str | may be duplicated; original or synthetic |
| `cookie_id` | str | = `user_id` in v1 (fragmented in Phase 2) |
| `session_id` | str | |
| `observed_timestamp` | datetime | noisy; may be reordered |
| `page` | str | |
| `event_type` | str | |

Rows dropped by the damage layer are absent. Duplicated rows share the original `event_id`. There is no `user_id`, no latent factors, no `converted` label — the analyst must derive these.

---

## Module Architecture

```
src/telemetry_sim/
├── __init__.py          # public API: run_simulation(), SimConfig
├── config.py            # all dataclasses
├── users.py             # generate_user_population()
├── timing.py            # visit schedule + arrival rate model
├── navigation.py        # Markov chain + default e-commerce graph
├── outcomes.py          # conversion, revenue, engagement, return-visit
├── damage.py            # drop / duplicate / timestamp-noise transforms
├── io.py                # write_parquet(), (future: write_sqlite(), etc.)
└── sim.py               # run_simulation() orchestrator
```

---

## Configuration

All parameters live in a hierarchy of frozen dataclasses. No YAML required for v1 — the dataclasses are the config.

```python
@dataclass(frozen=True)
class UserConfig:
    n_users: int = 5_000
    bot_fraction: float = 0.05
    # Factor distributions — (mean, std) for Beta-transformed normals
    engagement_dist: tuple[float, float] = (0.5, 0.2)
    price_sensitivity_dist: tuple[float, float] = (0.5, 0.2)
    brand_affinity_dist: tuple[float, float] = (0.3, 0.2)

@dataclass(frozen=True)
class TimingConfig:
    n_days: int = 30
    # Relative arrival weight by hour-of-day (len=24); will be normalised
    hourly_weights: tuple[float, ...] = DEFAULT_HOURLY_WEIGHTS  # peaks at 10am, 8pm
    # Base daily visit probability (modified by engagement factor)
    base_visit_prob: float = 0.15
    # Recency decay: visit prob multiplied by exp(-recency_decay * days_since_last)
    recency_decay: float = 0.1

@dataclass(frozen=True)
class DamageConfig:
    drop_rate: float = 0.05          # fraction of events silently dropped
    drop_corr_engagement: float = 0.3  # higher engagement → less dropping (adblocker proxy)
    duplicate_rate: float = 0.02     # fraction of events duplicated
    timestamp_noise_seconds: float = 30.0  # std dev of Gaussian noise on timestamps
    reorder_window: int = 3          # max events shuffled within a window

@dataclass(frozen=True)
class SimConfig:
    seed: int = 42
    users: UserConfig = field(default_factory=UserConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    damage: DamageConfig = field(default_factory=DamageConfig)
    output_dir: str | None = None    # None = return DataFrames; path = write Parquet
```

---

## User Population Generation (`users.py`)

```python
def generate_user_population(config: UserConfig, rng: np.random.Generator) -> pd.DataFrame:
    ...
```

- Draw `n_users` rows of continuous factors. Each factor uses a clipped normal → rescaled to [0,1] (Beta-equivalent without the parameterisation complexity).
- Bot flag: Bernoulli(`bot_fraction`). Bots get the same factor structure but `engagement` is drawn from `(0.9, 0.05)` and `brand_affinity` from `(0.1, 0.05)` to make them behaviorally distinct while still using the same Markov chain.
- Each user gets a **per-user child RNG** derived from the top-level seed + user index. All subsequent generation for that user uses the child RNG exclusively. This gives per-user reproducibility without global state.

---

## Visit Timing (`timing.py`)

### Daily visit schedule

For each user × day:

1. **Daily visit probability** = `base_visit_prob × (1 + engagement_factor)`, capped at 1.
2. **Recency adjustment**: multiply by `exp(-recency_decay × days_since_last_visit)` to model habituation — users who just visited are slightly less likely to return immediately.
3. Sample whether a visit occurs (Bernoulli).

### Intra-day arrival time

If a visit occurs, sample the hour from the `hourly_weights` distribution, then sample minutes/seconds uniformly within the hour.

### Output

`generate_visit_schedule(user_df, config, rng) → list[VisitRecord]` where `VisitRecord` is a simple dataclass `(user_id, session_id, scheduled_start: datetime)`.

---

## Navigation — Markov Chain (`navigation.py`)

### Default e-commerce page graph

States: `HOME`, `CATEGORY`, `PRODUCT`, `CART`, `CHECKOUT`, `CONFIRMATION`, `EXIT`

`EXIT` is the absorbing state (session ends).

Transition matrix rows sum to 1. Base matrix hardcoded; each user's actual transition matrix is a **softmax-blended** version:

```
P_user(s → s') = softmax(log P_base(s → s') + w_engagement × Δ_engagement(s→s') + w_price × Δ_price(s→s'))
```

Where `Δ_engagement` and `Δ_price` are pre-defined perturbation matrices encoding, e.g.:
- High engagement → higher probability of `PRODUCT → CART`, lower probability of `PRODUCT → EXIT`
- High price_sensitivity → lower probability of `CART → CHECKOUT`
- High brand_affinity → higher probability of `CHECKOUT → CONFIRMATION`

This means latent traits causally determine navigation and downstream outcomes without any explicit treatment variable.

### Session simulation

```python
def simulate_session(user: UserRow, start_time: datetime, site: SiteConfig, rng) -> list[Event]:
    ...
```

Runs the Markov chain until `EXIT`, recording each state transition as a `pageview` event (or `add_to_cart` / `purchase` for specific transitions). Returns the raw event list with true timestamps (start + cumulative dwell times drawn from a lognormal parameterised by engagement).

---

## Outcomes (`outcomes.py`)

Outcomes are deterministic functions of the session trace + latent factors, with noise:

| Outcome | Formula |
|---|---|
| `converted` | `CONFIRMATION` state was reached in session |
| `revenue` | `LogNormal(μ=3 - 2×price_sensitivity, σ=0.5)` if converted |
| `session_depth` | count of non-EXIT states visited |
| `return_visit` | user has ≥1 visit in the 7 days following the session |

`return_visit` is computed after the full schedule is generated (requires cross-session lookback).

---

## Damage Layer (`damage.py`)

Applied as a pure transform: `apply_damage(true_events: pd.DataFrame, config: DamageConfig, rng) → pd.DataFrame`.

### Drop

- Each event gets a drop probability = `drop_rate × (1 - drop_corr_engagement × user_engagement)`.
- Events are dropped independently. No whole-session dropping in v1.

### Duplicate

- Each surviving event is duplicated with probability `duplicate_rate`.
- Duplicates share the original `event_id` (simulating retry deduplication problem) and get a small timestamp offset (+0 to +5 seconds).

### Timestamp noise

- Gaussian noise `N(0, timestamp_noise_seconds²)` added to each surviving event's timestamp.
- Then sort by `session_id` and apply local shuffle within windows of size `reorder_window` to simulate batched send reordering.

### Order of operations

Drop → Duplicate → Timestamp noise → Reorder. This order matters: duplicates inherit the original timestamp before noise is applied, so duplicates are not perfectly identical.

---

## Orchestrator (`sim.py`)

```python
def run_simulation(config: SimConfig = SimConfig()) -> SimResult:
    ...
```

Returns:

```python
@dataclass
class SimResult:
    true_users: pd.DataFrame
    true_events: pd.DataFrame
    observed_events: pd.DataFrame
    config: SimConfig
```

If `config.output_dir` is set, also writes three Parquet files: `true_users.parquet`, `true_events.parquet`, `observed_events.parquet`.

**Generation pipeline:**
1. Init top-level `np.random.default_rng(config.seed)`
2. `generate_user_population(config.users, rng)` → `true_users`
3. `generate_visit_schedule(true_users, config.timing, rng)` → visit list
4. For each visit, `simulate_session(...)` → raw events; collect into `true_events`
5. Compute `return_visit` for all users by looking ahead in the schedule
6. `apply_damage(true_events, config.damage, rng)` → `observed_events`
7. Return / write

---

## Public API (what goes in `__init__.py`)

```python
from telemetry_sim import run_simulation, SimConfig, UserConfig, TimingConfig, DamageConfig, SimResult
```

Minimal surface: users import the config dataclasses, call `run_simulation()`, get back a `SimResult`. Everything else is an implementation detail.

---

## Phase 2 Roadmap

1. **Causal DAG spec** — lightweight Python builder for expressing `latent → observed` relationships; allows injecting new confounders without touching simulation internals.
2. **Treatment assignment** — add a `TreatmentConfig` to `SimConfig`; treatment can be RCT or confounded (function of latent factors).
3. **Identity fragmentation** — cookie resets mid-journey; `observed_events` gets a `cookie_id` that differs from `user_id` for a configurable fraction of sessions.
4. **Programmatic page graph** — `SiteGraph` builder API; default e-commerce graph becomes one preset.
5. **YAML serialization** — `SimConfig.to_yaml()` / `SimConfig.from_yaml()` for reproducible experiment files.
