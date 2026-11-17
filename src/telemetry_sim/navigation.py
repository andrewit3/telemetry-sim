from __future__ import annotations
import uuid
from datetime import datetime, timedelta

import numpy as np

# ── State indices ────────────────────────────────────────────────────────────
HOME         = 0
CATEGORY     = 1
PRODUCT      = 2
CART         = 3
CHECKOUT     = 4
CONFIRMATION = 5
EXIT         = 6

STATE_NAMES: list[str] = [
    "HOME", "CATEGORY", "PRODUCT", "CART", "CHECKOUT", "CONFIRMATION", "EXIT"
]
N_STATES = 7
MAX_SESSION_STEPS = 50

STATE_EVENT_TYPES: dict[int, str] = {
    HOME:         "pageview",
    CATEGORY:     "pageview",
    PRODUCT:      "pageview",
    CART:         "add_to_cart",
    CHECKOUT:     "pageview",
    CONFIRMATION: "purchase",
}

# ── Base transition matrix ───────────────────────────────────────────────────
# Rows = from-state, cols = to-state.
# Encodes a "typical" visitor with no brand loyalty or price sensitivity.
_P_BASE = np.array([
    # HOME  CAT   PROD  CART  CHECK CONF  EXIT
    [0.05, 0.50, 0.05, 0.00, 0.00, 0.00, 0.40],  # HOME
    [0.15, 0.05, 0.55, 0.00, 0.00, 0.00, 0.25],  # CATEGORY
    [0.05, 0.30, 0.10, 0.10, 0.00, 0.00, 0.45],  # PRODUCT
    [0.05, 0.00, 0.20, 0.00, 0.55, 0.00, 0.20],  # CART
    [0.00, 0.00, 0.00, 0.10, 0.00, 0.65, 0.25],  # CHECKOUT
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # CONFIRMATION
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # EXIT (absorbing)
], dtype=float)

# ── Latent-factor perturbation matrices (additive logit scale) ───────────────
# P_user[s] = softmax(log P_base[s] + engagement*Δ_e[s] + price*Δ_p[s] + brand*Δ_b[s])
# Positive = boost that transition; negative = suppress it.

_DELTA_ENGAGEMENT = np.zeros((N_STATES, N_STATES))
_DELTA_ENGAGEMENT[PRODUCT, CART]     =  1.5   # engaged → more add-to-cart
_DELTA_ENGAGEMENT[PRODUCT, EXIT]     = -1.0   # engaged → less product bouncing
_DELTA_ENGAGEMENT[PRODUCT, CATEGORY] =  0.5   # engaged → more cross-browsing
_DELTA_ENGAGEMENT[CATEGORY, PRODUCT] =  0.5   # engaged → drills into products
_DELTA_ENGAGEMENT[CATEGORY, EXIT]    = -0.5   # engaged → less category bouncing

_DELTA_PRICE = np.zeros((N_STATES, N_STATES))
_DELTA_PRICE[CART, CHECKOUT]         = -1.5   # price-sensitive → cart abandonment
_DELTA_PRICE[CART, EXIT]             =  1.0   # price-sensitive → exits from cart
_DELTA_PRICE[CHECKOUT, CONFIRMATION] = -1.0   # price-sensitive → hesitates at checkout
_DELTA_PRICE[CHECKOUT, EXIT]         =  0.5

_DELTA_BRAND = np.zeros((N_STATES, N_STATES))
_DELTA_BRAND[CHECKOUT, CONFIRMATION] =  1.5   # brand loyalty → completes purchase
_DELTA_BRAND[CHECKOUT, EXIT]         = -1.0   # brand loyalty → less checkout abandonment
_DELTA_BRAND[CART, CHECKOUT]         =  0.5   # brand loyalty → pushes through cart


def _softmax_row(logits: np.ndarray) -> np.ndarray:
    finite = np.isfinite(logits)
    if not finite.any():
        return np.zeros_like(logits)
    shift = logits[finite].max()
    exp_vals = np.where(finite, np.exp(logits - shift), 0.0)
    total = exp_vals.sum()
    if total == 0.0:
        return np.zeros_like(logits)
    return exp_vals / total


def build_user_transition_matrix(
    engagement: float,
    price_sensitivity: float,
    brand_affinity: float,
) -> np.ndarray:
    perturbation = (
        engagement       * _DELTA_ENGAGEMENT
        + price_sensitivity * _DELTA_PRICE
        + brand_affinity    * _DELTA_BRAND
    )
    # log(0) → -inf keeps zero-probability transitions at zero after softmax
    with np.errstate(divide="ignore"):
        logits = np.where(_P_BASE > 0.0, np.log(_P_BASE), -np.inf) + perturbation

    P = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        P[s] = _softmax_row(logits[s])

    # Force absorbing states regardless of perturbation
    P[CONFIRMATION] = 0.0
    P[CONFIRMATION, EXIT] = 1.0
    P[EXIT] = 0.0
    P[EXIT, EXIT] = 1.0

    return P


def simulate_session(
    user_id: str,
    session_id: str,
    start_time: datetime,
    engagement: float,
    price_sensitivity: float,
    brand_affinity: float,
    rng: np.random.Generator,
) -> list[dict]:
    P = build_user_transition_matrix(engagement, price_sensitivity, brand_affinity)
    state_indices = np.arange(N_STATES)

    # Dwell time: lognormal with mean that scales with engagement
    mean_dwell = 15.0 + 45.0 * engagement
    dwell_sigma = 0.6
    dwell_mu = np.log(mean_dwell) - 0.5 * dwell_sigma ** 2

    state = HOME
    events: list[dict] = []
    current_time = start_time

    while state != EXIT and len(events) < MAX_SESSION_STEPS:
        events.append({
            "event_id":       str(uuid.UUID(bytes=rng.bytes(16))),
            "user_id":        user_id,
            "session_id":     session_id,
            "true_timestamp": current_time,
            "page":           STATE_NAMES[state],
            "event_type":     STATE_EVENT_TYPES[state],
        })
        dwell = float(rng.lognormal(dwell_mu, dwell_sigma))
        current_time += timedelta(seconds=dwell)
        state = int(rng.choice(state_indices, p=P[state]))

    if len(events) == 1:
        events[0]["event_type"] = "bounce"

    return events
