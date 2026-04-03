from __future__ import annotations

from typing import Dict, List

from .models import State


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def grade_trajectory(trajectory: List[dict], final_state: State) -> Dict[str, float]:
    """Deterministic grader for fulfillment quality, speed, efficiency, and errors."""

    total_actions = max(1, final_state.total_actions)
    completed = len(final_state.completed_orders)
    failed = len(final_state.failed_orders)
    canceled = len(final_state.canceled_orders)

    attempted = max(1, completed + failed + canceled)
    accuracy = _safe_div(completed, attempted)

    on_time_deliveries = 0
    positive_steps = 0
    negative_steps = 0
    total_penalty_magnitude = 0.0

    for step in trajectory:
        reward_payload = step.get("reward", {})
        step_reward = float(reward_payload.get("step_reward", 0.0))
        reasons = reward_payload.get("reasons", [])

        if "delivered_before_deadline" in reasons:
            on_time_deliveries += 1
        if step_reward >= 0:
            positive_steps += 1
        else:
            negative_steps += 1
            total_penalty_magnitude += abs(step_reward)

    delivery_time = _safe_div(on_time_deliveries, max(1, completed))
    efficiency = _safe_div(positive_steps, total_actions)
    error_rate = _safe_div(final_state.total_errors, total_actions)

    # Penalty regularizer discourages noisy workflows while remaining deterministic.
    normalized_penalty = min(1.0, total_penalty_magnitude / (total_actions * 30.0))

    score = (
        0.40 * accuracy
        + 0.30 * delivery_time
        + 0.20 * efficiency
        + 0.10 * (1.0 - error_rate)
        - 0.05 * normalized_penalty
    )
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "accuracy": round(accuracy, 4),
        "delivery_time": round(delivery_time, 4),
        "efficiency": round(efficiency, 4),
        "error_rate": round(error_rate, 4),
        "negative_steps": float(negative_steps),
    }
