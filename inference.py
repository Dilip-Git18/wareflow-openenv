from __future__ import annotations

import json
import os
from typing import Dict, Optional

from openai import OpenAI

from wareflow.environment import WareFlowEnv
from wareflow.grader import grade_trajectory
from wareflow.models import Action, ActionType, Observation, OrderStatus


def _heuristic_action(obs: Observation) -> Action:
    active = obs.active_order
    if active is None:
        return Action(action_type=ActionType.PRIORITIZE_ORDER, order_id=None, target_order_id=None)

    required = active.required_items
    picked = active.picked_items

    for item_id, qty in required.items():
        already = picked.get(item_id, 0)
        if already < qty:
            if obs.inventory_levels.get(item_id, 0) <= 0:
                return Action(action_type=ActionType.DELAY_ORDER, order_id=active.order_id)
            return Action(
                action_type=ActionType.PICK_ITEM,
                order_id=active.order_id,
                item_id=item_id,
                quantity=1,
            )

    if active.status != OrderStatus.PACKED:
        return Action(action_type=ActionType.PACK_ORDER, order_id=active.order_id)

    shipping_method = "express" if active.priority.value == "high" else "ground"
    return Action(
        action_type=ActionType.SHIP_ORDER,
        order_id=active.order_id,
        shipping_method=shipping_method,
    )


def _llm_action(client: OpenAI, model_name: str, obs: Observation) -> Optional[Action]:
    system = (
        "You are a warehouse fulfillment controller. "
        "Return one JSON object only with keys: action_type, order_id, item_id, quantity, shipping_method, target_order_id. "
        "Allowed action_type: pick_item, pack_order, ship_order, delay_order, cancel_order, prioritize_order."
    )
    user = {
        "observation": obs.model_dump(),
        "requirements": [
            "Prefer high-priority and earliest deadlines.",
            "Avoid unnecessary actions and cancellations.",
            "Pick correct items before packing.",
            "Ship packed orders quickly.",
        ],
    }

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
    )

    text = response.choices[0].message.content or ""
    parsed = json.loads(text)
    return Action.model_validate(parsed)


def choose_action(client: OpenAI, model_name: str, obs: Observation) -> Action:
    try:
        action = _llm_action(client, model_name, obs)
        if action is not None:
            return action
    except Exception:
        pass
    return _heuristic_action(obs)


def run_task(client: OpenAI, model_name: str, task_name: str, seed: int = 42) -> Dict[str, float]:
    env = WareFlowEnv(task_name=task_name, seed=seed)
    obs = env.reset(task_name=task_name, seed=seed)

    print(f"[START] task={task_name}")

    done = False
    while not done:
        action = choose_action(client, model_name, obs)
        obs, reward, done, info = env.step(action)
        print(
            "[STEP] "
            f"task={task_name} step={obs.current_step} action={action.action_type.value} "
            f"order={action.order_id or '-'} reward={reward.step_reward:.2f} pending={info['pending_orders']}"
        )

    final_state = env.state()
    result = grade_trajectory(env.trajectory, final_state)
    print(f"[END] task={task_name} score={result['score']:.4f}")
    return result


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("OPENAI_API_KEY")

    if not model_name:
        raise RuntimeError("MODEL_NAME is required.")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    results = {}
    for task in ["easy", "medium", "hard"]:
        results[task] = run_task(client=client, model_name=model_name, task_name=task, seed=42)

    print("Final scores:")
    for task, metrics in results.items():
        print(
            f"- {task}: score={metrics['score']:.4f}, accuracy={metrics['accuracy']:.4f}, "
            f"delivery_time={metrics['delivery_time']:.4f}, efficiency={metrics['efficiency']:.4f}, "
            f"error_rate={metrics['error_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
