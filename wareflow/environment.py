from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import (
    Action,
    ActionType,
    ActiveOrderDetails,
    CustomerType,
    InventoryItem,
    Observation,
    Order,
    OrderItem,
    OrderStatus,
    Priority,
    Reward,
    State,
    WarehouseState,
)


@dataclass(frozen=True)
class OrderTemplate:
    order_id: str
    items: List[Tuple[str, int]]
    priority: Priority
    deadline_step: int
    customer_type: CustomerType


@dataclass(frozen=True)
class TaskSpec:
    name: str
    goal: str
    max_steps: int
    packing_slots: int
    initial_inventory: Dict[str, Tuple[int, str]]
    initial_orders: List[OrderTemplate]
    incoming_orders: List[Tuple[int, OrderTemplate]]
    restocks: List[Tuple[int, str, int]]
    dynamic_inventory: bool


class WareFlowEnv:
    """Deterministic warehouse fulfillment environment with OpenEnv-style API."""

    def __init__(self, task_name: str = "easy", seed: int = 42) -> None:
        self.base_seed = seed
        self.task_name = task_name
        self.rng = random.Random(seed)
        self.task_specs = self._build_task_specs()
        if task_name not in self.task_specs:
            raise ValueError(f"Unsupported task '{task_name}'. Expected one of: {list(self.task_specs)}")

        self._state: Optional[WarehouseState] = None
        self._incoming_queue: List[Tuple[int, OrderTemplate]] = []
        self._restock_queue: List[Tuple[int, str, int]] = []
        self._last_shipped: Optional[Tuple[str, int]] = None
        self.trajectory: List[dict] = []

    def reset(self, task_name: Optional[str] = None, seed: Optional[int] = None) -> Observation:
        if task_name is not None:
            if task_name not in self.task_specs:
                raise ValueError(f"Unsupported task '{task_name}'.")
            self.task_name = task_name

        effective_seed = self.base_seed if seed is None else seed
        task_offset = {"easy": 11, "medium": 29, "hard": 47}[self.task_name]
        self.rng = random.Random(effective_seed + task_offset)

        spec = self.task_specs[self.task_name]
        inventory = {
            item_id: InventoryItem(item_id=item_id, quantity=qty, location=location)
            for item_id, (qty, location) in spec.initial_inventory.items()
        }
        pending_orders = [self._template_to_order(order, created_step=0) for order in spec.initial_orders]

        self._state = WarehouseState(
            task_name=spec.name,
            current_step=0,
            max_steps=spec.max_steps,
            pending_orders=pending_orders,
            completed_orders=[],
            delayed_orders=[],
            canceled_orders=[],
            failed_orders=[],
            inventory=inventory,
            packing_slots_total=spec.packing_slots,
            packing_slots_in_use=0,
            total_errors=0,
            total_actions=0,
            cumulative_reward=0.0,
        )

        self._incoming_queue = sorted(spec.incoming_orders, key=lambda x: x[0])
        self._restock_queue = sorted(spec.restocks, key=lambda x: x[0])
        self._last_shipped = None
        self.trajectory = []

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._state is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        if isinstance(action, dict):
            action = Action.model_validate(action)

        reasons: List[str] = []
        step_reward = 0.0

        current_step = self._state.current_step
        self._ingest_new_orders(current_step)
        self._apply_inventory_events(current_step)

        self._state.total_actions += 1
        action_reward, action_reasons = self._apply_action(action)
        step_reward += action_reward
        reasons.extend(action_reasons)

        deadline_reward, deadline_reasons = self._apply_deadline_updates()
        step_reward += deadline_reward
        reasons.extend(deadline_reasons)

        self._state.cumulative_reward += step_reward
        reward = Reward(
            step_reward=step_reward,
            cumulative_reward=self._state.cumulative_reward,
            reasons=reasons,
        )

        self.trajectory.append(
            {
                "step": current_step,
                "action": action.model_dump(),
                "reward": reward.model_dump(),
                "pending": len(self._state.pending_orders),
                "completed": len(self._state.completed_orders),
                "failed": len(self._state.failed_orders),
                "errors": self._state.total_errors,
            }
        )

        self._state.current_step += 1
        done = self._is_done()
        obs = self._build_observation()

        info = {
            "task": self.task_name,
            "goal": self.task_specs[self.task_name].goal,
            "pending_orders": len(self._state.pending_orders),
            "completed_orders": len(self._state.completed_orders),
            "delayed_orders": len(self._state.delayed_orders),
            "failed_orders": len(self._state.failed_orders),
            "errors": self._state.total_errors,
            "available_packing_slots": self._state.packing_slots_total - self._state.packing_slots_in_use,
        }
        return obs, reward, done, info

    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("Environment must be reset() before calling state().")
        return State.model_validate(self._state.model_dump())

    def _build_task_specs(self) -> Dict[str, TaskSpec]:
        easy = TaskSpec(
            name="easy",
            goal="Fulfill a small number of retail orders without delays.",
            max_steps=30,
            packing_slots=3,
            initial_inventory={
                "SKU_A": (20, "A1"),
                "SKU_B": (20, "A2"),
                "SKU_C": (12, "B1"),
            },
            initial_orders=[
                OrderTemplate("E-1001", [("SKU_A", 2), ("SKU_B", 1)], Priority.LOW, 8, CustomerType.RETAIL),
                OrderTemplate("E-1002", [("SKU_C", 2)], Priority.MEDIUM, 10, CustomerType.RETAIL),
                OrderTemplate("E-1003", [("SKU_A", 1), ("SKU_C", 1)], Priority.MEDIUM, 12, CustomerType.WHOLESALE),
            ],
            incoming_orders=[],
            restocks=[],
            dynamic_inventory=False,
        )

        medium = TaskSpec(
            name="medium",
            goal="Manage limited inventory while meeting moderate deadlines.",
            max_steps=45,
            packing_slots=2,
            initial_inventory={
                "SKU_A": (10, "A1"),
                "SKU_B": (6, "A2"),
                "SKU_C": (8, "B1"),
                "SKU_D": (5, "C1"),
            },
            initial_orders=[
                OrderTemplate("M-2001", [("SKU_A", 3), ("SKU_B", 1)], Priority.HIGH, 10, CustomerType.ENTERPRISE),
                OrderTemplate("M-2002", [("SKU_C", 2)], Priority.MEDIUM, 12, CustomerType.RETAIL),
                OrderTemplate("M-2003", [("SKU_D", 2), ("SKU_B", 1)], Priority.MEDIUM, 14, CustomerType.WHOLESALE),
                OrderTemplate("M-2004", [("SKU_A", 2)], Priority.LOW, 18, CustomerType.RETAIL),
            ],
            incoming_orders=[
                (5, OrderTemplate("M-2005", [("SKU_C", 2), ("SKU_D", 1)], Priority.HIGH, 16, CustomerType.ENTERPRISE)),
                (9, OrderTemplate("M-2006", [("SKU_B", 2)], Priority.MEDIUM, 20, CustomerType.WHOLESALE)),
            ],
            restocks=[(8, "SKU_B", 3), (12, "SKU_D", 2)],
            dynamic_inventory=False,
        )

        hard = TaskSpec(
            name="hard",
            goal="Handle high-volume, tight-deadline fulfillment with shortages and prioritization.",
            max_steps=65,
            packing_slots=2,
            initial_inventory={
                "SKU_A": (14, "A1"),
                "SKU_B": (10, "A2"),
                "SKU_C": (9, "B1"),
                "SKU_D": (8, "C1"),
                "SKU_E": (6, "D1"),
            },
            initial_orders=[
                OrderTemplate("H-3001", [("SKU_A", 3), ("SKU_E", 1)], Priority.HIGH, 8, CustomerType.ENTERPRISE),
                OrderTemplate("H-3002", [("SKU_B", 2), ("SKU_C", 2)], Priority.MEDIUM, 10, CustomerType.WHOLESALE),
                OrderTemplate("H-3003", [("SKU_D", 2)], Priority.HIGH, 9, CustomerType.ENTERPRISE),
                OrderTemplate("H-3004", [("SKU_A", 2), ("SKU_B", 1)], Priority.LOW, 13, CustomerType.RETAIL),
                OrderTemplate("H-3005", [("SKU_E", 2)], Priority.MEDIUM, 11, CustomerType.RETAIL),
            ],
            incoming_orders=[
                (3, OrderTemplate("H-3006", [("SKU_C", 3)], Priority.HIGH, 12, CustomerType.ENTERPRISE)),
                (6, OrderTemplate("H-3007", [("SKU_B", 2), ("SKU_D", 1)], Priority.MEDIUM, 14, CustomerType.WHOLESALE)),
                (10, OrderTemplate("H-3008", [("SKU_A", 1), ("SKU_E", 1)], Priority.HIGH, 15, CustomerType.ENTERPRISE)),
                (15, OrderTemplate("H-3009", [("SKU_D", 3)], Priority.MEDIUM, 20, CustomerType.WHOLESALE)),
            ],
            restocks=[(7, "SKU_E", 2), (11, "SKU_C", 2), (16, "SKU_B", 2)],
            dynamic_inventory=True,
        )

        return {"easy": easy, "medium": medium, "hard": hard}

    def _template_to_order(self, template: OrderTemplate, created_step: int) -> Order:
        return Order(
            order_id=template.order_id,
            items=[OrderItem(item_id=item_id, quantity=qty) for item_id, qty in template.items],
            priority=template.priority,
            deadline_step=template.deadline_step,
            customer_type=template.customer_type,
            status=OrderStatus.PENDING,
            picked_items={},
            packed=False,
            shipping_method=None,
            created_step=created_step,
            delayed_count=0,
        )

    def _build_observation(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment not initialized")

        active_order = self._select_active_order()
        active_order_details = None
        if active_order is not None:
            required = {item.item_id: item.quantity for item in active_order.items}
            active_order_details = ActiveOrderDetails(
                order_id=active_order.order_id,
                priority=active_order.priority,
                deadline_step=active_order.deadline_step,
                customer_type=active_order.customer_type,
                status=active_order.status,
                required_items=required,
                picked_items=active_order.picked_items,
            )

        pending_orders = [o for o in self._state.pending_orders if o.status in {OrderStatus.PENDING, OrderStatus.PICKING, OrderStatus.PACKED, OrderStatus.DELAYED}]
        return Observation(
            current_step=self._state.current_step,
            active_order=active_order_details,
            inventory_levels={item_id: item.quantity for item_id, item in self._state.inventory.items()},
            pending_orders_count=len(pending_orders),
            deadlines={o.order_id: o.deadline_step for o in pending_orders},
            available_packing_slots=self._state.packing_slots_total - self._state.packing_slots_in_use,
        )

    def _select_active_order(self) -> Optional[Order]:
        if self._state is None:
            return None
        candidates = [
            o
            for o in self._state.pending_orders
            if o.status in {OrderStatus.PENDING, OrderStatus.PICKING, OrderStatus.PACKED, OrderStatus.DELAYED}
        ]
        if not candidates:
            return None

        priority_weight = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        candidates.sort(key=lambda o: (priority_weight[o.priority], o.deadline_step, o.created_step, o.order_id))
        return candidates[0]

    def _ingest_new_orders(self, current_step: int) -> None:
        if self._state is None:
            return

        arrived: List[Order] = []
        while self._incoming_queue and self._incoming_queue[0][0] <= current_step:
            _, template = self._incoming_queue.pop(0)
            arrived.append(self._template_to_order(template, created_step=current_step))

        if arrived:
            self._state.pending_orders.extend(arrived)

    def _apply_inventory_events(self, current_step: int) -> None:
        if self._state is None:
            return

        while self._restock_queue and self._restock_queue[0][0] <= current_step:
            _, item_id, qty = self._restock_queue.pop(0)
            if item_id in self._state.inventory:
                self._state.inventory[item_id].quantity += qty

        spec = self.task_specs[self.task_name]
        if spec.dynamic_inventory and current_step > 0 and current_step % 4 == 0:
            impacted_items = sorted(self._state.inventory.keys())
            item_id = impacted_items[current_step % len(impacted_items)]
            current_qty = self._state.inventory[item_id].quantity
            if current_qty > 0:
                self._state.inventory[item_id].quantity -= 1

    def _apply_action(self, action: Action) -> tuple[float, List[str]]:
        if self._state is None:
            return 0.0, []

        reward = 0.0
        reasons: List[str] = []

        if action.action_type == ActionType.PICK_ITEM:
            delta, msgs = self._handle_pick_item(action)
            reward += delta
            reasons.extend(msgs)
        elif action.action_type == ActionType.PACK_ORDER:
            delta, msgs = self._handle_pack_order(action)
            reward += delta
            reasons.extend(msgs)
        elif action.action_type == ActionType.SHIP_ORDER:
            delta, msgs = self._handle_ship_order(action)
            reward += delta
            reasons.extend(msgs)
        elif action.action_type == ActionType.DELAY_ORDER:
            delta, msgs = self._handle_delay_order(action)
            reward += delta
            reasons.extend(msgs)
        elif action.action_type == ActionType.CANCEL_ORDER:
            delta, msgs = self._handle_cancel_order(action)
            reward += delta
            reasons.extend(msgs)
        elif action.action_type == ActionType.PRIORITIZE_ORDER:
            delta, msgs = self._handle_prioritize_order(action)
            reward += delta
            reasons.extend(msgs)
        else:
            reward -= 2.0
            reasons.append("unsupported_action")
            self._state.total_errors += 1

        return reward, reasons

    def _find_order(self, order_id: Optional[str]) -> Optional[Order]:
        if self._state is None or order_id is None:
            return None
        for order in self._state.pending_orders:
            if order.order_id == order_id:
                return order
        return None

    def _required_qty(self, order: Order, item_id: str) -> int:
        for item in order.items:
            if item.item_id == item_id:
                return item.quantity
        return 0

    def _handle_pick_item(self, action: Action) -> tuple[float, List[str]]:
        if self._state is None:
            return 0.0, []

        order = self._find_order(action.order_id)
        if order is None or action.item_id is None:
            self._state.total_errors += 1
            return -2.0, ["pick_missing_order_or_item"]

        if order.status in {OrderStatus.CANCELED, OrderStatus.SHIPPED, OrderStatus.FAILED}:
            self._state.total_errors += 1
            return -2.0, ["pick_invalid_order_status"]

        required = self._required_qty(order, action.item_id)
        already = order.picked_items.get(action.item_id, 0)
        if required == 0 or already >= required:
            self._state.total_errors += 1
            return -5.0, ["wrong_item_picked"]

        stock = self._state.inventory.get(action.item_id)
        if stock is None or stock.quantity < action.quantity:
            self._state.total_errors += 1
            return -5.0, ["inventory_shortage"]

        pick_qty = min(action.quantity, required - already)
        stock.quantity -= pick_qty
        order.picked_items[action.item_id] = already + pick_qty
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.PICKING

        return 10.0, ["correct_item_picked"]

    def _is_order_ready_for_pack(self, order: Order) -> bool:
        for item in order.items:
            if order.picked_items.get(item.item_id, 0) < item.quantity:
                return False
        return True

    def _handle_pack_order(self, action: Action) -> tuple[float, List[str]]:
        if self._state is None:
            return 0.0, []

        order = self._find_order(action.order_id)
        if order is None:
            self._state.total_errors += 1
            return -2.0, ["pack_missing_order"]

        if not self._is_order_ready_for_pack(order):
            self._state.total_errors += 1
            return -2.0, ["pack_before_picking_complete"]

        if self._state.packing_slots_in_use >= self._state.packing_slots_total:
            self._state.total_errors += 1
            return -2.0, ["no_packing_slot_available"]

        if order.packed:
            self._state.total_errors += 1
            return -2.0, ["already_packed"]

        order.packed = True
        order.status = OrderStatus.PACKED
        self._state.packing_slots_in_use += 1

        reward = 5.0
        reasons = ["efficient_packing"]

        packed_orders = [o for o in self._state.pending_orders if o.status == OrderStatus.PACKED]
        if len(packed_orders) >= 2:
            reward += 2.0
            reasons.append("multi_order_batching_ready")

        return reward, reasons

    def _handle_ship_order(self, action: Action) -> tuple[float, List[str]]:
        if self._state is None:
            return 0.0, []

        order = self._find_order(action.order_id)
        if order is None:
            self._state.total_errors += 1
            return -2.0, ["ship_missing_order"]

        if not order.packed:
            self._state.total_errors += 1
            return -2.0, ["ship_before_pack"]

        shipping_method = action.shipping_method or "ground"
        order.shipping_method = shipping_method
        order.status = OrderStatus.SHIPPED
        self._state.pending_orders = [o for o in self._state.pending_orders if o.order_id != order.order_id]
        self._state.completed_orders.append(order)

        if self._state.packing_slots_in_use > 0:
            self._state.packing_slots_in_use -= 1

        reward = 20.0
        reasons = ["correct_order_completed"]

        if self._state.current_step <= order.deadline_step:
            reward += 30.0
            reasons.append("delivered_before_deadline")

        if self._last_shipped and self._last_shipped[0] == shipping_method and self._state.current_step - self._last_shipped[1] <= 2:
            reward += 3.0
            reasons.append("batch_shipping_optimization")

        self._last_shipped = (shipping_method, self._state.current_step)
        return reward, reasons

    def _handle_delay_order(self, action: Action) -> tuple[float, List[str]]:
        if self._state is None:
            return 0.0, []

        order = self._find_order(action.order_id)
        if order is None:
            self._state.total_errors += 1
            return -2.0, ["delay_missing_order"]

        order.deadline_step += 1
        order.delayed_count += 1
        order.status = OrderStatus.DELAYED

        if all(existing.order_id != order.order_id for existing in self._state.delayed_orders):
            self._state.delayed_orders.append(order.model_copy(deep=True))

        return -10.0, ["delayed_order"]

    def _handle_cancel_order(self, action: Action) -> tuple[float, List[str]]:
        if self._state is None:
            return 0.0, []

        order = self._find_order(action.order_id)
        if order is None:
            self._state.total_errors += 1
            return -2.0, ["cancel_missing_order"]

        order.status = OrderStatus.CANCELED
        self._state.pending_orders = [o for o in self._state.pending_orders if o.order_id != order.order_id]
        self._state.canceled_orders.append(order)
        if order.packed and self._state.packing_slots_in_use > 0:
            self._state.packing_slots_in_use -= 1

        return -20.0, ["order_canceled_penalty"]

    def _handle_prioritize_order(self, action: Action) -> tuple[float, List[str]]:
        if self._state is None:
            return 0.0, []

        target_id = action.target_order_id or action.order_id
        order = self._find_order(target_id)
        if order is None:
            self._state.total_errors += 1
            return -2.0, ["prioritize_missing_order"]

        if order.priority != Priority.HIGH:
            order.priority = Priority.HIGH
            return 1.0, ["order_prioritized"]

        return -2.0, ["unnecessary_action"]

    def _apply_deadline_updates(self) -> tuple[float, List[str]]:
        if self._state is None:
            return 0.0, []

        reward = 0.0
        reasons: List[str] = []
        updated_pending: List[Order] = []

        for order in self._state.pending_orders:
            if order.status in {OrderStatus.SHIPPED, OrderStatus.CANCELED, OrderStatus.FAILED}:
                continue

            if self._state.current_step > order.deadline_step:
                reward -= 10.0
                reasons.append("deadline_missed")
                order.delayed_count += 1
                order.status = OrderStatus.DELAYED
                if all(existing.order_id != order.order_id for existing in self._state.delayed_orders):
                    self._state.delayed_orders.append(order.model_copy(deep=True))

                if order.delayed_count >= 3:
                    order.status = OrderStatus.FAILED
                    reward -= 20.0
                    reasons.append("failed_order")
                    self._state.failed_orders.append(order.model_copy(deep=True))
                    self._state.total_errors += 1
                    if order.packed and self._state.packing_slots_in_use > 0:
                        self._state.packing_slots_in_use -= 1
                    continue

            updated_pending.append(order)

        self._state.pending_orders = updated_pending
        return reward, reasons

    def _is_done(self) -> bool:
        if self._state is None:
            return True

        if self._state.current_step >= self._state.max_steps:
            return True

        no_pending = len(self._state.pending_orders) == 0
        no_future_orders = len(self._incoming_queue) == 0
        return no_pending and no_future_orders
