from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CustomerType(str, Enum):
    RETAIL = "retail"
    WHOLESALE = "wholesale"
    ENTERPRISE = "enterprise"


class OrderStatus(str, Enum):
    PENDING = "pending"
    PICKING = "picking"
    PACKED = "packed"
    SHIPPED = "shipped"
    DELAYED = "delayed"
    CANCELED = "canceled"
    FAILED = "failed"


class ActionType(str, Enum):
    PICK_ITEM = "pick_item"
    PACK_ORDER = "pack_order"
    SHIP_ORDER = "ship_order"
    DELAY_ORDER = "delay_order"
    CANCEL_ORDER = "cancel_order"
    PRIORITIZE_ORDER = "prioritize_order"


class InventoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    quantity: int = Field(ge=0)
    location: str


class OrderItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    quantity: int = Field(gt=0)


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    items: List[OrderItem]
    priority: Priority
    deadline_step: int = Field(ge=0)
    customer_type: CustomerType
    status: OrderStatus = OrderStatus.PENDING
    picked_items: Dict[str, int] = Field(default_factory=dict)
    packed: bool = False
    shipping_method: Optional[str] = None
    created_step: int = Field(default=0, ge=0)
    delayed_count: int = Field(default=0, ge=0)


class ActiveOrderDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    priority: Priority
    deadline_step: int
    customer_type: CustomerType
    status: OrderStatus
    required_items: Dict[str, int]
    picked_items: Dict[str, int]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_step: int = Field(ge=0)
    active_order: Optional[ActiveOrderDetails] = None
    inventory_levels: Dict[str, int]
    pending_orders_count: int = Field(ge=0)
    deadlines: Dict[str, int]
    available_packing_slots: int = Field(ge=0)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    order_id: Optional[str] = None
    item_id: Optional[str] = None
    quantity: int = Field(default=1, gt=0)
    shipping_method: Optional[str] = None
    target_order_id: Optional[str] = None


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_reward: float
    cumulative_reward: float
    reasons: List[str] = Field(default_factory=list)


class WarehouseState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str
    current_step: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    pending_orders: List[Order]
    completed_orders: List[Order]
    delayed_orders: List[Order]
    canceled_orders: List[Order]
    failed_orders: List[Order]
    inventory: Dict[str, InventoryItem]
    packing_slots_total: int = Field(gt=0)
    packing_slots_in_use: int = Field(ge=0)
    total_errors: int = Field(ge=0)
    total_actions: int = Field(ge=0)
    cumulative_reward: float = 0.0


class State(WarehouseState):
    pass
