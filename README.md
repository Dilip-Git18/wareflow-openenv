# WareFlow: AI Warehouse Order Fulfillment Environment

## Overview
WareFlow is a deterministic, real-world reinforcement learning environment that simulates warehouse order fulfillment operations. The environment models inventory constraints, order priorities, packing station capacity, deadlines, delays, and shipping decisions.

This project is built for OpenEnv hackathon submission with strict typed schemas and reproducible behavior.

## Real-World Relevance
WareFlow represents operational problems that fulfillment centers solve every day:
- Picking the correct stock from shelf locations
- Managing finite packing stations
- Meeting customer deadlines under shortages
- Prioritizing high-value / enterprise orders
- Minimizing fulfillment errors and workflow inefficiencies

## OpenEnv Compliance
Implemented requirements:
- Typed Pydantic models: `Observation`, `Action`, `Reward`, `State`
- Core API: `reset()`, `step(action)`, `state()`
- `step()` return signature: `(observation, reward, done, info)`
- Included artifacts: `openenv.yaml`, `inference.py`, `Dockerfile`, `README.md`
- Deterministic behavior using fixed seed (`42`) and deterministic task schedules

## Environment Design
### Entities
- Inventory items: item id, quantity, location
- Orders: order id, line items, priority, deadline, customer type
- Warehouse state: pending/completed/delayed/canceled/failed orders, inventory, packing capacity, errors

### Step Logic
Each `step` executes this sequence:
1. New orders arrive (task schedule)
2. Inventory updates (restocks and hard-task dynamic drift)
3. Agent action applied
4. Deadlines and delays resolved
5. Dense reward issued and trajectory recorded

## Observation Space
Observation includes:
- Current step
- Active order details
- Inventory levels
- Pending order count
- Order deadlines
- Available packing slots

## Action Space
Supported actions:
- `pick_item`
- `pack_order`
- `ship_order`
- `delay_order`
- `cancel_order`
- `prioritize_order`

Action payload supports:
- `order_id`
- `item_id` (when needed)
- `quantity`
- `shipping_method`
- `target_order_id`

## Reward Design (Dense)
Primary shaping rewards:
- `+10` correct item picked
- `+20` correct order completed
- `+30` delivered before deadline
- `+5` efficient packing

Penalties:
- `-5` wrong item or shortage-driven bad pick
- `-10` delayed order
- `-20` failed order
- `-2` unnecessary/invalid action

Additional judge-friendly mechanics:
- Inventory shortage scenarios
- Multi-order batching bonus
- Batch shipping optimization bonus
- Penalties for inefficient workflow/noisy actions

## Tasks
All tasks return a normalized score `0.0` to `1.0` through the deterministic grader.

### Easy
- Few initial orders
- Simple, ample inventory
- No delayed arrivals
- Goal: finish cleanly without deadline misses

### Medium
- More orders
- Limited inventory with scheduled restock
- Moderate deadlines
- Goal: prioritize correctly while avoiding stock errors

### Hard
- Many orders
- Dynamic inventory changes + shortages
- Tight deadlines with priority pressure
- Goal: maintain high throughput with low error rate

## Deterministic Grader
Input:
- Full agent trajectory + final environment state

Output:
- `score` in `[0.0, 1.0]`
- Metrics:
  - Fulfillment accuracy
  - Delivery time (on-time ratio)
  - Efficiency (productive action ratio)
  - Error rate

The grader is deterministic and repeatable for fixed trajectories.

## Project Structure
```text
.
├── Dockerfile
├── README.md
├── inference.py
├── openenv.yaml
├── requirements.txt
└── wareflow
    ├── __init__.py
    ├── environment.py
    ├── grader.py
    └── models.py
```

## Setup
### Local
1. Create Python 3.10 environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```bash
   export API_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-4.1-mini"
   export OPENAI_API_KEY="<your_key>"
   ```
4. Run:
   ```bash
   python inference.py
   ```

### Docker
```bash
docker build -t wareflow .
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4.1-mini" \
  -e OPENAI_API_KEY="<your_key>" \
  wareflow
```

## Inference Logging Contract
`inference.py` logs execution using required tags:
- `[START]`
- `[STEP]`
- `[END]`

and prints final per-task metrics and scores.

## Notes for Hackathon Judges
- Environment is not a toy game; it models practical fulfillment operations and constraints.
- Behavior is reproducible with fixed seed and deterministic schedules.
- API and schema design are modular for easy policy training, evaluation, and extension.