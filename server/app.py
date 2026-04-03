from fastapi import FastAPI
from wareflow.environment import WareFlowEnv
from wareflow.models import Action

app = FastAPI()
env = WareFlowEnv()


def serialize(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


@app.get("/")
def root():
    return {"message": "WareFlow OpenEnv is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return serialize(obs)


@app.get("/state")
def state():
    st = env.state()
    return serialize(st)


@app.post("/step")
def step(action: dict):
    action_obj = Action(**action)
    observation, reward, done, info = env.step(action_obj)
    return {
        "observation": serialize(observation),
        "reward": serialize(reward),
        "done": done,
        "info": serialize(info),
    }