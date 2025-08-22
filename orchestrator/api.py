# orchestrator/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from orchestrator.planner import make_hf_pipeline, make_langchain_llm, make_planner_chain, plan_subgoals

app = FastAPI()
hf = make_hf_pipeline()
llm = make_langchain_llm(hf)
chain = make_planner_chain(llm)

class PlanRequest(BaseModel):
    summary: str

@app.post("/plan")
async def plan(req: PlanRequest):
    subgoals = plan_subgoals(req.summary, chain)
    return {"subgoals": subgoals}
