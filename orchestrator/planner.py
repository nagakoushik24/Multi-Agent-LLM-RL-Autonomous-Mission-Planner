# orchestrator/planner.py
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

import json
import re

# Create a HF text2text pipeline (flan-t5-small is compact)
def make_hf_pipeline(model_name="google/flan-t5-small", device=0):
    # device = -1 for CPU; set 0 for GPU if available
    pipe = pipeline("text2text-generation", model=model_name, device=device)
    return pipe

def make_langchain_llm(hf_pipe):
    llm = HuggingFacePipeline(pipeline=hf_pipe, model_kwargs={"max_length":256})
    return llm

# prompt template: input is the state summary
planner_template = """You are a mission planner for multiple agents in a grid world.
Given the following short environment summary, propose subgoals for each agent.
Output JSON with an array "subgoals", each item: {{"agent_id": "agent_0", "goal_type":"goto", "target":[r,c]}}.
Choose reachable cells (not obstacles) and prioritize reaching the goal quickly.

Summary:
{summary}

Return ONLY valid JSON.
"""

def make_planner_chain(llm):
    prompt = PromptTemplate(input_variables=["summary"], template=planner_template)
    chain = prompt | llm  # modern RunnableSequence
    return chain

def plan_subgoals(summary, chain):
    resp = chain.invoke({"summary": summary})
    text = resp if isinstance(resp, str) else resp.content

    try:
        js = json.loads(text)
        return js.get("subgoals", [])
    except Exception:
        # fallback regex to extract JSON
        m = re.search(r'(\{.*\})', text, re.S)
        if m:
            try:
                js = json.loads(m.group(1))
                return js.get("subgoals", [])
            except:
                pass
    return []
