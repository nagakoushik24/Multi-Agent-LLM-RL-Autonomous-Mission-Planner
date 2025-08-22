# 🧭 Multi-Agent LLM + RL Autonomous Mission Planner

🚀 A runnable **starter project** that combines **multi-agent gridworld environments** with a **LangChain + Hugging Face Transformers planner** (no OpenAI API required).  
Agents navigate using **A\*-style scripted policies** (deterministic & explainable) while the **LLM planner** proposes subgoals. Skeleton code for RL (Stable-Baselines3 PPO) is also included.

---

## 📂 Repository Structure
```
maze-marl-llm/
├─ README.md
├─ requirements.txt
├─ run_demo.py # Run a full local demo (agents + planner + env)
├─ docker-compose.yml 
├─ envs/
│ └─ grid_env.py  
├─ agents/
│ ├─ scripted_agent.py  
│ └─ train_ppo.py  
├─ orchestrator/
│ ├─ planner.py # LangChain + HuggingFace planner
│ └─ api.py  
└─ utils/
├─ state_encoder.py # Convert env state → textual summary
└─ a_star.py # A* pathfinding helper
```

### What this Prject does:
- Generate a random maze
- Spawn 2 scripted agents
- Use an LLM planner (flan-t5-small) via LangChain to assign subgoals
- Agents execute subgoals with A*-style pathfinding
- Console shows planner decisions + agent moves
- Saves a tiny demo GIF → demos/demo_run.gif

### Run demo
```
python run_demo.py
```

## Sample output

![Demo Run](demo_run.gif)


