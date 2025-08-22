# run_demo.py
import time
from envs.grid_env import MultiAgentGridEnv
from agents.scripted_agent import ScriptedAgent
from utils.state_encoder import encode_state_summary
from orchestrator.planner import make_hf_pipeline, make_langchain_llm, make_planner_chain, plan_subgoals
import imageio
import os

def demo_loop(render_save_frames=True):
    env = MultiAgentGridEnv(height=11, width=11, n_agents=2, obstacle_prob=0.18, seed=42)
    obs = env.reset()
    # create scripted agents
    agents = {aid: ScriptedAgent(aid, env) for aid in env.agent_ids}
    # create planner
    hf = make_hf_pipeline(device=0)  # -1 CPU; set device=0 for GPU
    llm = make_langchain_llm(hf)
    chain = make_planner_chain(llm)

    frames = []
    done = False
    tick = 0
    while not done and tick < 80:
        summary = encode_state_summary(env)
        subgoals = plan_subgoals(summary, chain)
        # apply subgoals: assign the first subgoal per agent if available
        for sg in subgoals:
            aid = sg.get("agent_id")
            target = sg.get("target")
            if aid in agents and target:
                ok = agents[aid].assign_subgoal(target)
                print(f"[Planner] Assign {aid} -> {target} (path found={ok})")
        # if any agent had no subgoal assigned, keep them idle or assign goal cell
        for aid, agent in agents.items():
            if not agent.current_path:
                # assign goal as fallback
                agent.assign_subgoal(env.goal)
        # collect actions for this step
        actions = {}
        for aid, agent in agents.items():
            actions[aid] = agent.step()
        obs, rewards, done, _ = env.step(actions)
        print(f"Tick {tick} actions: {actions} rewards: {rewards}")
        if render_save_frames:
            frames.append(env.render())
        tick += 1
        time.sleep(0.03)
    # save frames to gif
    if frames:
        os.makedirs("demos", exist_ok=True)
        imageio.mimsave("demos/demo_run.gif", frames, duration=0.08)
        print("Saved demos/demo_run.gif")
    return

if __name__ == "__main__":
    demo_loop()
