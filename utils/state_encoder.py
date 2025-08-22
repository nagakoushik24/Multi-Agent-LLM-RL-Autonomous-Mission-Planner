# utils/state_encoder.py
def encode_state_summary(env):
    """
    Simple textual encoding:
    "Goal at (r,c). Agent agent_0 at (r0,c0). Agent agent_1 at (r1,c1). Obstacles: number X. Steps left Y."
    Keep short so LLM can reason.
    """
    lines = []
    lines.append(f"Grid size {env.h}x{env.w}. Goal at {env.goal}.")
    for aid, pos in env.agent_pos.items():
        lines.append(f"{aid} at {pos}.")
    # a short list of obstacles count
    obstacles = int((env.grid==1).sum())
    lines.append(f"Obstacles: {obstacles}. Step {env.steps}/{env.max_steps}.")
    return " ".join(lines)
