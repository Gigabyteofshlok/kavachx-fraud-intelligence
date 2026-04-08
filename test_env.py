from environment import KavachXEnv
from grader import KavachXGrader
import random

def phase1_test():
    env = KavachXEnv(scenario_path="scenarios/easy_001.json", render_mode="human")
    obs, info = env.reset()
    for _ in range(30):
        action = {"action_type": "IGNORE"}
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render() # Keep logging minimal
        if terminated:
            print(f"Phase 1 - Episode terminated correctly on day {info['day']}")
            break

def phase2_random_agent():
    env = KavachXEnv(scenario_path="scenarios/easy_001.json", render_mode=None)
    from environment import ACTION_NAMES
    import numpy as np
    
    obs, info = env.reset()
    entities = list(env.entities.keys())
    
    for _ in range(100):
        action_type = random.choice(ACTION_NAMES)
        action = {"action_type": action_type}
        
        if action_type in ["FLAG_SUSPICIOUS", "FREEZE_ENTITY", "REQUEST_AUDIT", "CROSS_VERIFY", "FLAG_FOR_MONITORING"]:
            action["target"] = random.choice(entities)
            action["targets"] = [action["target"]]
        elif action_type == "LINK_ENTITIES":
            action["targets"] = [random.choice(entities), random.choice(entities)]
            
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Random Agent: Episode terminated on day {info['day']}")
            
            # Use grader!
            grader = KavachXGrader(env.scenario)
            action_history = env.action_history
            prediction_day = info.get("prediction_day")
            result = grader.grade(action_history, prediction_day)
            print("Random Agent Score:", result["final_score"])
            print("Grader stats:", result["stats"])
            break

if __name__ == "__main__":
    print("--- Phase 1: Environment Sanity Check ---")
    phase1_test()
    print("\n--- Phase 2: Random Agent Check ---")
    
    # run random agent 3 times to see variation
    for i in range(3):
        print(f"Run {i+1}:")
        phase2_random_agent()
        print("")
