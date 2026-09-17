from traffic_rl.environment.env import SumoEnvironment
import time
import os 

env = SumoEnvironment(
    net_file=r"C:\Traffic_project\SUMO_RL\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection.net.xml",
    route_file=r"C:\Traffic_project\SUMO_RL\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection-vhvh.rou.xml",
    use_gui=True,
    num_seconds=3600,
    yellow_time=3,
    all_red_time=2,
)

state, info = env.reset()

print("Initial state:", state)
print("Action space:", env.action_space)

for t in range(1000):
    action = env.action_space.sample()

    print(f"\nEpoch {t}")
    print("Action:", action)

    next_state, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.2)
    print("Next state:", next_state)
    print("Reward:", reward)
    print("SUMO time:", info["step"])

    state = next_state

    if terminated or truncated:
        break

env.close()