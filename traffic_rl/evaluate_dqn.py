from traffic_rl.environment.env import SumoEnvironment
from traffic_rl.algorithm.vanilla_dqn import DQNAgent
from traffic_rl.algorithm.ocba_dqn import OCBADQNAgent

env = SumoEnvironment(
    net_file=r"C:\Traffic_project\SUMO_RL\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection.net.xml",
    route_file=r"C:\Traffic_project\SUMO_RL\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection-vhvh.rou.xml",
    use_gui=True,
    num_seconds=3600,
    yellow_time=3,
    all_red_time=2,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = OCBADQNAgent(state_dim=state_dim, action_dim=action_dim)
#agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
agent.load("ocba_dqn_model.pth")
#agent.load("vanilla_dqn_model.pth")

state, info = env.reset()

total_reward = 0.0
step = 0

while True:

    action = agent.select_action(state, training=False)

    next_state, reward, terminated, truncated, info = env.step(action)

    green_time = env.traffic_signal.get_green_duration(action)

    print(
        f"Step: {step} | "
        f"Action: {action} | "
        f"Green: {green_time}s | "
        f"Reward: {reward:.2f}"
    )

    total_reward += reward
    state = next_state
    step += 1

    if terminated or truncated:
        break

print("Total reward:", total_reward)

env.close()