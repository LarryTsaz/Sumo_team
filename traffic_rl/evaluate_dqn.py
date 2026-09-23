from traffic_rl.environment.env import SumoEnvironment
from traffic_rl.algorithm.vanilla_dqn import DQNAgent
from traffic_rl.algorithm.ocba_dqn import OCBADQNAgent
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NETWORK_DIR = Path(__file__).resolve().parent / "nets" / "2way-single-intersection"
OCBA_MODEL_PATH = PROJECT_ROOT / "ocba_dqn_model.pth"
VANILLA_MODEL_PATH = PROJECT_ROOT / "vanilla_dqn_model.pth"

env = SumoEnvironment(
    net_file=str(NETWORK_DIR / "single-intersection.net.xml"),
    route_file=str(NETWORK_DIR / "single-intersection-vhvh.rou.xml"),
    use_gui=True,
    num_seconds=3600,
    yellow_time=3,
    all_red_time=2,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = OCBADQNAgent(state_dim=state_dim, action_dim=action_dim)
#agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
agent.load(str(OCBA_MODEL_PATH))
#agent.load(str(VANILLA_MODEL_PATH))

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
