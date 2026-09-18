from traffic_rl.environment.env import SumoEnvironment
#from traffic_rl.algorithm.vanilla_dqn import DQNAgent
from traffic_rl.algorithm.ocba_dqn import OCBADQNAgent
import matplotlib.pyplot as plt
import numpy as np

env = SumoEnvironment(
    net_file=r"C:\Traffic_project\SUMO_RL\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection.net.xml",
    route_file=r"C:\Traffic_project\SUMO_RL\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection-vhvh.rou.xml",
    use_gui=False,
    num_seconds=3600,
    yellow_time=3,
    all_red_time=2,
)

state_dim = env.observation_space.shape[0]

action_dim = env.action_space.n

agent = OCBADQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    ensemble_size=5,
    epsilon=0.1,
    eta=0.99,
    gamma=0.99,
    batch_size=64,
    lr=0.001,
    tau=0.01,
)

agent.load("ocba_dqn_model.pth")

print("Loaded previous model.")
print("Current epsilon:", agent.epsilon)

state, info = env.reset()
print("Initial state:", state)
print("Action space:", env.action_space)

start_episode = 1000
num_episodes = 1000

episode_rewards = []
episode_avg_rewards = []
episode_avg_losses = []

for episode in range( start_episode, start_episode + num_episodes ):

    state, info = env.reset()

    episode_reward = 0.0

    episode_losses = []
    
    step = 0

    while True:

        action = agent.select_action(state, training=True)

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done,)

        loss = agent.train_step()

        if loss is not None:
            episode_losses.append(loss)

        state = next_state

        episode_reward += reward

        step += 1

        if done:
            break

    avg_reward = episode_reward / step

    if len(episode_losses) > 0:
        avg_loss = np.mean(episode_losses)
    else:
        avg_loss = None

    episode_rewards.append(episode_reward)
    episode_avg_rewards.append(avg_reward)
    episode_avg_losses.append(avg_loss)

    print(
        f"Episode: {episode} | "
        f"Reward: {episode_reward:.2f} | "
        f"Avg Reward/Step: {avg_reward:.2f} | "
        f"Epsilon: {agent.epsilon:.3f} | "
        f"Steps: {step} | "
        f"Avg Loss: {avg_loss if avg_loss is not None else 'N/A'}"
    )

agent.save("ocba_dqn_model.pth")

env.close()

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("DQN Training Reward")
plt.show()