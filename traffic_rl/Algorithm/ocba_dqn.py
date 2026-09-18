import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .vanilla_dqn import DQNNetwork, ReplayBuffer
from .Policy import Policy


class OCBADQNAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        ensemble_size=5,
        lr=0.001,
        gamma=0.99,
        epsilon=0.1,
        eta=0.99,
        batch_size=64,
        buffer_size=10000,
        tau=0.01,
        seed=0,
        device=None,
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.eta = eta
        self.batch_size = batch_size
        self.tau = tau

        self.rng = np.random.default_rng(seed)

        torch.manual_seed(seed)

        # M current Q networks
        self.policy_nets = [DQNNetwork(state_dim, action_dim).to(self.device) for _ in range(ensemble_size)]


        # M target Q networks
        self.target_nets = [DQNNetwork(state_dim, action_dim).to(self.device) for _ in range(ensemble_size)]

        for policy_net, target_net in zip(self.policy_nets, self.target_nets):
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

        # One optimizer for all current Q networks
        all_parameters = []
        for policy_net in self.policy_nets:
            all_parameters.extend(policy_net.parameters())

        self.optimizer = optim.Adam(all_parameters, lr=lr)


        # Shared replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)


        # APCS + epsilon-OCBA policy
        self.policy = Policy()
        self.learn_step = 0



    # Ensemble Q statistics
    def get_q_statistics(self, state):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        q_values = []

        with torch.no_grad():

            for policy_net in self.policy_nets:

                q = policy_net(state_tensor)

                q = q.squeeze(0).cpu().numpy()

                q_values.append(q)

        # shape = [M, action_dim]
        q_values = np.array(q_values)

        # Eq. (15)
        q_mean = np.mean(q_values, axis=0)

        # Eq. (16): paper uses 1/M
        q_variance = np.var(q_values, axis=0, ddof=0)

        return q_mean, q_variance



    # Select action
    def select_action(self, state, training=True):

        q_mean, q_variance = self.get_q_statistics(state)

        if not training:
            return int(np.argmax(q_mean))

        action = self.policy.epsilon_OCBA_policy(
            mean_of_design=q_mean,
            variance_of_design=q_variance,
            epsilon=self.epsilon,
            eta=self.eta,
            rng=self.rng,
        )

        return int(action)



    # Store transition
    def store_transition(self, state, action, reward, next_state, done):

        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done,
        )



    # Train
    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:
            return None


        # One shared minibatch for all ensemble members
        states, actions, rewards, next_states, dones = (self.replay_buffer.sample(self.batch_size))

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)


      
        # Target:
        # y_t = r_t + gamma * max_a mean_i Q_target_i(s', a)
        with torch.no_grad():

            target_q_ensemble = []

            for target_net in self.target_nets: 
                target_q_ensemble.append(target_net(next_states))

            # [M, batch_size, action_dim]
            target_q_ensemble = torch.stack(target_q_ensemble,dim=0,)

            # mean over M networks
            # [batch_size, action_dim]
            target_q_mean = target_q_ensemble.mean(dim=0)

            # max over actions
            # [batch_size, 1]
            max_next_q = target_q_mean.max(dim=1, keepdim=True,)[0]

            target_q = (rewards + self.gamma * (1.0 - dones) * max_next_q)



        # L(theta_i) for every Q network
        losses = []

        total_loss = 0.0

        for policy_net in self.policy_nets:

            current_q = policy_net(states).gather(1,actions,)

            # Paper Eq. (26): squared residual
            loss_i = nn.functional.mse_loss(current_q,target_q,)

            losses.append(loss_i.item())

            total_loss = total_loss + loss_i

        # One optimizer updates all current networks
        self.optimizer.zero_grad()

        total_loss.backward()

        self.optimizer.step()

        # Soft target update
        # theta_target = (1-tau)*theta_target + tau*theta

        self.update_target_networks()

        self.learn_step += 1

        return float(np.mean(losses))



    # Soft target update

    def update_target_networks(self):

        with torch.no_grad():

            for policy_net, target_net in zip(
                self.policy_nets,
                self.target_nets,
            ):

                for policy_param, target_param in zip(
                    policy_net.parameters(),
                    target_net.parameters(),
                ):

                    target_param.data.copy_(
                        (1.0 - self.tau) * target_param.data
                        + self.tau * policy_param.data
                    )



    # Get APCS for observation/debugging

    def get_apcs(self, state):

        q_mean, q_variance = self.get_q_statistics(state)

        apcs = self.policy.ocba.calculate_APCS(q_mean, q_variance,)

        return float(apcs)



    # Save
    def save(self, path):

        checkpoint = {
            "policy_nets": [
                net.state_dict()
                for net in self.policy_nets
            ],

            "target_nets": [
                net.state_dict()
                for net in self.target_nets
            ],

            "optimizer": self.optimizer.state_dict(),

            "epsilon": self.epsilon,
            "eta": self.eta,
            "tau": self.tau,

            "ensemble_size": self.ensemble_size,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,

            "learn_step": self.learn_step,

            "rng_state": self.rng.bit_generator.state,
        }

        torch.save(checkpoint, path)



    # Load
    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        if checkpoint["ensemble_size"] != self.ensemble_size:
            raise ValueError("Checkpoint ensemble_size 與目前設定不同")

        if checkpoint["state_dim"] != self.state_dim:
            raise ValueError("Checkpoint state_dim 與目前設定不同")

        if checkpoint["action_dim"] != self.action_dim:
            raise ValueError("Checkpoint action_dim 與目前設定不同")

        for policy_net, state_dict in zip(
            self.policy_nets,
            checkpoint["policy_nets"],
        ):
            policy_net.load_state_dict(state_dict)

        for target_net, state_dict in zip(
            self.target_nets,
            checkpoint["target_nets"],
        ):
            target_net.load_state_dict(state_dict)

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]
        self.eta = checkpoint["eta"]
        self.tau = checkpoint["tau"]
        self.learn_step = checkpoint["learn_step"]

        if "rng_state" in checkpoint:
            self.rng.bit_generator.state = checkpoint["rng_state"]