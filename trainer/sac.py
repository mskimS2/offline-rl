import time
import torch
import numpy as np
from typing import Callable, Dict, Any
from copy import deepcopy
from torch import nn
from models.sac import SAC
from buffer import OfflineReplayBuffer, sample_action
from utils import count_vars


class SACTrainer:
    def __init__(
        self,
        env_fn: Callable,
        network: nn.Module = SAC,
        ac_kwargs: Dict[str, Any] = dict(),
        steps_per_epoch: int = 5000,
        epochs: int = 5,
        replay_size: int = int(1e6),
        gamma=0.99,
        polyak: float = 0.995,
        lr: float = 1e-3,
        alpha: float = 0.2,
        batch_size: int = 100,
        start_steps: int = 10000,
        update_after: int = 1000,
        update_every: int = 50,
        num_test_episodes: int = 40,
        max_ep_len: int = 1000,
    ):
        self.gamma = gamma
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.polyak = polyak
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len

        self.EpRet = []
        self.TestEpRet = []
        self.EpLen = []
        self.TestEpLen = []
        self.TotalEnvInteracts = []
        self.Q1Vals = []
        self.Q2Vals = []
        self.LogPi = []
        self.LossPi = []
        self.LossQ = []
        self.Time = []

        self.env = env_fn()
        self.test_env = env_fn()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.sac = network(self.obs_dim, self.act_dim, self.act_limit, **ac_kwargs)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        self.sac_trg = deepcopy(self.sac)
        for p in self.sac_trg.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and q-function
        self.pi_optimizer = torch.optim.Adam(self.sac.pi.parameters(), lr=lr)
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = list(self.sac.q1.parameters()) + list(self.sac.q2.parameters())
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=lr)

        # Experience buffer
        self.replay_buffer = OfflineReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size)

    def __repr__(self):
        # Count variables (protip: try to get a feel for how different size networks behave!)
        print(
            "\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n"
            % tuple(count_vars(module) for module in [self.sac.pi, self.sac.q1, self.sac.q2]),
        )

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]
        q1 = self.sac.q1(o, a)
        q2 = self.sac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.sac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.sac_trg.q1(o2, a2)
            q2_pi_targ = self.sac_trg.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data["obs"]
        a, logp_pi = self.sac.pi(o)
        q1_pi = self.sac.q1(o, a)
        q2_pi = self.sac.q2(o, a)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.LossQ.append(loss_q.item())
        self.Q1Vals.append(q_info["Q1Vals"])
        self.Q2Vals.append(q_info["Q2Vals"])

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.LossPi.append(loss_pi.item())
        self.LogPi.append(pi_info["LogPi"])

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.sac.parameters(), self.sac_trg.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.sac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            self.TestEpRet.append(ep_ret)
            self.TestEpLen.append(ep_len)

    def train(self):
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > self.start_steps:
                a = self.get_action(o)
            else:
                a = sample_action(self.act_dim, self.act_limit)

            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.EpRet.append(ep_ret)
                self.EpLen.append(ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample(self.batch_size)
                    self.update(data=batch)

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                self.TotalEnvInteracts.append(t)
                self.Time.append(time.time() - start_time)
                print(
                    "global_step=",
                    (epoch + 1) * self.steps_per_epoch,
                )
                for k, v in {
                    "Min-EpRet": np.min(self.EpRet[-10:]),
                    "Min-EpRet": np.min(self.EpRet[-10:]),
                    "Mean-EpRet": np.mean(self.EpRet[-10:]),
                    "Max-EpRet": np.max(self.EpRet[-10:]),
                    "EpLen": np.mean(self.EpLen[-10:]),
                    "Q1Vals": np.mean(self.Q1Vals[-10:]),
                    "Q2Vals": np.mean(self.Q2Vals[-10:]),
                    "LogPi": np.mean(self.LogPi[-10:]),
                    "LossPi": np.mean(self.LossPi[-10:]),
                    "LossQ": np.mean(self.LossQ[-10:]),
                    "time": time.time() - start_time,
                }.items():
                    print(f"{k}: {v}", end=",")

                self.save(epoch)

    def save(self, epoch: int, path: str = "outputs/sac_half_cheetah.pth") -> None:
        torch.save(
            {
                "model": self.sac.state_dict(),
                "optimizer_pi": self.pi_optimizer.state_dict(),
                "optimizer_critic": self.q_optimizer.state_dict(),
                "iteration": epoch * self.steps_per_epoch,
            },
            path,
        )
