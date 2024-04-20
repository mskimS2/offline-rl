import os
import gym
import time
import torch
import numpy as np
from torch import nn
from pytorch_lightning import LightningModule
from models.ppo import PPO
from buffer import OnlineReplayBuffer
from utils import count_vars
from loggers.base import Logger


class PPOTrainer:

    def __init__(
        self,
        env_fn,
        network=PPO,
        args=dict(),
        buffer=OnlineReplayBuffer,
        writer: Logger = None,
        pi_optimizer=torch.optim,
        vf_optimizer=torch.optim,
    ):
        self.args = args
        self.EpRet = []
        self.EpLen = []
        self.ClipFrac = []
        self.VVals = []
        self.LossPi = []
        self.LossV = []
        self.Entropy = []
        self.KL = []
        self.DeltaLossPi = []
        self.DeltaLossV = []

        # Instantiate environment
        self.env = env_fn()
        self.obs_dim = self.env.observation_space.shape[0] if hasattr(self.env, "observation_space") else None
        self.act_dim = self.env.action_space.shape[0] if hasattr(self.env, "action_space") else None
        self.writer = writer
        self.buffer = buffer
        self.network = network
        self.pi_optimizer = pi_optimizer
        self.vf_optimizer = vf_optimizer

    def train(self):
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        for epoch in range(self.args.epochs):

            for t in range(self.args.steps_per_epoch):
                a, v, logp = self.network(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1

                # save and log
                self.buffer.store(o, a, r, v, logp)
                self.VVals.append(v)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.args.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.args.steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print(
                            "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                            flush=True,
                        )
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.network(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0

                    self.buffer.finish_path(v)
                    if terminal:
                        self.EpRet.append(ep_ret)
                        self.EpLen.append(ep_len)

                    o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Perform PPO update!
            self.update()

            self.writer.add_scalars(
                {
                    "Min-EpRet": np.min(self.EpRet[-10:]),
                    "Min-EpRet": np.min(self.EpRet[-10:]),
                    "Mean-EpRet": np.mean(self.EpRet[-10:]),
                    "Max-EpRet": np.max(self.EpRet[-10:]),
                    "EpLen": np.mean(self.EpLen[-10:]),
                    "VVals": np.mean(self.VVals[-10:]),
                    "LossPi": np.mean(self.LossPi[-10:]),
                    "LossV": np.mean(self.LossV[-10:]),
                    "Entropy": np.mean(self.Entropy[-10:]),
                    "KL": np.mean(self.KL[-10:]),
                    "time": time.time() - start_time,
                },
                global_step=(epoch + 1) * self.args.steps_per_epoch,
            )

            print(
                "global_step=",
                (epoch + 1) * self.args.steps_per_epoch,
            )
            for k, v in {
                "Min-EpRet": np.min(self.EpRet[-10:]),
                "Min-EpRet": np.min(self.EpRet[-10:]),
                "Mean-EpRet": np.mean(self.EpRet[-10:]),
                "Max-EpRet": np.max(self.EpRet[-10:]),
                "EpLen": np.mean(self.EpLen[-10:]),
                "VVals": np.mean(self.VVals[-10:]),
                "LossPi": np.mean(self.LossPi[-10:]),
                "LossV": np.mean(self.LossV[-10:]),
                "Entropy": np.mean(self.Entropy[-10:]),
                "KL": np.mean(self.KL[-10:]),
                "time": time.time() - start_time,
            }.items():
                print(f"{k}: {v}", end=",")
            print()
            self.save(epoch + 1)

    def __repr__(self) -> str:
        print(f"\nNumber of parameters: \t pi: {count_vars(self.ac.pi)}, \t v: {count_vars(self.ac.v)}\n")

    def save(self, epoch: int, path: str = "outputs/ppo_half_cheetah.pth") -> None:
        torch.save(
            {
                "model": self.network.state_dict(),
                "optimizer_pi": self.pi_optimizer.state_dict(),
                "optimizer_critic": self.vf_optimizer.state_dict(),
                "args": self.args,
                "iteration": epoch * self.args.steps_per_epoch,
            },
            path,
        )

    # Set up optimizers for policy and value function

    def update(self):
        data = self.buffer.get()

        pi_l_old, pi_info_old = self.network.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.network.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.args.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.network.compute_loss_pi(data)
            kl = np.mean(pi_info["kl"])
            if kl > self.args.target_kl:
                print("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.network.pi.parameters(), self.args.max_grad_norm)
            self.pi_optimizer.step()

        # Value function learning
        for i in range(self.args.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.network.compute_loss_v(data)
            loss_v.backward()
            nn.utils.clip_grad_norm_(self.network.v.parameters(), self.args.max_grad_norm)
            self.vf_optimizer.step()

        # Log changes from update
        self.LossPi.append(pi_l_old)
        self.LossV.append(v_l_old)
        self.KL.append(pi_info["kl"])
        self.Entropy.append(pi_info_old["ent"])
        self.ClipFrac.append(pi_info["cf"])
        self.DeltaLossPi.append(loss_pi.item() - pi_l_old)
        self.DeltaLossV.append(loss_v.item() - v_l_old)