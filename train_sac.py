import os
import gym
from copy import deepcopy
from torch.optim import Adam
import time
import torch
import numpy as np
from torch import nn
from sac import MLPActorCritic
from buffer import ReplayBuffer
from utils import count_vars
from torch.utils.tensorboard import SummaryWriter

device = "cuda"


def sample_action(action_dim, action_limit):
    return (2.0 * np.random.uniform(size=(action_dim,)) - 1) * action_limit


def sac(
    env_fn,
    actor_critic=MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=5000,
    epochs=5,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    lr=1e-3,
    alpha=0.2,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=40,
    max_ep_len=1000,
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    EpRet = []
    TestEpRet = []
    EpLen = []
    TestEpLen = []
    TotalEnvInteracts = []
    Q1Vals = []
    Q2Vals = []
    LogPi = []
    LossPi = []
    LossQ = []
    Time = []

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(obs_dim, act_dim, act_limit, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # List of parameters for both Q-networks (save this for convenience)
    q_params = list(ac.q1.parameters()) + list(ac.q2.parameters())

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    print("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data["obs"]
        a, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, a)
        q2_pi = ac.q2(o, a)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        LossQ.append(loss_q.item())
        Q1Vals.append(q_info["Q1Vals"])
        Q2Vals.append(q_info["Q2Vals"])

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        LossPi.append(loss_pi.item())
        LogPi.append(pi_info["LogPi"])

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            TestEpRet.append(ep_ret)
            TestEpLen.append(ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
        else:
            a = sample_action(act_dim, act_limit)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            EpRet.append(ep_ret)
            EpLen.append(ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_agent()

            TotalEnvInteracts.append(t)
            Time.append(time.time() - start_time)
            print(
                f"[Epoch:{epoch}] TestEpRet:{np.min(TestEpRet[-10:]):8.2f} < {np.mean(TestEpRet[-10:]):8.2f} < {np.max(TestEpRet[-10:]):8.2f}, TestEpLen:{np.mean(TestEpLen[-10:]):8.2f}, EpRet:{np.min(EpRet[-10:]):8.2f} < {np.mean(EpRet[-10:]):8.2f} < {np.max(EpRet[-10:]):8.2f}, EpLen:{np.mean(EpLen[-10:]):8.2f}, Q1Vals:{np.mean(Q1Vals[-10:]):8.2f}, Q2Vals:{np.mean(Q2Vals[-10:]):8.2f}, TotalEnvInteracts:{TotalEnvInteracts[-1]:8d}, LossPi:{np.mean(LossPi[-10:]):8.2f}, LossQ:{np.mean(LossQ[-10:]):8.2f}, Time:{Time[-1]:8.2f}"
            )
    return ac, EpRet, EpLen, Q1Vals, Q2Vals, TotalEnvInteracts, LossPi, LossQ, Time


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    lr = 1e-3
    env = gym.make("HalfCheetah-v4")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = MLPActorCritic(obs_dim, act_dim, act_limit)
    ac_targ = deepcopy(ac)

    # List of parameters for both Q-networks (save this for convenience)
    q_params = list(ac.q1.parameters()) + list(ac.q2.parameters())

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    print("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
    sac = sac(
        lambda: env,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[256, 256]),
        gamma=0.99,
        seed=0,
        epochs=10,
    )
