import os
import gym
import time
import torch
import numpy as np
from torch import nn
from ppo import MLPActorCritic
from buffer import PPOBuffer
from utils import count_vars
from torch.utils.tensorboard import SummaryWriter

device = "cpu"


def ppo(
    env_fn,
    actor_critic=MLPActorCritic,
    ac_kwargs=dict(),
    seed=2024,
    steps_per_epoch=5000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=3e-4,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=2048,
    target_kl=0.02,
    save_freq=10,
    max_grad_norm=0.5,
):
    writer = SummaryWriter()

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    EpRet = []
    EpLen = []
    VVals = []
    TotalEnvInteracts = []
    LossPi = []
    LossV = []
    DeltaLossPi = []
    DeltaLossV = []
    Entropy = []
    KL = []
    ClipFrac = []
    StopIter = []
    Time = []

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create actor-critic module
    ac = actor_critic(
        np.prod(env.observation_space.shape),
        np.prod(env.action_space.shape),
        **ac_kwargs,
    )

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    print("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = np.mean(pi_info["kl"])
            if kl > 1.5 * target_kl:
                print("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
            vf_optimizer.step()

        # Log changes from update
        LossPi.append(pi_l_old)
        LossV.append(v_l_old)
        KL.append(pi_info["kl"])
        Entropy.append(pi_info_old["ent"])
        ClipFrac.append(pi_info["cf"])
        DeltaLossPi.append(loss_pi.item() - pi_l_old)
        DeltaLossV.append(loss_v.item() - v_l_old)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            VVals.append(v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                        flush=True,
                    )
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    EpRet.append(ep_ret)
                    EpLen.append(ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Perform PPO update!
        update()

        TotalEnvInteracts.append((epoch + 1) * steps_per_epoch)
        Time.append(time.time() - start_time)

        print(
            f"[Epoch:{epoch}] EpRet:{np.min(EpRet[-10:]):8.2f} < {np.mean(EpRet[-10:]):8.2f} < {np.max(EpRet[-10:]):8.2f}, EpLen:{np.mean(EpLen[-10:]):8.2f}, VVals:{np.mean(VVals[-10:]):8.2f}, TotalEnvInteracts:{TotalEnvInteracts[-1]:8d}, LossPi:{np.mean(LossPi[-10:]):8.2f}, LossV:{np.mean(LossV[-10:]):8.2f}, Entropy:{np.mean(Entropy[-10:]):8.2f}, KL:{np.mean(KL[-10:]):8.2f}, Time:{Time[-1]:8.2f}"
        )

        writer.add_scalar(
            "Min-EpRet", np.min(EpRet[-10:]), global_step=TotalEnvInteracts[-1]
        )
        writer.add_scalar(
            "Mean-EpRet", np.mean(EpRet[-10:]), global_step=TotalEnvInteracts[-1]
        )
        writer.add_scalar(
            "Max-EpRet", np.max(EpRet[-10:]), global_step=TotalEnvInteracts[-1]
        )
        writer.add_scalar(
            "EpLen", np.mean(EpLen[-10:]), global_step=TotalEnvInteracts[-1]
        )
        writer.add_scalar(
            "VVals", np.mean(VVals[-10:]), global_step=TotalEnvInteracts[-1]
        )
        writer.add_scalar(
            "LossPi", np.mean(LossPi[-10:]), global_step=TotalEnvInteracts[-1]
        )
        writer.add_scalar(
            "LossV", np.mean(LossV[-10:]), global_step=TotalEnvInteracts[-1]
        )
        writer.add_scalar(
            "Entropy", np.mean(Entropy[-10:]), global_step=TotalEnvInteracts[-1]
        )
        writer.add_scalar("KL", np.mean(KL[-10:]), global_step=TotalEnvInteracts[-1])

        torch.save(ac.state_dict(), "outputs/ppo_half_cheetah.pth")
    return ac, EpRet, EpLen, VVals, TotalEnvInteracts, LossPi, LossV, Entropy, KL, Time


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    ac, EpRet, EpLen, VVals, TotalEnvInteracts, LossPi, LossV, Entropy, KL, Time = ppo(
        # lambda: gym.make("HalfCheetah-v4"),
        lambda: gym.make("HalfCheetah-v4"),
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[64, 64]),
        gamma=0.99,
        seed=42,
        steps_per_epoch=5000,
        epochs=600,
    )
