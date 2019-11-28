import torch
import numpy as np
from torch.autograd import Variable


def entropy(p):
    return -torch.sum(p * torch.log(p), 1)


def discount_reward(rewards, gamma, final_r):
    discounted_r = np.zeros_like(rewards)
    running_add = final_r
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


def roll_out(actor, value, env, sample_nums, init_state, action_dim, device):
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    for j in range(sample_nums):
        states.append(state)
        probs = actor(Variable(torch.Tensor([state]).to(device)))
        action = np.random.choice(action_dim, p=probs.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(action_dim)]
        next_state, reward, done, _ = env.step(action)
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state = env.reset()
            break
    if not is_done:
        final_r = value(Variable(torch.Tensor([final_state]).to(device))).cpu().data.numpy()

    return states, actions, rewards, final_r, state

