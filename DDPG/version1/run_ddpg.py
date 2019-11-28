import gym
import torch
import numpy as np
import time

from DDPG.version1.Agent import DDPG
from tensorboardX import SummaryWriter


tensorboard_flag = False
if tensorboard_flag:
    writer = SummaryWriter()

MAX_EPISODE = 200
MAX_EP_STEPS = 200
LR_A = 0.001
LR_C = 0.001
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ddpg = DDPG(a_dim, s_dim, lr_a=LR_A, lr_c=LR_C, reward_decay=GAMMA, memory_size=MEMORY_CAPACITY,
            batch_size=BATCH_SIZE, tau=TAU, device=device)

var = 3
t1 = time.time()
RENDER = False
for episode in range(MAX_EPISODE):
    state = env.reset()
    ep_r = 0

    for step in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        s = np.float32(state)
        action = ddpg.choose_action(state)
        action = np.clip(np.random.normal(action, var), -2, 2)

        state_, reward, done, info = env.step(action)

        ddpg.store_transition(state, action, reward/10, state_, done)

        if ddpg.ReplayBuffer.count() == MEMORY_CAPACITY:
            var *= 0.9995  # decay the action randomness
            ddpg.learn()

        state = state_
        ep_r += reward

        if step == MAX_EP_STEPS - 1:
            if tensorboard_flag:
                writer.add_scalar('scalar/reward', round(ep_r, 2), episode)
            print('Episode:', episode, ' Reward: %i' % int(ep_r), 'Explore: %.2f' % var, )
            if ep_r > -300:
                RENDER = False
            break

print('Running time: ', time.time() - t1)







