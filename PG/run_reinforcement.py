import gym
import torch
from tensorboardX import SummaryWriter

from PG.Agent import PG


tensorboard_flag = False
if tensorboard_flag:
    writer = SummaryWriter()

MAX_EPISODE = 500
MAX_EPISODE_STEPS = 50000
LR = 0.02
GAMMA = 0.995
RENDER = False
DISPLAY_REWARD_THRESHOLD = -2000

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pg = PG(state_dim=state_dim, action_dim=action_dim, learning_rate=LR, reward_decay=GAMMA, device=device)

for episode in range(MAX_EPISODE):
    total_step = 0
    state = env.reset()

    while True:
        if RENDER:
            env.render()

        action = pg.choose_action(state)
        state_, reward, done, info = env.step(action)
        pg.store_transition(state, action, reward)

        if total_step > MAX_EPISODE_STEPS:
            done = True

        if done:
            ep_rs_sum = sum(pg.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = False  # rendering

            writer.add_scalar('scalar/reward', int(running_reward), episode)
            print("episode:", episode, "  reward:", int(running_reward))

            vt = pg.learn()
            break

        state = state_
        total_step += 1

if tensorboard_flag:
    writer.close()


