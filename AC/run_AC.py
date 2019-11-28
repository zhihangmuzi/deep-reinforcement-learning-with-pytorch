import gym
import torch
from tensorboardX import SummaryWriter

from AC.Agent import AC


tensorboard_flag = False
if tensorboard_flag:
    writer = SummaryWriter()

MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200
MAX_EP_STEPS = 1000
RENDER = False
GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ac = AC(state_dim=state_dim, action_dim=action_dim, lr_a=LR_A, lr_c=LR_C, reward_decay=GAMMA, device=device)

for episode in range(MAX_EPISODE):
    state = env.reset()
    t = 0
    track_r = []

    while True:
        if RENDER:
            env.render()

        action = ac.choose_action(state)
        state_, reward, done, info = env.step(action)

        if done:
            reward = -20

        track_r.append(reward)

        td_error = ac.critic_learn(state, reward, state_).detach().numpy()
        ac.actor_learn(state, action, td_error)

        state = state_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True
            if tensorboard_flag:
                writer.add_scalar('scaler/reward', int(running_reward), episode)
            print("episode:", episode, "  reward:", int(running_reward))
            break

if tensorboard_flag:
    writer.close()
