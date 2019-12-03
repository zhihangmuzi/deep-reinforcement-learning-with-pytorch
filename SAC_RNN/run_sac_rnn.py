import gym
import torch

from SAC_RNN.Agent import SACRNN
from SAC_RNN.utils import NormalizedActions


MAX_EPISODE = 100000
MAX_EP_STEPS = 2000
CAPACITY = 50000
mode = 'gru'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = NormalizedActions(gym.make("Pendulum-v0"))
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

sac = SACRNN(state_dim=state_dim, action_dim=action_dim, max_action=max_action, mode=mode, device=device)

print("====================================")
print("Collection Experience...")
print("====================================")


for episode in range(MAX_EPISODE):
    state = env.reset()
    ep_r = 0
    last_action = env.action_space.sample()
    if mode == "lstm":
        hidden_out = (torch.zeros([1, 1, 512]).float().to(device), torch.zeros([1, 1, 512]).float().to(device))
    else:
        hidden_out = torch.zeros([1, 1, 512]).float().to(device)

    episode_state = []
    episode_action = []
    episode_last_action = []
    episode_reward = []
    episode_next_state = []
    episode_done = []

    for step in range(MAX_EP_STEPS):
        hidden_in = hidden_out
        action, hidden_out = sac.choose_action(state, last_action, hidden_in)
        next_state, reward, done, info = env.step(action)
        ep_r += reward

        if step > 0:
            ini_hidden_in = hidden_in
            ini_hidden_out = hidden_out
            episode_state.append(state)
            episode_action.append(action)
            episode_last_action.append(last_action)
            episode_reward.append(reward)
            episode_next_state.append(next_state)
            episode_done.append(done)

        state = next_state
        last_action = action

        if len(sac.ReplayBuffer) > sac.batch_size:
            sac.update()

        if done or step == 199:
            if episode % 10 == 0:
                print("Ep_i {}, the ep_r is {}, the t is {}".format(episode, ep_r, step))
            break

    sac.ReplayBuffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                           episode_reward, episode_next_state, episode_done)



