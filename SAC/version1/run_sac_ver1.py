import gym
import torch

from SAC.version1.Agent import SAC
from SAC.version1.utils import NormalizedActions


MAX_EPISODE = 100000
MAX_EP_STEPS = 2000
CAPACITY = 50000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = NormalizedActions(gym.make("Pendulum-v0"))
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

sac = SAC(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)

print("====================================")
print("Collection Experience...")
print("====================================")


for episode in range(MAX_EPISODE):
    state = env.reset()
    ep_r = 0
    for step in range(MAX_EP_STEPS):
        action = sac.choose_action(state)
        next_state, reward, done, info = env.step(action)
        ep_r += reward
        sac.store(state, action, reward, next_state, done)

        if sac.ReplayBuffer.count() >= sac.capacity-1:
            sac.update()

        state = next_state

        if done or step == 199:
            if episode % 10 == 0:
                print("Ep_i {}, the ep_r is {}, the t is {}".format(episode, ep_r, step))
            break






