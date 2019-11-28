import numpy as np
import gym
import torch

from TD3.version2.Agent import TD3


MAX_EPISODE = 100000
MAX_EP_STEPS = 2000
CAPACITY = 50000
exploration_noise = 0.1
print_log = 5


device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make("Pendulum-v0")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = TD3(state_dim, action_dim, max_action, device=device)
ep_r = 0

print("====================================")
print("Collection Experience...")
print("====================================")
print(device)

for episode in range(MAX_EPISODE):
    state = env.reset()
    for step in range(MAX_EP_STEPS):
        action = agent.select_action(state)
        action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
        action = action.clip(env.action_space.low, env.action_space.high)
        next_state, reward, done, info = env.step(action)
        ep_r += reward

        agent.store_transition(state, action, reward, next_state, np.float(done))

        if agent.ReplayBuffer.count() >= CAPACITY-1:
            agent.update(5)
        state = next_state

        if done or step == MAX_EP_STEPS-1:
            if episode % print_log == 0:
                print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(episode, ep_r, step))
            ep_r = 0
            break


