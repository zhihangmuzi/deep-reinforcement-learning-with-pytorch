import gym
from collections import namedtuple

from PPO.version2.Agent import PPO


MAX_EPISODE = 1000

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state', 'done'])

ppo = PPO(state_dim=state_dim, action_dim=action_dim)

for episode in range(MAX_EPISODE):
    state = env.reset()
    total_reward = 0

    while True:
        action, action_prob = ppo.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        trans = Transition(state, action, action_prob, reward, next_state, done)
        ppo.store_transition(trans)
        state = next_state

        total_reward += reward

        if done:
            print("episode: {}, total reward: {}".format(episode, total_reward))
            if len(ppo.ReplayBuffer) >= ppo.batch_size:
                ppo.update()
            break
