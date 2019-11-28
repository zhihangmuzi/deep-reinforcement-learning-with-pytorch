import gym
from collections import namedtuple

from PPO_Continuous.version2.Agent import PPO


MAX_EPISODE = 1000
MAX_EP_STEP = 200

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state'])

ppo = PPO(state_dim=state_dim, action_dim=action_dim)

running_reward = -1000
render = False
for episode in range(MAX_EPISODE):
    score = 0
    state = env.reset()
    for step in range(200):
        action, action_log_prob = ppo.choose_action(state)
        next_state, reward, done, info = env.step([action])
        trans = Transition(state, action, reward, action_log_prob, next_state)
        if render:
            env.render()
        if ppo.store_transition(trans):
            ppo.update()
        score += reward
        state = next_state

    running_reward = running_reward * 0.9 + score * 0.1
    # training_records.append(TrainingRecord(i_epoch, running_reward))
    if episode % 10 == 0:
        print("Epoch {}, Moving average score is: {:.2f} ".format(episode, running_reward))
        if running_reward > -600:
            render = True
    if running_reward > -200:
        print("Solved! Moving average score is now {}!".format(running_reward))
        env.close()
        # agent.save_param()
        break
