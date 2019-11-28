import gym
import torch
import datetime
from tensorboardX import SummaryWriter

from PER.Agent import PERDQN


tensorboard_flag = False
if tensorboard_flag:
    writer = SummaryWriter()

env = gym.make('MountainCar-v0')
env = env.unwrapped   # 还原env的原始设置

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

MAX_EPISODE = 500
LR = 0.001
GAMMA = 0.9
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

perdqn = PERDQN(state_dim=state_dim, action_dim=action_dim, learning_rate=LR, reward_decay=GAMMA,
                memory_size=MEMORY_CAPACITY, batch_size=BATCH_SIZE)

steps = []
episodes = []

starttime = datetime.datetime.now()

for episode in range(20):
    total_step = 0
    state = env.reset()
    while True:

        # env.render()

        action = perdqn.choose_action(state)
        state_, reward, done, info = env.step(action)

        if done:
            reward = 20

        perdqn.store_transition(state, action, reward, state_, done)

        if total_step > MEMORY_CAPACITY:
            perdqn.learn()

        if done:
            print('episode ', episode, ' finished')
            writer.add_scalar('scalar/steps', total_step, episode)
            break

        state_ = state
        total_step += 1

if tensorboard_flag:
    writer.close()

endtime = datetime.datetime.now()
n_time = (endtime - starttime).seconds
print(n_time)





