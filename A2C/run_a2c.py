import gym
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from A2C.Agent import A2C
from A2C.utils import roll_out

tensorboard_flag = False
if tensorboard_flag:
    writer = SummaryWriter()
    record_step = 0

MAX_EPISODE = 2000
SAMPLE_NUMS = 1000
RENDER = False
GAMMA = 0.99
LR_A = 0.01
LR_V = 0.01

env = gym.make('CartPole-v0')
env = env.unwrapped

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a2c = A2C(state_dim=state_dim, action_dim=action_dim, lr_a=LR_A, lr_v=LR_V, reward_decay=GAMMA, device=device)

init_state = env.reset()
steps = []
task_episodes = []
test_results = []
for episode in range(MAX_EPISODE):
    states, actions, rewards, final_r, current_state = roll_out(a2c.actor, a2c.value, env, SAMPLE_NUMS, init_state,
                                                                action_dim, device)
    init_state = current_state
    a2c.learn(states, actions, rewards, final_r)

    # Testing
    if (episode + 1) % 50 == 0:
        result = 0
        test_task = gym.make("CartPole-v0")
        for test_epi in range(10):
            state = test_task.reset()
            for test_step in range(200):
                softmax_action = torch.exp(a2c.actor(Variable(torch.Tensor([state]).to(device))))
                # print(softmax_action.data)
                action = np.argmax(softmax_action.cpu().data.numpy()[0])
                next_state, reward, done, _ = test_task.step(action)
                result += reward
                state = next_state
                if done:
                    break
        print("episode:", episode + 1, "test result:", result / 10.0)
        if tensorboard_flag:
            writer.add_scalar('scalar/test_reward', round(result/10.0, 2), record_step)
            record_step += 1
        steps.append(episode + 1)
        test_results.append(result / 10)

if tensorboard_flag:
    writer.close()




