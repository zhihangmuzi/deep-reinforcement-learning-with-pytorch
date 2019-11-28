import gym
from DQN.Agent import DQN
import torch
from tensorboardX import SummaryWriter


tensorboard_flag = False
if tensorboard_flag:
    writer = SummaryWriter()

env = gym.make('CartPole-v0')
env = env.unwrapped   # 还原env的原始设置

"""
print(env.action_space)            # 查看这个环境可用的action有多少个
print(env.observation_space)       # 查看这个环境中可用的state的observation有多少个
print(env.observation_space.high)  # 查看observation最高取值
print(env.observation_space.low)   # 查看observation最低取值
"""

state_dim = env.observation_space.shape[0]  # 环境维度
action_dim = env.action_space.n             # 动作维度

total_step = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN(state_dim=state_dim, action_dim=action_dim, device=device)
flag = False

for episode in range(400):
    state = env.reset()
    ep_r = 0
    while True:
        # env.render()

        action = dqn.choose_action(state)
        state_, reward, done, info = env.step(action)

        x, x_dot, theta, theta_dot = state_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_reward = r1 + r2

        dqn.store_transition(state, action, new_reward, state_, done)

        ep_r += new_reward
        if ep_r > 1200:
            torch.save(dqn.eval_net.state_dict(), 'eval.pth')
            flag = True
            done = True

        if total_step > 1000:
            dqn.learn()

        if done:
            if tensorboard_flag:
                writer.add_scalar('scalar/reward', round(ep_r, 2), episode)
            print('episode: ', episode,
                  'ep_r: ', round(ep_r, 2),
                  'epsilon: ', round(dqn.epsilon, 2))
            break

        state = state_
        total_step += 1

    if flag:
        break

if tensorboard_flag:
    writer.close()

