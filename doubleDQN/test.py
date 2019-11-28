import gym
from doubleDQN.Network import QNet
import torch


def choose_action(in_state, eval_net, device):
    in_state = torch.Tensor(in_state).float().unsqueeze(0).to(device)
    actions_value = eval_net(in_state)
    out_action = torch.max(actions_value.cpu(), 1)[1].data.numpy()
    act = out_action[0]
    return act


env = gym.make('CartPole-v0')
env = env.unwrapped   # 还原env的原始设置

state_dim = env.observation_space.shape[0]  # 环境维度
action_dim = env.action_space.n             # 动作维度

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_net = QNet(state_dim, action_dim).to(device)
eval_net.load_state_dict(torch.load('eval.pth'))

state = env.reset()
ep_r = 0
while True:
    env.render()

    action = choose_action(state, eval_net, device)
    state_, reward, done, info = env.step(action)

    x, x_dot, theta, theta_dot = state_
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    new_reward = r1 + r2

    ep_r += new_reward

    if done:
        print(ep_r)
        break

    state = state_

