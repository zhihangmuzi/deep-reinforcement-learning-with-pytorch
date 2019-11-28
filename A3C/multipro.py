import torch.multiprocessing as mp
import gym

from A3C.Agent import A3C
from A3C.utils import v_wrap, push_and_pull, record


MAX_EPISODE = 4000
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9

env0 = gym.make('CartPole-v0')
state_dim = env0.observation_space.shape[0]
action_dim = env0.action_space.n


class Worker(mp.Process):
    def __init__(self, global_agent, opt, global_ep, global_ep_r, res_queue, name):
        """
        :param gnet: global network
        :param opt: optimizer
        :param global_ep: global_episode  全局共享内存变量
        :param global_ep_r: global_episode_reward 全局共享内存奖励
        :param res_queue: multiprocess queue 进程队列
        :param name: wi i->the number of worker
        """
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_agent, self.opt = global_agent, opt
        self.local_agent = A3C(state_dim=state_dim, action_dim=action_dim)  # local network
        self.env = gym.make('CartPole-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EPISODE:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            while True:
                if self.name == 'w0':
                    self.env.render()
                a = self.local_agent.choose_action(v_wrap(s[None, :]))   # choose action
                s_, r, done, _ = self.env.step(a)
                if done:
                    r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.opt, self.local_agent, self.global_agent, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)  # 学习
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)  # multiprocessor output
