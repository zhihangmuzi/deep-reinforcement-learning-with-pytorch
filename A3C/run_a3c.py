import multiprocessing as mp
import matplotlib.pyplot as plt
import gym
import os

from A3C.multipro import Worker
from A3C.shared_adam import ShareAdam
from A3C.Agent import A3C


os.environ["OMP_NUM_THREADS"] = "1"  # 只使用一个线程

MAX_EPISODE = 4000
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9

env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


if __name__ == '__main__':
    global_agent = A3C(state_dim=state_dim, action_dim=action_dim)  # global network
    global_agent.actor_value.share_memory()   # share the global parameters in multiprocessing
    opt = ShareAdam(global_agent.actor_value.parameters(), lr=0.0001)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    # mp.Queue 多线程的结果回储存在Queue上，直接从序列上读取结果

    workers = [Worker(global_agent, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    # print(mp.cpu_count())   output-->4
    [w.start() for w in workers]  # p1.start() 开启线程运算
    res = []      # record episode reward to plot
    while True:
        r = res_queue.get()  # get the multiprocess output from queue
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]  # p1.join() 加入线程运算

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
