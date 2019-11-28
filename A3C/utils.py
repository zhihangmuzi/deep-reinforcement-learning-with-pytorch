from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    """
    np_array -> torch.tensor
    :param np_array:
    :param dtype:
    :return: torch.float.tensor
    """
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    """
    :param layers: pytorch net layer
    :return: initiation the layer
    """
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, local_agent, global_agent, done, s_, bs, ba, br, gamma):
    """
    :param opt: optimizer: share optimizer
    :param local_agent: local network
    :param global_agent: global network
    :param done: episode done
    :param s_: the next state
    :param bs: buffer_s
    :param ba: buffer_a
    :param br: buffer_r
    :param gamma: reward_decay
    :return: update the network
    """
    if done:
        v_s_ = 0.  # if done, value(s_)=0
    else:
        v_s_ = local_agent.actor_value(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]  # value(s_)

    buffer_v_target = []   # record decay reward
    for r in br[::-1]:
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = local_agent.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(local_agent.actor_value.parameters(), global_agent.actor_value.parameters()):
        gp._grad = lp.grad  # local network grad --copy to--> global network
    opt.step()

    # pull global parameters
    # local network --load parameters--> global network
    local_agent.actor_value.load_state_dict(global_agent.actor_value.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    """
    :param global_ep: global_episode  全局共享内存变量
    :param global_ep_r: global_episode_reward 全局共享内存奖励
    :param ep_r: 一次episode的奖励和
    :param res_queue: multiprocess队列输出
    :param name:
    :return:
    """
    with global_ep.get_lock():
        global_ep.value += 1  # global episode +1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(name, "Ep:", global_ep.value, "| Ep_r: %.0f" % global_ep_r.value,)

    """
            1. with
                with lock:
                    do

            2. mp.Lock()
                l = mp.Lock()
                l.acquire()
                do
                l.release()
            """
