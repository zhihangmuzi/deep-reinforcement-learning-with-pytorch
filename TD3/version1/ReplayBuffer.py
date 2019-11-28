from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque()
        self.num_experiences = 0

    def add(self, transition):
        if self.num_experiences < self.capacity:
            self.buffer.append(transition)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    def sample(self, batch_size):
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def count(self):
        if self.num_experiences < self.capacity:
            return self.num_experiences
        else:
            return self.capacity

    def __len__(self):
        return self.capacity



