from tensorboardX import SummaryWriter
from DuelingDQN.Network import QNet
import torch


def plot_network():
    writer = SummaryWriter()

    dummy_input = torch.rand(1, 1, 1, 4)

    model = QNet(4, 2)

    writer.add_graph(model, dummy_input, True)

    writer.close()


if __name__ == "__main__":
    plot_network()
