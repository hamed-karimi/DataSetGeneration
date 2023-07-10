import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
import Utilities

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def num_flat_features(x):  # This is a function we added for convenience to find out the number of features in a layer.
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class hDQN(nn.Module):  # meta controller network
    def __init__(self):
        utilities = Utilities.Utilities()
        params = utilities.params
        super(hDQN, self).__init__()
        env_layer_num = params.OBJECT_TYPE_NUM + 1  # +1 for agent layer
        kernel_size = 2
        self.conv1 = nn.Conv2d(in_channels=env_layer_num,
                               out_channels=params.DQN_CONV1_OUT_CHANNEL,
                               kernel_size=kernel_size)
        self.max_pool = nn.MaxPool2d(kernel_size=3,
                                     stride=2)
        self.conv2 = nn.Conv2d(in_channels=params.DQN_CONV1_OUT_CHANNEL,
                               out_channels=params.DQN_CONV1_OUT_CHANNEL,
                               kernel_size=3)
        self.fc1 = nn.Linear(in_features=params.DQN_CONV1_OUT_CHANNEL,
                             out_features=32)
        self.fc2 = nn.Linear(in_features=32 + params.OBJECT_TYPE_NUM,
                             out_features=16)  # +2 for needs
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 3)

    def forward(self, state_batch):
        env_map = state_batch.env_map
        agent_need = state_batch.agent_need

        y = F.relu(self.conv1(env_map))
        y = self.max_pool(y)
        y = F.relu(self.conv2(y))
        y = y.flatten(start_dim=1, end_dim=-1)
        y = F.relu(self.fc1(y))
        y = torch.concat([y, agent_need], dim=1)
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = self.fc4(y)
        return y


class lDQN(nn.Module):  # controller network
    def __init__(self):
        super(lDQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 4)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 9)

    def forward(self, state_batch):
        x = state_batch.env_map.clone() # agent map and goal map (the second layer contains a single 1)
        y = F.relu(self.conv1(x))
        y = y.view(-1, num_flat_features(y))
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return y
