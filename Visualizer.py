import math
import numpy as np
import torch
from copy import deepcopy
from State_batch import State_batch
import matplotlib.pyplot as plt
import itertools
from matplotlib.ticker import FormatStrFormatter


# def get_predefined_needs():
#     temp_need = [[-10, -5, 0, 5, 10]] * 2
#     need_num = len(temp_need[0]) ** 2
#     need_batch = torch.zeros((need_num, 2))
#     for i, (n1, n2) in enumerate(itertools.product(*temp_need)):
#         need_batch[i, :] = torch.tensor([n1, n2])
#     return need_batch
#
#
# def get_reward_plot(ax, r, c, **kwargs):
#     ax[r, c].plot(kwargs['reward'], linewidth=1)
#     ax[r, c].set_title(kwargs['title'], fontsize=9)
#     # ax[r, c].set_box_aspect(aspect=1)
#     c += 1
#     return ax, r, c
#
#
# def get_loss_plot(ax, r, c, **kwargs):
#     ax[r, c].plot(kwargs['loss'], linewidth=1)
#     ax[r, c].set_title(kwargs['title'], fontsize=9)
#     # ax[r, c].set_box_aspect(aspect=1)
#     c += 1
#     return ax, r, c


class Visualizer:
    def __init__(self, utility):
        params = utility.get_params()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = params.HEIGHT
        self.width = params.WIDTH
        allactions_np = [np.array([0, 0]), np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1]),
                         np.array([1, 1]), np.array([-1, -1]), np.array([-1, 1]), np.array([1, -1])]
        self.allactions = [torch.from_numpy(x).unsqueeze(0) for x in allactions_np]
        self.action_mask = np.zeros((self.height, self.width, 1, len(self.allactions)))
        self.initialize_action_masks()
        self.color_options = [[1, 0, .2], [0, .8, .2], [1, 1, 1]]

    def get_epsilon_plot(self, ax, r, c, steps_done, **kwargs):
        pass

    def initialize_action_masks(self):
        for i in range(self.height):
            for j in range(self.width):
                agent_location = torch.tensor([[i, j]])
                aa = np.ones((agent_location.size(0), len(self.allactions)))
                for ind, location in enumerate(agent_location):
                    if location[0] == 0:
                        aa[ind, 2] = 0
                        aa[ind, 6] = 0
                        aa[ind, 7] = 0
                    if location[0] == self.height - 1:
                        aa[ind, 1] = 0
                        aa[ind, 5] = 0
                        aa[ind, 8] = 0
                    if location[1] == 0:
                        aa[ind, 4] = 0
                        aa[ind, 6] = 0
                        aa[ind, 8] = 0
                    if location[1] == self.width - 1:
                        aa[ind, 3] = 0
                        aa[ind, 5] = 0
                        aa[ind, 7] = 0
                self.action_mask[i, j, :, :] = aa

    def map_to_image(self, agent, environment):

        agent_location = environment.agent_location
        objects_location = environment.object_locations

        fig, ax = plt.subplots(figsize=(15, 10))
        arrows_x = np.zeros((self.height, self.width))
        arrows_y = np.zeros((self.height, self.width))

        Xs = np.arange(0, self.height, 1)
        Ys = np.arange(0, self.width, 1)

        ax.quiver(Xs, Ys, arrows_x, arrows_y, scale=10)
        ax.set_title("$n_{{red}}: {:.2f}, n_{{green}}: {:.2f}$".format(agent.need[0, 0], agent.need[0, 1]), fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        for obj_type in range(objects_location.shape[0]):
            for obj in range(environment.each_type_object_num[obj_type]):
                ax.scatter(objects_location[obj_type, obj, 1],
                           objects_location[obj_type, obj, 0],
                           marker='*', s=500, facecolor=self.color_options[obj_type])
        ax.set_box_aspect(aspect=1)

        ax.scatter(agent_location[0, 1], agent_location[0, 0], s=380, facecolors='b', edgecolors='k')
        plt.tight_layout(pad=0.4, w_pad=1, h_pad=1)
        return fig, ax
