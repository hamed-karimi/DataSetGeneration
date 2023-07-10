import json
from types import SimpleNamespace
import os
from datetime import datetime
import numpy as np


class Utilities:
    def __init__(self):
        with open('./Parameters.json', 'r') as json_file:
            self.params = json.load(json_file,
                                    object_hook=lambda d: SimpleNamespace(**d))

    def get_params(self):
        return self.params

    def make_res_folder(self, sub_folder=''):
        now = datetime.now().strftime("%d-%m-%Y_%H-%M")
        folder = '{0}_{1}'.format(now, self.params.STEPS_NUM)

        dirname = os.path.join(folder, sub_folder)

        if os.path.exists(folder) and not os.path.exists(dirname):
            os.mkdir(dirname)
        elif not os.path.exists(dirname):
            os.makedirs(dirname)
        return dirname

    def get_environment_probability_map(self, style, params): # style: 'equal', or 'edges'
        if style == 'equal':
            prob_map = np.ones(params.HEIGHT * params.WIDTH) * 100 / (params.HEIGHT * params.WIDTH)
        elif style == 'edges':
            prob_map = np.ones((params.HEIGHT, params.WIDTH))
            prob_map[[0, params.WIDTH - 1], :] *= 3
            prob_map[1:-1, [0, params.HEIGHT - 1]] *= 3

