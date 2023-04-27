from Agent import Agent
from Environment import Environment
from Controller import Controller
from MetaController import MetaController

from copy import deepcopy


class ObjectFactory:
    def __init__(self, utility):
        self.agent = None
        self.environment = None
        self.controller = None
        self.meta_controller = None
        self.params = utility.get_params()

    def get_agent(self, need_num):
        agent = Agent(self.params.HEIGHT, self.params.WIDTH,
                      n=need_num,
                      prob_init_needs_equal=self.params.PROB_OF_INIT_NEEDS_EQUAL,
                      rho_function=self.params.RHO_FUNCTION)
        self.agent = agent
        return agent

    def get_environment(self, probability_map, num_object):
        env = Environment(self.params.HEIGHT, self.params.WIDTH, self.agent, probability_map,
                          reward_of_object=self.params.REWARD_OF_OBJECT,
                          far_objects_prob=self.params.PROB_OF_FAR_OBJECTS_FOR_TWO,
                          num_object=num_object)
        self.environment = env
        return env

    def get_controller(self):
        controller = Controller(self.params.CONTROLLER_DIRECTORY)
        self.controller = deepcopy(controller)
        return controller

    def get_meta_controller(self):
        meta_controller = MetaController(self.params.META_CONTROLLER_DIRECTORY)
        self.meta_controller = deepcopy(meta_controller)
        return meta_controller
