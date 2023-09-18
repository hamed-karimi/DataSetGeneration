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

    def get_agent(self, pre_location, preassigned_needs):
        agent = Agent(self.params.HEIGHT, self.params.WIDTH, n=self.params.OBJECT_TYPE_NUM,
                      agent_type=self.params.AGENT_TYPE,
                      prob_init_needs_equal=self.params.PROB_OF_INIT_NEEDS_EQUAL,
                      predefined_location=pre_location,
                      preassigned_needs=preassigned_needs)
        self.agent = agent
        return agent

    def get_environment(self, few_many, probability_map, pre_located_objects_num, pre_located_objects_location,
                        random_new_object_type, random_new_object):  # pre_located_objects is a 2D list
        environment = Environment(few_many, self.params.HEIGHT, self.params.WIDTH, self.agent, probability_map,
                                  reward_of_object=self.params.REWARD_OF_OBJECT,
                                  far_objects_prob=self.params.PROB_OF_FAR_OBJECTS_FOR_TWO,
                                  num_object=self.params.OBJECT_TYPE_NUM,
                                  pre_located_objects_num=pre_located_objects_num,
                                  pre_located_objects_location=pre_located_objects_location,
                                  random_new_object_type=random_new_object_type,
                                  random_new_object=random_new_object)
        self.environment = environment
        return environment

    def get_controller(self):
        controller = Controller(self.params.HEIGHT, self.params.WIDTH)
        self.controller = deepcopy(controller)
        return controller

    def get_meta_controller(self):
        meta_controller = MetaController(self.params.META_CONTROLLER_DIRECTORY)
        self.meta_controller = deepcopy(meta_controller)
        return meta_controller
