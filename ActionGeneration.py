import torch
from ObjectFactory import ObjectFactory
from Utilities import Utilities
import numpy as np
from Visualizer import Visualizer
import matplotlib.pyplot as plt
import pickle
from os.path import exists as pexists
import os


def agent_reached_goal(environment, goal_index):
    if goal_index == environment.nObj:
        return False
    target_goal_layer = goal_index + 1
    agent_object_maps_equality = torch.all(torch.eq(environment.env_map[0, 0, :, :],
                                                    environment.env_map[0, target_goal_layer, :, :]))
    if agent_object_maps_equality:
        return True
    return False


def create_tensors(params):
    environments = torch.zeros((params.EPOCHS_NUM,
                                params.STEPS_NUM,
                                params.OBJECT_NUM + 1,
                                params.HEIGHT,
                                params.WIDTH), dtype=torch.float32)
    needs = torch.zeros((params.EPOCHS_NUM,
                         params.STEPS_NUM,
                         params.OBJECT_NUM), dtype=torch.float32)
    actions = torch.zeros((params.EPOCHS_NUM,
                           params.STEPS_NUM), dtype=torch.int32)
    selected_goals = torch.zeros((params.EPOCHS_NUM,
                                  params.STEPS_NUM), dtype=torch.int32)
    goal_reached = torch.zeros((params.EPOCHS_NUM,
                                params.STEPS_NUM), dtype=torch.bool)

    return environments, needs, actions, selected_goals, goal_reached


def generate_action():
    print('start')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not pexists('./Data'):
        os.mkdir('./Data')

    utility = Utilities('Parameters.json')
    params = utility.get_params()
    factory = ObjectFactory(utility)
    res_folder = utility.make_res_folder()

    environment_initialization_prob_map = np.ones(params.HEIGHT * params.WIDTH) * 100 / (params.HEIGHT * params.WIDTH)
    meta_controller = factory.get_meta_controller().to(device)
    controller = factory.get_controller().to(device)

    print_threshold = 1
    visualizer = Visualizer(utility)
    environments, needs, actions, selected_goals, goal_reached = create_tensors(params)
    for epoch in range(params.EPOCHS_NUM):
        batch_environments_ll = []
        batch_actions_ll = []
        batch_needs_ll = []
        batch_selected_goals_ll = []

        agent = factory.get_agent(need_num=2)
        environment = factory.get_environment(environment_initialization_prob_map, num_object=2)
        for step in range(params.STEPS_NUM):
            goal_map, goal_index = meta_controller.get_goal_map(environment, agent)

            batch_environments_ll.append(environment.env_map.clone())
            batch_needs_ll.append(agent.need.clone())
            batch_selected_goals_ll.append(goal_index.cpu().clone())

            if epoch < print_threshold:
                fig, ax = visualizer.map_to_image(agent, environment)
                fig.savefig('{0}/epoch_{1}_step_{2}.png'.format(res_folder, epoch, step))
                plt.close()

            agent_goal_map = torch.stack([environment.env_map[0, 0, :, :],
                                          goal_map.cpu()], dim=0).unsqueeze(0).to(device)
            action_id = controller.get_action(agent_goal_map, environment.get_action_mask())
            _, _ = agent.take_action(environment, action_id.cpu())
            at_step_goal_reached = agent_reached_goal(environment, goal_index.cpu())
            goal_reached[epoch, step] = at_step_goal_reached

            batch_actions_ll.append(action_id.clone())

            if at_step_goal_reached:
                environment = factory.get_environment(environment_initialization_prob_map, num_object=2)

        environments[epoch, :, :, :, :] = torch.cat(batch_environments_ll, dim=0)
        needs[epoch, :, :] = torch.cat(batch_needs_ll, dim=0)
        selected_goals[epoch, :] = torch.cat(batch_selected_goals_ll, dim=0)
        actions[epoch, :] = torch.cat(batch_actions_ll, dim=0)

        if epoch % 100 == 0:
            print(epoch)

        # Saving to memory

    torch.save(environments, './Data/environments.pt')
    torch.save(needs, './Data/needs.pt')
    torch.save(selected_goals, './Data/selected_goals.pt')
    torch.save(goal_reached, './Data/goal_reached.pt')
    torch.save(actions, './Data/actions.pt')
