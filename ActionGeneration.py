import torch
from ObjectFactory import ObjectFactory
from Utilities import Utilities
import numpy as np
from Visualizer import Visualizer
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from os.path import exists as pexists
import os


def agent_reached_goal(environment, goal_index):
    target_goal_layer = 0 if goal_index == environment.object_type_num else goal_index.item() + 1
    same_elements_on_grid_num = torch.logical_and(environment.env_map[0, 0, :, :],
                                                  environment.env_map[0, target_goal_layer, :, :]).sum()
    if same_elements_on_grid_num > 0:
        return True
    return False


def update_pre_located_objects(object_locations, agent_location, goal_reached):
    pre_located_objects = []

    if goal_reached:
        for obj_type in object_locations:
            temp = []
            for loc in obj_type:
                if any(~torch.eq(loc, agent_location[0])):
                    temp.append(loc.tolist())
                else:
                    temp.append([-1, -1])
            pre_located_objects.append(temp)
    return pre_located_objects


def create_tensors(params):
    environments = torch.zeros((params.EPOCHS_NUM,
                                params.STEPS_NUM,
                                params.OBJECT_TYPE_NUM + 1,
                                params.HEIGHT,
                                params.WIDTH), dtype=torch.float32)
    needs = torch.zeros((params.EPOCHS_NUM,
                         params.STEPS_NUM,
                         params.OBJECT_TYPE_NUM), dtype=torch.float32)
    actions = torch.zeros((params.EPOCHS_NUM,
                           params.STEPS_NUM), dtype=torch.int32)
    selected_goals = torch.zeros((params.EPOCHS_NUM,
                                  params.STEPS_NUM), dtype=torch.int32)
    goal_reached = torch.zeros((params.EPOCHS_NUM,
                                params.STEPS_NUM), dtype=torch.bool)

    return environments, needs, actions, selected_goals, goal_reached


def generate_action():
    if not pexists('./Data'):
        os.mkdir('./Data')

    utility = Utilities()
    params = utility.get_params()
    factory = ObjectFactory(utility)
    res_folder = utility.make_res_folder()

    environment_initialization_prob_map = np.ones(params.HEIGHT * params.WIDTH) * 100 / (params.HEIGHT * params.WIDTH)
    meta_controller = factory.get_meta_controller()
    controller = factory.get_controller()

    print_threshold = 2
    visualizer = Visualizer(utility)
    environments, needs, actions, selected_goals, goal_reached = create_tensors(params)
    for epoch in range(params.EPOCHS_NUM):
        batch_environments_ll = []
        batch_actions_ll = []
        batch_needs_ll = []
        batch_selected_goals_ll = []
        pre_located_objects_location = [[[]]] * params.OBJECT_TYPE_NUM
        pre_located_objects_num = [] * params.OBJECT_TYPE_NUM
        object_amount_options = ['few', 'many']
        episode_object_amount = [np.random.choice(object_amount_options) for _ in range(params.OBJECT_TYPE_NUM)]

        agent = factory.get_agent(pre_location=[[]],
                                  preassigned_needs=[[]])
        environment = factory.get_environment(episode_object_amount,
                                              environment_initialization_prob_map,
                                              pre_located_objects_num,
                                              pre_located_objects_location,
                                              random_new_object_type=None,
                                              random_new_object=False)
        n_step = 0
        n_goal = 0
        while True:

            # environment_0 = deepcopy(environment)
            # agent_0 = deepcopy(agent)
            goal_map, goal_type = meta_controller.get_goal_map(environment,
                                                               agent)  # goal type is either 0 or 1
            n_goal += 1
            while True:
                batch_environments_ll.append(environment.env_map.clone())
                batch_needs_ll.append(agent.need.clone())
                batch_selected_goals_ll.append(goal_type.cpu().clone())

                if epoch < print_threshold:
                    fig, ax = visualizer.map_to_image(agent, environment)
                    fig.savefig('{0}/epoch_{1}_goal_{2}_step_{3}.png'.format(res_folder, epoch, n_goal, n_step))
                    plt.close()

                agent_goal_map = torch.stack([environment.env_map[:, 0, :, :], goal_map], dim=1)
                action_id = controller.get_action(agent_goal_map).clone()
                agent.take_action(environment, action_id)

                step_goal_reached = agent_reached_goal(environment, goal_type)
                goal_reached[epoch, n_step] = step_goal_reached

                batch_actions_ll.append(action_id.clone())
                # all_actions += 1
                n_step += 1

                if step_goal_reached or n_step == params.STEPS_NUM:
                    if step_goal_reached:
                        pre_located_objects_location = update_pre_located_objects(environment.object_locations,
                                                                                  agent.location,
                                                                                  step_goal_reached)
                        pre_located_objects_num = environment.each_type_object_num
                        # pre_located_agent = agent.location.tolist()
                        # pre_assigned_needs = agent.need.tolist()

                        environment = factory.get_environment(episode_object_amount,
                                                              environment_initialization_prob_map,
                                                              pre_located_objects_num,
                                                              pre_located_objects_location,
                                                              random_new_object_type=None,
                                                              random_new_object=False)
                    break

            if n_step == params.STEPS_NUM:
                break
        # for step in range(params.STEPS_NUM):
        #     goal_map, goal_index = meta_controller.get_goal_map(environment, agent)
        #
        #     batch_environments_ll.append(environment.env_map.clone())
        #     batch_needs_ll.append(agent.need.clone())
        #     batch_selected_goals_ll.append(goal_index.cpu().clone())
        #
        #     if epoch < print_threshold:
        #         fig, ax = visualizer.map_to_image(agent, environment)
        #         fig.savefig('{0}/epoch_{1}_step_{2}.png'.format(res_folder, epoch, step))
        #         plt.close()
        #
        #     agent_goal_map = torch.stack([environment.env_map[0, 0, :, :],
        #                                   goal_map.cpu()], dim=0).unsqueeze(0)
        #     action_id = controller.get_action(agent_goal_map)
        #     _, _ = agent.take_action(environment, action_id.cpu())
        #     at_step_goal_reached = agent_reached_goal(environment, goal_index.cpu())
        #     goal_reached[epoch, step] = at_step_goal_reached
        #
        #     batch_actions_ll.append(action_id.clone())
        #
        #     if at_step_goal_reached:
        #         environment = factory.get_environment(environment_initialization_prob_map, num_object=2)

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
