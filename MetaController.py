import torch
from State_batch import State_batch
from DQN import hDQN
import os


class MetaController:

    def __init__(self, trained_meta_controller_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = hDQN().to(self.device)
        self.target_net = hDQN().to(self.device)
        self.weights_path = trained_meta_controller_weights_path
        self.load_target_net_from_memory()

    def load_target_net_from_memory(self):
        model_path = torch.load(os.path.join(self.weights_path, 'meta_controller_model.pt'),
                                map_location=self.device)
        self.policy_net.load_state_dict(model_path)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_goal_map(self, environment, agent):

        with torch.no_grad():
            env_map = environment.env_map.clone().to(self.device)
            need = agent.need.to(self.device)
            state = State_batch(env_map, need)
            goal_values = self.policy_net(state).squeeze()
            ind = goal_values.argmax().cpu()

        # stay
        if ind == environment.object_locations.shape[0]:
            goal_map = environment.env_map[0, 0, :, :].clone()  # agent map as goal map
            return goal_map, ind.unsqueeze(0)

        # goal
        selected_obj_location = environment.object_locations[ind, :]
        goal_map = torch.zeros((environment.height, environment.width), dtype=torch.float32)
        goal_map[selected_obj_location[0], selected_obj_location[1]] = 1
        return goal_map, ind.unsqueeze(0)
