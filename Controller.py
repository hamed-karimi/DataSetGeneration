import os.path
import torch
from State_batch import State_batch
from DQN import lDQN, weights_init_orthogonal


class Controller:

    def __init__(self, trained_controller_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = lDQN().to(self.device)
        self.target_net = lDQN().to(self.device)
        self.weights_path = trained_controller_weights_path
        self.load_target_net_from_memory()

    def load_target_net_from_memory(self):
        model_path = torch.load(os.path.join(self.weights_path, 'controller_model.pt'),
                                map_location=self.device)
        self.policy_net.load_state_dict(model_path)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, goal_map, action_mask):

        with torch.no_grad():
            env_map = goal_map.clone()
            # action_mask = environment.get_action_mask()
            state = State_batch(env_map.to(self.device), None)
            action_values = self.policy_net(state).squeeze()
            action_values[torch.logical_not(torch.from_numpy(action_mask).bool())[0]] = -3.40e+38
            action_id = action_values.argmax().unsqueeze(0)
        return action_id
