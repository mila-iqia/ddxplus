import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class ASDMLP(nn.Module):
    def __init__(
        self,
        inp_size,
        hidden_sizes,
        action_state_size,
        symptom_prediction_size,
        patho_size=None
    ):
        super(ASDMLP, self).__init__()

        self.fc_net = [nn.Linear(inp_size, hidden_sizes[0])]
        for size_idx in range(1, len(hidden_sizes)):
            self.fc_net.append(
                nn.Linear(hidden_sizes[size_idx - 1], hidden_sizes[size_idx])
            )
        self.fc_net = nn.ModuleList(self.fc_net)
        self.action_branch = nn.Linear(hidden_sizes[-1], action_state_size)
        self.sympt_branch = nn.Linear(hidden_sizes[-1], symptom_prediction_size)
        self.patho_branch = (
            None
            if (patho_size is None) or (patho_size == 0)
            else nn.Linear(hidden_sizes[-1], patho_size)
        )

    def forward(self, x_sym, x_ag):
        # add hidden layer, with relu activation function
        for layer in self.fc_net:
            x_ag = F.relu(layer(x_ag))
            x_sym = F.relu(layer(x_sym))
        action_prediction = self.action_branch(x_ag)
        sympt_prediction = self.sympt_branch(x_sym)
        patho_prediction = (
            None
            if self.patho_branch is None
            else self.patho_branch(x_ag)
        )
        return sympt_prediction, action_prediction, patho_prediction
