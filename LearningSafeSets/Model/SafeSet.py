## Copyright 2020 Alexander Liniger

## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at

##     http://www.apache.org/licenses/LICENSE-2.0

## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
###########################################################################
###########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeSet(nn.Module):
    def __init__(self,config):
        super(SafeSet, self).__init__()

        self.n_neurons = config['n_neurons']

        self.n_batch = config['n_batch']
        self.n_inputs = config['n_states']


        if config['activation'] == "Softplus":
            self.model = nn.Sequential(
                nn.Linear(self.n_inputs, self.n_neurons),
                nn.Softplus(beta=1),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.Softplus(beta=1),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.Softplus(beta=1),
                nn.Linear(self.n_neurons, 1),
                nn.Sigmoid()
            )
        elif config['activation'] == "Tanh":

            self.model = nn.Sequential(
                nn.Linear(self.n_inputs, self.n_neurons),
                nn.Tanh(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.Tanh(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.Tanh(),
                nn.Linear(self.n_neurons, 1),
                nn.Sigmoid()
            )
        elif config['activation'] == "ReLU":
            self.model = nn.Sequential(
                nn.Linear(self.n_inputs, self.n_neurons),
                nn.ReLU(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.ReLU(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.ReLU(),
                nn.Linear(self.n_neurons, 1),
                nn.Sigmoid()
            )
        elif config['activation'] == "ELU":
            self.model = nn.Sequential(
                nn.Linear(self.n_inputs, self.n_neurons),
                nn.ELU(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.ELU(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.Dropout(0.5),
                nn.ELU(),
                nn.Linear(self.n_neurons, 1),
                nn.Sigmoid()
            )
        elif config['activation'] == "ELU2":
            self.model = nn.Sequential(
                nn.Linear(self.n_inputs, self.n_neurons),
                nn.ELU(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.ELU(),
                nn.Linear(self.n_neurons, 1),
                nn.Sigmoid()
            )
        elif config['activation'] == "ELU6":
            self.model = nn.Sequential(
                nn.Linear(self.n_inputs, self.n_neurons),
                nn.ELU(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.ELU(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.ELU(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.ELU(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.ELU(),
                nn.Linear(self.n_neurons, self.n_neurons),
                nn.ELU(),
                nn.Linear(self.n_neurons, 1),
                nn.Sigmoid()
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(self.n_inputs, self.n_neurons),
                nn.Linear(self.n_neurons, 1),
                nn.Sigmoid()
            )

    def forward(self, input):
        return self.model(input)