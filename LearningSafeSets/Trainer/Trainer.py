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
import torch.optim as optim
import numpy as np
from Validation import Validation


class Trainer:
    def __init__(self,config):

        self.n_epochs = config['epochs']
        self.learning_rate = config['lr']

        self.validation = Validation.Validation(config)


    def train(self,model,data):

        criterion = nn.BCELoss()

        metric_mse = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        model.train()

        avg_loss = 0
        avg_acc = 0
        for epoch in range(self.n_epochs):
            data.shuffleTrainSet()
            for i in range(data.n_train_batches):
                state, safe = data.giveBatch(i)
                model.zero_grad()

                safe_model = model(state)

                loss = criterion(safe_model.view(-1), safe)

                loss.backward()
                optimizer.step()

                metric = metric_mse(safe_model.view(-1), safe).item()
                safe_model_max = (safe_model.view(-1).detach() >= 0.5).type(torch.FloatTensor)
                acc = (safe_model_max == safe).sum().item()/data.n_batch

                running_loss = loss.mean().item()
                l_filter = 0.01
                avg_loss = (1 - l_filter) * avg_loss + l_filter * running_loss
                avg_acc =  (1 - l_filter) * avg_acc + l_filter * acc


                if i % 50 == 0:
                    print('[%d/%d][%d/%d] \tLoss: %.4f AvgLoss: %.4f, MSE: %.4f, Acc: %.4f, Avg_Acc: %.4f'
                          % (epoch + 1, self.n_epochs, i+1, data.n_train_batches, running_loss, avg_loss,metric,acc,avg_acc))


            scheduler.step()
            print(optimizer.param_groups[0]['lr'])
            # self.validation.validateTest(model,data)