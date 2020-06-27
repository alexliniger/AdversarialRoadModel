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

import numpy as np
import torch


class DataLoader:
    def __init__(self,config):
        self.path = config['path']

        self.n_d = config['NGD']
        self.n_mu = config['NGMU']
        self.n_v = config['NGV']
        self.n_kappa = config['NGKAPPA']

        self.n_batch = config['n_batch']
        self.train_val_ratio = config['train_val_ratio']

        self.n_state = config['n_states']
        self.n_points = self.n_d*self.n_mu*(self.n_v+1)*self.n_kappa
        self.n_train_batches = int(np.floor(self.train_val_ratio * self.n_points / self.n_batch))
        self.n_all_batches = int(np.floor(self.n_points/self.n_batch))

        self.middle = torch.zeros(self.n_state)
        self.half_range = torch.zeros(self.n_state)

        self.state = torch.zeros((0, self.n_state))
        self.safe = torch.zeros((0))

        np.random.seed(123)
        self.rand_index = np.random.choice(self.n_points, self.n_points, replace=False)

        # curvaturs in the training set
        curvature = np.array([0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.004, 0.003, 0.002, 0.0015, 0.00125, 0.001])

        for c in curvature:
            print(c)
            [state_r, safe_r] = self.loadData(1/c, c)
            self.state = torch.cat((self.state, state_r))
            self.safe = torch.cat((self.safe, safe_r))

        curvature = np.array([0.015, 0.0035])

        self.n_d_test = config['NGD_T']
        self.n_mu_test = config['NGMU_T']
        self.n_v_test = config['NGV_T']
        self.n_kappa_test = config['NGKAPPA_T']

        self.n_points_test = self.n_d_test * self.n_mu_test * self.n_v_test * self.n_kappa_test
        self.n_points_test_case = self.n_d_test * self.n_mu_test * self.n_v_test

        self.state_test = torch.zeros((0, self.n_state))
        self.safe_test = torch.zeros((0))

        for c in curvature:
            print(c)
            [state_r, safe_r] = self.loadDataTest(1 / c, c)
            self.state_test = torch.cat((self.state_test, state_r))
            self.safe_test = torch.cat((self.safe_test, safe_r))

        self.findNormalization()

    def loadData(self,radius,kappa):
        name = self.path + "/Disc-" + str(int(np.floor(radius))) + ".bin"
        value_fun_cm = torch.from_numpy(np.fromfile(name, dtype=np.bool, count=-1, sep="")).type(torch.FloatTensor)
        # stat values

        d = torch.linspace(-0.3415, 0.3415, self.n_d)
        mu = torch.linspace(-0.2, 0.2, self.n_mu)
        v = torch.linspace(0, min(np.sqrt(radius * 1.6),35), self.n_v)
        diff_v = v[2] - v[1]
        v_end = v[-1]
        v = np.append(v,  [v_end+diff_v])
        n_points_wo_kappa = self.n_d*self.n_mu*(self.n_v+1)
        # initialize data
        state = torch.zeros((n_points_wo_kappa, self.n_state))
        safe = torch.zeros((n_points_wo_kappa))

        # init running variables
        o = 0
        # transfer column major value function into points
        for i in range(self.n_d):
            for j in range(self.n_mu):
                for k in range(self.n_v+1):

                    if k >= self.n_v:
                        value = 0
                    else:
                        value = value_fun_cm[i + self.n_d * j + self.n_d * self.n_mu * k]

                    state[o, :] = torch.tensor([d[i], mu[j], v[k], kappa])
                    safe[o] = value
                    o = o + 1


        return [state, safe]

    def loadDataTest(self, radius, kappa):
        name = self.path + "/Disc-Test-" + str(int(np.floor(radius))) + ".bin"
        value_fun_cm = torch.from_numpy(np.fromfile(name, dtype=np.bool, count=-1, sep="")).type(torch.FloatTensor)
        # stat values

        d = torch.linspace(-0.3415, 0.3415, self.n_d_test)
        mu = torch.linspace(-0.2, 0.2, self.n_mu_test)
        v = torch.linspace(0, min(np.sqrt(radius * 1.6), 35), self.n_v_test)
        n_points_wo_kappa = self.n_d_test * self.n_mu_test * self.n_v_test
        # initialize data
        state = torch.zeros((n_points_wo_kappa, self.n_state))
        safe = torch.zeros((n_points_wo_kappa))

        # init running variables
        o = 0
        # transfer column major value function into points
        for i in range(self.n_d_test):
            for j in range(self.n_mu_test):
                for k in range(self.n_v_test):
                    value = value_fun_cm[i + self.n_d_test * j + self.n_d_test * self.n_mu_test * k]
                    state[o, :] = torch.tensor([d[i], mu[j], v[k], kappa])
                    safe[o] = value
                    o = o + 1

        return [state, safe]

    def findNormalization(self):
        max_state = np.max(self.state.numpy(), 0)
        min_state = np.min(self.state.numpy(), 0)
        self.middle = torch.from_numpy(min_state + (max_state - min_state) / 2)
        self.half_range = torch.from_numpy((max_state - min_state) / 2)

    def normalize(self,state):
        return (state - self.middle)/self.half_range

    def denomralize(self,state):
        return state*self.half_range + self.middle

    def setBatchSize(self,n_batch,train_val_ratio):
        self.n_batch = n_batch
        self.train_val_ratio = train_val_ratio
        self.n_train_batches = int(np.floor(self.train_val_ratio * self.n_points / self.n_batch))
        self.n_all_batches = int(np.floor(self.n_points / self.n_batch))

    def giveBatch(self, i):
        batch_index = range(i * self.n_batch, (i + 1) * self.n_batch)
        rand_batch_index = self.rand_index[batch_index]
        return self.normalize(self.state[rand_batch_index, :]),self.safe[rand_batch_index]

    def giveTest(self, i):
        batch_index = range(i * self.n_points_test_case, (i + 1) * self.n_points_test_case)
        return self.normalize(self.state_test[batch_index, :]),self.safe_test[batch_index]

    def shuffleTrainSet(self):
        train_data_points = self.rand_index[range(self.n_train_batches*self.n_batch)]
        np.random.shuffle(train_data_points)
        self.rand_index[range(self.n_train_batches*self.n_batch)] = train_data_points

