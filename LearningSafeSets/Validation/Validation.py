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
import numpy as np
import json


class Validation:
    def __init__(self,config):
        self.config = config
        self.cut_off = config["cut_off"]
        self.data = {}
        self.result_dir = config["result_dir"]

    def validate(self,model,data):
        # model.to(device)
        criterion_bce = nn.BCELoss()
        # criterion = nn.BCEWithLogitsLoss()
        criterion_mse = nn.MSELoss()
        model.eval()

        correct = 0
        # false_safe = 0
        under_approx = 0
        over_approx = 0
        total = 0
        metric_mse = []
        metric_bce = []

        for i in range(data.n_all_batches): 
            state, safe = data.giveBatch(i)

            safe_model = model(state).view(-1)
            safe_model_max = (safe_model >= self.cut_off).type(torch.FloatTensor)

            metric_mse.append(criterion_mse(safe_model, safe).item())
            metric_bce.append(criterion_bce(safe_model, safe).item())
            total += safe.size(0)

            correct += (safe_model_max == safe).sum().item()
            under_approx += (safe_model_max < safe).sum().item()
            over_approx += (safe_model_max > safe).sum().item()

        print('\tMSE: %.4f, BCE: %.4f, Acc: %.4f, UnderApprox: %.4f, OverApprox: %.4f'
              % (np.mean(metric_mse), np.mean(metric_bce), correct / total, under_approx / total, over_approx / total))

        self.data['full_set'] = []
        self.data['full_set'].append({
            'acc': correct / total,
            'under': under_approx / total,
            'over': over_approx / total,
            'total': total,
            'correct': correct,
            'mse': np.mean(metric_mse),
            'bce': np.mean(metric_bce)
        })


    def validateTest(self,model,data):

        criterion_bce = nn.BCELoss()
        criterion_mse = nn.MSELoss()
        model.eval()

        correct = 0
        # false_safe = 0
        under_approx = 0
        over_approx = 0
        total = 0
        metric_mse = []
        metric_bce = []

        for i in range(data.n_train_batches,data.n_all_batches): 
            state, safe = data.giveBatch(i)

            safe_model = model(state).view(-1)
            safe_model_max = (safe_model >= self.cut_off).type(torch.FloatTensor)

            metric_mse.append(criterion_mse(safe_model, safe).item())
            metric_bce.append(criterion_bce(safe_model, safe).item())
            total += safe.size(0)
  
            correct += (safe_model_max == safe).sum().item()
            under_approx += (safe_model_max < safe).sum().item()
            over_approx += (safe_model_max > safe).sum().item()

        print('\tMSE: %.4f, BCE: %.4f, Acc: %.4f, UnderApprox: %.4f, OverApprox: %.4f'
              % (np.mean(metric_mse), np.mean(metric_bce), correct / total, under_approx / total, over_approx / total))

        self.data['val_set'] = []
        self.data['val_set'].append({
            'acc': correct / total,
            'under': under_approx / total,
            'over': over_approx / total,
            'total': total,
            'correct': correct,
            'mse': np.mean(metric_mse),
            'bce': np.mean(metric_bce)
        })

    def validateTestUnseen(self,model,data):

        criterion_bce = nn.BCELoss()
        criterion_mse = nn.MSELoss()
        model.eval()

        correct = 0
        false_safe = 0
        under_approx = 0
        over_approx = 0
        total = 0
        metric_mse = []
        metric_bce = []

        for i in range(self.config['NGKAPPA_T']):
            state, safe = data.giveTest(i)

            safe_model = model(state).view(-1)
            safe_model_max = (safe_model >= self.cut_off).type(torch.FloatTensor)

            metric_mse.append(criterion_mse(safe_model, safe).item())
            metric_bce.append(criterion_bce(safe_model, safe).item())
            total += safe.size(0)

            correct += (safe_model_max == safe).sum().item()
            under_approx += (safe_model_max < safe).sum().item()
            over_approx += (safe_model_max > safe).sum().item()

            name = self.result_dir+"/RobustInv-Pred-"+str(i)+".bin"
            fh = open(name, "bw")
            safe_model_max.detach().numpy().astype(bool).tofile(fh)


        print('\tMSE: %.4f, BCE: %.4f, Acc: %.4f, UnderApprox: %.4f, OverApprox: %.4f'
                % (np.mean(metric_mse), np.mean(metric_bce),correct/total,under_approx/total,over_approx/total))

        self.data['test_set'] = []
        self.data['test_set'].append({
            'acc': correct / total,
            'under': under_approx / total,
            'over': over_approx / total,
            'total': total,
            'correct': correct,
            'mse': np.mean(metric_mse),
            'bce': np.mean(metric_bce)
        })

    def save_val(self):
        with open(self.result_dir + '/val.txt', 'w') as outfile:
            json.dump(self.data, outfile, indent=4)

    def save_model(self,model):
        model_dict = {}
        k = 0
        for i in range(len(model._modules['model']._modules)):
            if len(model._modules['model']._modules[str(i)]._parameters) > 0:
                W = model._modules['model']._modules[str(i)]._parameters['weight'].data.detach().numpy().tolist()
                b = model._modules['model']._modules[str(i)]._parameters['bias'].data.detach().numpy().tolist()
                model_dict[str(k)] = []
                model_dict[str(k)].append({
                    'W': W,
                    'b': b })
                k+=1

        model_dict["length"] = k

        with open(self.result_dir+'/model.txt', 'w') as outfile:
            json.dump(model_dict, outfile, indent=4)
