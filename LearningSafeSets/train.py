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

import json
import argparse

from DataLoader import DataLoader
from Model import SafeSet
from Trainer import Trainer
from Validation import Validation

import torch
import matplotlib.pyplot as plt
import random
import numpy as np

def main(config,args):

    config["result_dir"] = args.path
    print(config["result_dir"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    path_data = config['path'] + "/safe_set_data.pth"
    if config['load_data']:
        data_loader = torch.load(path_data)
        data_loader.setBatchSize(config['n_batch'],config['train_val_ratio'])
    else:
        data_loader = DataLoader.DataLoader(config)
        torch.save(data_loader, path_data)

    print(data_loader.n_all_batches)
    model = SafeSet.SafeSet(config)

    path_model = config['path'] + "/safe_set_model.pth"
    path_model_results = config["result_dir"] + "/safe_set_model.pth"
    if not config['load_model']:
        trainer = Trainer.Trainer(config)
        trainer.train(model,data_loader)

        torch.save(model, path_model)
        torch.save(model, path_model_results)
    else:
        model = torch.load(path_model)


    validation = Validation.Validation(config)
    validation.validate(model,data_loader)
    validation.validateTest(model, data_loader)
    validation.validateTestUnseen(model, data_loader)
    validation.save_val()
    validation.save_model(model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-p', '--path', default=Data, type=str,
                        help='run save directory path (default: None)')

    args = parser.parse_args()
    print(args)
    config = json.load(open('config.json'))
    main(config, args)