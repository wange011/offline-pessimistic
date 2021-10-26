import torch

import argparse
import json
import os
from copy import deepcopy

from trainer import Trainer, MultiTrainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('alg', type=str, choices=['safari'])
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--n_updates', type=int, default=40)
    parser.add_argument('--n_samples', type=int, default=32)
    parser.add_argument('--no_parallel', dest='parallel', action='store_false')
    parser.add_argument('--max_parallel', type=int, default=3)
    parser.add_argument('--device', type=str, default='0')
    parser.set_defaults(parallel=True)
    args = parser.parse_args()

    if args.alg == 'safari':
        from configs import config_safari
        config = config_safari.get_config()
    else:
        raise NotImplementedError

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() and config.main.cuda else "cpu")

    log_path = os.path.join(os.getcwd(), "results", config.main.dir_main)
    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    if args.n_seeds > 1:
        seeds = range(args.n_seeds)
        config_list = []
        for seed in seeds:
            config_copy = deepcopy(config)
            config_copy.main.seed = seed
            config_copy.main.dir_main = config.main.dir_main + "_{}".format(seed)
            
            if 'dir_data' in config.main:
                config_copy.main.dir_data = config.main.dir_data + "/{}".format(seed)
            
            config_list.append(config_copy)
        trainer = MultiTrainer(config_list, device)
        torch.multiprocessing.set_start_method('spawn')
        trainer.train(args.max_parallel)
    else:
        if 'dir_data' in config.main:
            config.main.dir_data = config.main.dir_data + "/{}".format(0)

        trainer = Trainer(config, device)
        trainer.train()
