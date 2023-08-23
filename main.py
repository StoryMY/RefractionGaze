import os
import json
import argparse
from argparse import Namespace
import torch
import random
import numpy as np
from trainer import Trainer
from tester import Tester
from dataloader import get_loader
from networks import MLPGazeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_test(args):
    # model
    net = MLPGazeNet()

    # Train
    train_dataloader = get_loader(args, args.train_h5, is_train=True)
    trainer = Trainer(args, train_dataloader, net, device)
    trainer.train()

    # Test
    log_dir = trainer.log_dir
    # log_dir = <path/to/log_dir>
    for test_h5 in args.test_h5_list:
        save_name = os.path.split(test_h5)[-1][:-3]
        test_dataloader = get_loader(args, test_h5, is_train=False)
        tester = Tester(args, test_dataloader, net, log_dir,
                        os.path.join(log_dir, 'checkpoint', '%05d.pt' % args.epoch_num), 
                        save_name,
                        device)
        tester.test()

def test(args):
    # model
    net = MLPGazeNet()

    log_dir = args.pre_dir
    for test_h5 in args.test_h5_list:
        save_name = os.path.split(test_h5)[-1][:-3]
        test_dataloader = get_loader(args, test_h5, is_train=False)
        tester = Tester(args, test_dataloader, net, log_dir,
                        os.path.join(log_dir, 'checkpoint', '%05d.pt' % args.epoch_num), 
                        save_name,
                        device)
        tester.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/001.json", help='Path to config json')
    parser.add_argument('--pre_dir', type=str, default="", help='Path to pretrained directory')
    args = parser.parse_args()
    print(args.config)

    # Load basic config
    with open(args.config, 'r') as f:
        base_config = json.load(f)
    # Merge config
    config = {**base_config, **vars(args)}
    config = Namespace(**config)
    print(config)

    if config.pre_dir == "":
        train_test(config)
    else:
        test(config)
