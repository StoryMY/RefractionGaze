import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import datetime
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def smart_mkdir(dir, config_path):
    config_name = os.path.split(config_path)[-1][:-5]  # remove prefix (directoy) and suffix (.json)
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "") \
        .replace(":", "") \
        .replace(" ", "_")
    log_dir = os.path.join(dir, config_name + "_" + date)
    ckpt_dir = os.path.join(log_dir, 'checkpoint')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    return log_dir, ckpt_dir

class Trainer:
    def __init__(self, args, dataloader, net, device):
        # Options
        self.args = args
        self.device = device

        # DataLoader
        self.data_loader = dataloader

        self.Net = net.to(device)
        self.optim = torch.optim.Adam(self.Net.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lambda x: args.lr_ramp**(float(x)/float(args.epoch_num)))

        # Criterion
        self.criterionL1 = nn.L1Loss()
        self.criterionMSE = nn.MSELoss()

        self.log_dir, self.ckpt_dir = smart_mkdir(args.log_dir, args.config)
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    def train(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

        for epoch in range(self.args.epoch_num):
            total_loss = 0
            total_step = 0
            for step, (cps, poses, gazes, diopters) in enumerate(self.data_loader):
                self.cps = cps.to(self.device)
                self.poses = poses.to(self.device)
                self.gazes = gazes.to(self.device)

                self.pred_gaze = self.Net(self.cps, self.poses)

                self.optim.zero_grad()
                self.loss_gaze = self.criterionL1(self.pred_gaze, self.gazes)
                self.loss_gaze.backward()
                self.optim.step()

                loss_gaze_val = self.loss_gaze.mean().item()
                total_loss += loss_gaze_val
                total_step += 1
            
            self.scheduler.step()

            if (epoch + 1) % self.args.log_gap == 0:
                mean_loss = total_loss / total_step
                self.writer.add_scalar("loss/loss_gaze", mean_loss, epoch)
                print('Epoch: %d, lr: %f, Loss Gaze: %f' % (epoch + 1, self.optim.param_groups[-1]['lr'], mean_loss))

            if (epoch + 1) % self.args.save_gap == 0:
                torch.save({
                    "Net": self.Net.state_dict(),
                    "args": self.args
                },
                os.path.join(self.ckpt_dir, str(epoch + 1).zfill(5) + '.pt'))

        self.writer.close()
