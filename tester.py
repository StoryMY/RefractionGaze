import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import utils

class Tester:
    def __init__(self, args, dataloader, net, log_dir, ckpt_path, save_name, device):
        # Options
        self.args = args
        self.device = device

        # DataLoader
        self.data_loader = dataloader

        self.Net = net.to(device)

        # Criterion
        self.criterionL1 = nn.L1Loss()
        self.criterionMSE = nn.MSELoss()

        self.log_dir = log_dir
        self.ckpt_path = ckpt_path
        self.save_name = save_name

    def test(self):
        ckpt = torch.load(self.ckpt_path)
        self.Net.load_state_dict(ckpt['Net'])
        self.Net.eval()
        
        total_pitch_err = 0.0
        total_yaw_err = 0.0
        total_angle_err = 0.0
        gt_arr = []
        pred_arr = []
        pose_arr = []
        diopter_arr = []
        total_num = 0
        with torch.no_grad():
            for step, (cps, poses, gazes, diopters) in enumerate(self.data_loader):
                self.cps = cps.to(self.device)
                self.poses = poses.to(self.device)
                self.gazes = gazes.to(self.device)
                self.diopters = diopters.to(self.device)

                self.pred_gaze = self.Net(self.cps, self.poses)

                for i in range(self.cps.shape[0]):
                    gt = self.gazes[i].cpu().numpy()
                    pred = self.pred_gaze[i].cpu().numpy()
                    pose = self.poses[i].cpu().numpy()
                    diopter = self.diopters[i].cpu().numpy()

                    pitch_err = np.abs(pred[0] - gt[0])
                    yaw_err = np.abs(pred[1] - gt[1])
                    angle_err = utils.compute_angle_error(pred, gt)

                    gt_arr.append(gt)
                    pred_arr.append(pred)
                    pose_arr.append(pose)
                    diopter_arr.append(diopter)
                    total_pitch_err += pitch_err
                    total_yaw_err += yaw_err
                    total_angle_err += angle_err
                    total_num += 1
        
        mean_pitch_err = total_pitch_err / total_num * 180 / np.pi
        mean_yaw_err = total_yaw_err / total_num * 180 / np.pi
        mean_angle_err = total_angle_err / total_num * 180 / np.pi
        print('----------------------')
        print('pitch:', mean_pitch_err)
        print('yaw:', mean_yaw_err)
        print('gaze:', mean_angle_err)

        # save
        save_path = os.path.join(self.log_dir, self.save_name + '.npy')
        save_obj = {}
        save_obj['mean_pitch_err'] = mean_pitch_err
        save_obj['mean_yaw_err'] = mean_yaw_err
        save_obj['mean_angle_err'] = mean_angle_err
        save_obj['gt'] = gt_arr
        save_obj['pred'] = pred_arr
        save_obj['hpose'] = pose_arr
        save_obj['diopter'] = diopter_arr
        np.save(save_path, save_obj)
