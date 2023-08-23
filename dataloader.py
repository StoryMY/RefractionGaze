import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py

class OnePersonDataset(Dataset):
    def __init__(self, root):
        self.cpvalues = np.array(root['cpvalue'])
        self.poses = np.array(root['pose'])
        self.gazes = np.array(root['gaze'])
        self.diopters = np.array(root['diopter'])

    def __getitem__(self, index):
        cpvalue = torch.from_numpy(self.cpvalues[index])
        pose = torch.from_numpy(self.poses[index])
        gaze = torch.from_numpy(self.gazes[index])
        diopter = torch.from_numpy(np.array([self.diopters[index]]))

        return cpvalue, pose, gaze, diopter

    def __len__(self) -> int:
        return len(self.cpvalues)


def create_dataset(args, h5_path):
    root = h5py.File(h5_path, 'r')
    ret_dataset = OnePersonDataset(root)
    root.close()

    return ret_dataset


def get_loader(args, h5_path, is_train):
    # return the DataLoader
    dataset = create_dataset(args, h5_path)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch,
                            shuffle=is_train, num_workers=args.num_workers)

    return dataloader
