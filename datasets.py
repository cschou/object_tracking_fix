import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CoordinatesDataset(Dataset):
    def __init__(self, x_file_path, y_file_path):
        """
        Args:
            x_file_path (string): Path to the .npy file of detections with errors.
            y_file_path (string): Path to the .npy file of detections without errors (the groundtruths).
        """

        x_in = np.transpose(np.load(x_file_path), (0, 2, 1))
        y_in = np.transpose(np.load(y_file_path), (0, 2, 1))
        self.x_ = torch.tensor(x_in, dtype=torch.float32)
        self.y_ = torch.tensor(y_in, dtype=torch.float32)
    
    def __len__(self):
        return self.x_.size()[0]
    
    def __getitem__(self, index):
        x = self.x_[index]
        y = self.y_[index]

        return (x, y)
