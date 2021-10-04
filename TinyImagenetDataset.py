import torch, os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader


class TinyImagenet(Dataset):
    """Tiny-imagenet-200 dataste"""

    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (string): Path to the images.
            split (string): train/val/test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transforms.ToTensor() if transform is None else transform
        self.split = split
        self.ds = datasets.ImageFolder(os.path.join(data_dir, split), transform)
        
        self.data = [x[0] for x in self.ds]
        self.targets = [x[1] for x in self.ds]
        
    def __len__(self):
        assert len(self.data) == len(self.ds)
        return len(self.ds)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
            
        return image, target

## Usage example
# from TinyImagenetDataset import TinyImagenet

# import torch
# from torch.utils.data import DataLoader

# data_dir = '/usr/local/data02/zahra/datasets/tiny-imagenet-200'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tiny_ds=TinyImagenet(data_dir, split='train')
# print(len(tiny_ds))

# dl = DataLoader(tiny_ds, batch_size=256, shuffle=False, num_workers=4)
# for img,tgt in dl:
#     img.to(device)
#     tgt.to(device)
    
#     print(img.shape)
#     print(tgt.shape)
#     break
