import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly

import argparse
from pathlib import Path

from dataloader import NoisyCIFAR10
from models import MocoModel

pl.seed_everything(seed=42)

parser = argparse.ArgumentParser(description='MoCo Training')
parser.add_argument('--data', type=Path, default='data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--max-epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--memory-bank-size', default=4096, type=int, metavar='N',
                    help='memory bank size')
parser.add_argument('--backbone-model', default='resnet-50', type=str,
                    help='backbone model architechture')

args = parser.parse_args()

# load train/test data
dataset_train_moco = NoisyCIFAR10(root=args.data, 
                       train=True, 
                       download=True, 
                       noise_rate=0.0 
                    )
dataset_train_moco = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_moco) 

# # MoCo v2 uses SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=args.num_workers
)

## use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

# Train MoCo model
model = MocoModel(backbone_type = args.backbone_model)
trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=gpus, progress_bar_refresh_rate=100)
trainer.fit(
    model,
    dataloader_train_moco
)


### !!!! writes the noisy dataset to disk
# train_dataset.dump_(path_='checkpoint/cifar10_noisy.pkl')
