'''
source: https://github.com/IgorSusmelj/barlowtwins
'''

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly

import argparse
from pathlib import Path

from dataloader import NoisyCIFAR10
from models import BarlowTwinsLoss, BartonTwins
from transforms import train_classifier_transforms, test_transforms

pl.seed_everything(seed=42)

parser = argparse.ArgumentParser(description='BarlowTwins Training')
parser.add_argument('--data', type=Path, default='data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--max-epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--num-fltrs', default=2048, type=int, metavar='N',
                    help='number of filters') # 512
parser.add_argument('--backbone-model', default='resnet-50', type=str,
                    help='backbone model architechture') # resnet-18
parser.add_argument('--lr', default=6e-2, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='N',
                    help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='N',
                    help='weight decay')
parser.add_argument('--loss-temperature', default=0.1, type=float, metavar='N',
                    help='loss temperature')
parser.add_argument('--progress-refresh-rate', default=100, type=int, metavar='N',
                    help='progress bar refresh rate')
parser.add_argument('--crop-size', default=32, type=int, metavar='N',
                    help='image crop size for transform')
parser.add_argument('--num-classes', default=10, type=int, metavar='N',
                    help='num classes in training data')
parser.add_argument('--knn-k', default=200, type=int, metavar='N',
                    help='Number of k neighbors used for kNN')
parser.add_argument('--knn-t', default=0.1, type=float, metavar='N',
                    help='kNN temperature for reweighting')

args = parser.parse_args()

# # MoCo v2 uses SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=args.crop_size,
    gaussian_blur=0.,
)

# load train/test data
dataset_train_ssl = NoisyCIFAR10(root=args.data, 
                       train=True, 
                       download=True, 
                       noise_rate=0.0)

dataset_train_kNN = NoisyCIFAR10(root=args.data, 
                       train=True, 
                       download=True, 
                       noise_rate=0.0,
                       transform=test_transforms)

dataset_test = NoisyCIFAR10(root=args.data, 
                       train=False, 
                       download=True, 
                       noise_rate=0.0,
                       transform=test_transforms)

dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_ssl) 
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_kNN)
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_test)

dataloader_train_ssl = torch.utils.data.DataLoader(
    dataset_train_ssl,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=args.num_workers
)

dataloader_train_kNN = torch.utils.data.DataLoader(
    dataset_train_kNN,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers
)

## use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

# Train BarlowTwins model
model = BartonTwins(dataloader_train_kNN, 
                    gpus=gpus, 
                    backbone_type=args.backbone_model,
                    classes=args.num_classes, 
                    knn_k=args.knn_k, 
                    knn_t=args.knn_t,
                    max_epochs=args.max_epochs,
                    num_ftrs=args.num_fltrs,
                    num_mlp_layers=3,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)

trainer = pl.Trainer(max_epochs=args.max_epochs, 
                     gpus=gpus, 
                     progress_bar_refresh_rate=args.progress_refresh_rate)
trainer.fit(
    model,
    train_dataloader=dataloader_train_ssl,
    val_dataloaders=dataloader_test
)

print(f'Highest test accuracy: {model.max_accuracy:.4f}')

# python training_scripts/train_barlowtwins.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1