import torch
import torch.nn as nn
import torchvision
import lightly
import pytorch_lightning as pl

import argparse
from pathlib import Path
from os.path import exists
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from transforms import train_classifier_transforms, test_transforms
from models import Classifier2, SimpleResnet

from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings("ignore")

import torchvision.datasets as datasets
from NoisyCIFAR10Dataset import NoisyCIFAR10
from SimpleClassifier import SimpleResnetClassifier

pl.seed_everything(seed=42)

parser = argparse.ArgumentParser(description='Classifier Training')
parser.add_argument('--data', type=Path, default='data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--max-epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--noise-rate', default=0.01, type=float, metavar='NR',
                    help='training data noise rate')
parser.add_argument('--noise-type', default='sym', type=str,
                    help='training data noise type')
parser.add_argument('--backbone-model', default='resnet-18', type=str,
                    help='backbone model architechture')
parser.add_argument('--num-fltrs', default=512, type=int, metavar='N',
                    help='number of filters')
parser.add_argument('--lr', default=0.01, type=float, metavar='N',
                    help='learning rate') # 0.1
parser.add_argument('--momentum', default=0.9, type=float, metavar='N',
                    help='momentum')
parser.add_argument('--weight-decay', default=0.0001, type=float, metavar='N',
                    help='weight decay')
parser.add_argument('--progress-refresh-rate', default=100, type=int, metavar='N',
                    help='progress bar refresh rate')
parser.add_argument('--crop-size', default=32, type=int, metavar='N',
                    help='image crop size for transform')
parser.add_argument('--num-classes', default=10, type=int, metavar='N',
                    help='num classes in training data')

args = parser.parse_args()

ds = NoisyCIFAR10(root='./data', 
                         train=True, 
                         download=True, 
                         noise_type = 'sym', 
                         noise_rate=0.0, 
                         split_ratio=0.5, 
                         transform=train_classifier_transforms)
dataset_train_classifier = ds.pretrn_cifar10
# clf_noisy_dataset = ds

dataset_test = datasets.CIFAR10(root=args.data,
                                train=False, 
                                download=True,
                                transform=test_transforms)

dataset_train_classifier = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_classifier) 
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_test)

dataloader_train_classifier = torch.utils.data.DataLoader(
                                        dataset_train_classifier,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=args.num_workers)

dataloader_test = torch.utils.data.DataLoader(
                                        dataset_test,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=args.num_workers)

gpus = 1 if torch.cuda.is_available() else 0

model = SimpleResnetClassifier(lr=args.lr, 
                               weight_decay=args.weight_decay,
                               momentum=args.momentum,
                               max_epochs=args.max_epochs)
# Train on 50% of the data with clean labels
trainer = pl.Trainer(max_epochs=args.max_epochs, 
                     gpus=gpus, 
                     progress_bar_refresh_rate=args.progress_refresh_rate)
trainer.fit(
    model,
    dataloader_train_classifier,
    dataloader_test
)

val_acc=model.last_acc.cpu()
print(val_acc)


model.resnet

