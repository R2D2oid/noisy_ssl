import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

from dataloader import NoisyCIFAR10
from transforms import train_classifier_transforms, test_transforms
from models import Classifier, MocoModel

pl.seed_everything(seed=42)

parser = argparse.ArgumentParser(description='Classifier Training')
parser.add_argument('--data', type=Path, default='data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--max-epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--noise-rate', default=0.0, type=float, metavar='NR',
                    help='training data noise rate')
parser.add_argument('--noise-type', default='sym', type=str,
                    help='training data noise type')
parser.add_argument('--checkpoint', type=Path, metavar='DIR',
                    help='path to checkpoint')

args = parser.parse_args()

dataset_train_classifier = NoisyCIFAR10(root=args.data, 
                                       train=True, 
                                       download=True, 
                                       noise_type=args.noise_type, 
                                       noise_rate=args.noise_rate, 
                                       transform=train_classifier_transforms)

dataset_test = NoisyCIFAR10(root=args.data, 
                                       train=False, 
                                       download=True,
                                       noise_rate=0.0,
                                       transform=test_transforms)

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

# load MoCo model
model = MocoModel()
model = model.load_from_checkpoint(args.checkpoint)
model.eval()
classifier = Classifier(model.resnet_moco)

# Train classifier using MoCo representations on noisy CIFAR-10 dataset
gpus = 1 if torch.cuda.is_available() else 0
trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=gpus, progress_bar_refresh_rate=100)
trainer.fit(
    classifier,
    dataloader_train_classifier,
    dataloader_test
)