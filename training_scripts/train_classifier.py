import argparse
from pathlib import Path
from os.path import exists

import torch
import torch.nn as nn
import torchvision
import lightly
import pytorch_lightning as pl

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataloader import NoisyCIFAR10
from transforms import train_classifier_transforms, test_transforms
from models import Classifier, MocoModel, SimpleResnet, BartonTwins

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
parser.add_argument('--backbone-model', default='resnet-18', type=str,
                    help='backbone model architechture')
parser.add_argument('--pretrained-ssl-model', default='moco', type=str,
                    help='pretrained ssl model type to load') #barlowtwins
parser.add_argument('--num-fltrs', default=512, type=int, metavar='N',
                    help='number of filters')
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

path_ = f'CIFAR10_noisy_checkpoints/cifar10_noise_{args.noise_type}_{str(args.noise_rate)}.pkl'
if exists(path_):
    # loads previously generated dataset for re-use
    print(f'loads previously generated dataset from: {path_}')
    dataset_train_classifier = NoisyCIFAR10.load_(path_)
else:
    dataset_train_classifier = NoisyCIFAR10(root=args.data, 
                                       train=True, 
                                       download=True, 
                                       noise_type=args.noise_type, 
                                       noise_rate=args.noise_rate, 
                                       transform=train_classifier_transforms)

    # save dataset as a pickle for re-use
    print(f'stores noisy dataset at {path_} for future re-use')
    dataset_train_classifier.dump_(path_)

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

dataset_train_classifier = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_classifier) 
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_kNN)
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_test)

dataloader_train_classifier = torch.utils.data.DataLoader(
                                        dataset_train_classifier,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=args.num_workers)

dataloader_train_kNN = torch.utils.data.DataLoader(
                                        dataset_train_kNN,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=args.num_workers)

dataloader_test = torch.utils.data.DataLoader(
                                        dataset_test,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=args.num_workers)

gpus = 1 if torch.cuda.is_available() else 0

if args.pretrained_ssl_model == 'moco':
    print('Loading MoCo Model')
    # load MoCo model
    model = MocoModel()
    model = model.load_from_checkpoint(args.checkpoint)
    model.eval()
    classifier = Classifier(model.resnet_moco, lr=30., max_epochs=args.max_epochs)
elif args.pretrained_ssl_model == 'barlowtwins':
    print('Loading BarlowTwins Model')
    # load BarlowTwins model
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
    model = model.load_from_checkpoint(args.checkpoint, dataloader_kNN=dataloader_train_kNN, gpus=gpus)
    model.eval()
    classifier = Classifier(model.resnet_simsiam, lr=30., max_epochs=args.max_epochs)
elif args.pretrained_ssl_model == 'only_supervised':
    model = SimpleResnet()
    model.eval()
    classifier = Classifier(model, lr=30., max_epochs=args.max_epochs)
else:
    raise NotImplementedError(f'Undefined {args.pretrained_ssl_model}')
    
# Train classifier using ssl representations on noisy CIFAR-10 dataset
trainer = pl.Trainer(max_epochs=args.max_epochs, 
                     gpus=gpus, 
                     progress_bar_refresh_rate=args.progress_refresh_rate)
trainer.fit(
    classifier,
    dataloader_train_classifier,
    dataloader_test
)

