import argparse
import torch
import torch.nn as nn
import torchvision
import lightly
import pytorch_lightning as pl
from pathlib import Path

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
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

args = parser.parse_args()

print(args)
print('locading cifar dataset')
# load train/test data
dataset_train_moco = NoisyCIFAR10(root=args.data, 
                       train=True, 
                       download=True, 
                       noise_rate=0.0 # contrastive training of MoCo does not use labels so the noise-rate is irrelevant
                    )
print('creating dataloader')
dataset_train_moco = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_moco) 

print('done creating loader')
# # MoCo v2 uses SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=args.crop_size,
    gaussian_blur=0.,
)
print('1111')
dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=args.num_workers
)

## use GPU, if available
gpus = 1 if torch.cuda.is_available() else 0

# Train MoCo model
model = MocoModel(backbone_type = args.backbone_model, 
                max_epochs=args.max_epochs,
                memory_bank_size=args.memory_bank_size,
                num_ftrs = args.num_fltrs, 
                batch_shuffle = True,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                loss_temperature=args.loss_temperature
                )

print('trainer')
trainer = pl.Trainer(max_epochs=args.max_epochs, 
                     gpus=gpus, 
                     progress_bar_refresh_rate=args.progress_refresh_rate
                    )
print('training')
trainer.fit(
    model,
    dataloader_train_moco
)
              
# python training_scripts/train_moco.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1
