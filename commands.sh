#! /bin/bash

# train moco with resnet-18 -> lightening_logs/moco-resnet18-b512-e100
python training_scripts/train_moco.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1

# train barlowtwins with resnet-18 -> lightening_logs/bt-resnet18-b512-e100
python training_scripts/train_barlowtwins.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1

# train classifier on MoCo pretrained ssl resnet-18
python training_scripts/train_classifier.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --checkpoint lightning_logs/moco_resnet18_b512_e100/checkpoints/epoch\=99-step\=9699.ckpt --pretrained-ssl-model moco

# train classifier on BarlowTwins pretrained ssl resnet-18
python training_scripts/train_classifier.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --checkpoint lightning_logs/bt_resnet18_b512_e100/checkpoints/epoch\=99-step\=9699.ckpt --pretrained-ssl-model barlowtwins

# train classifier on resnet-18 without ssl
python training_scripts/train_classifier.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --pretrained-ssl-model only_supervised
