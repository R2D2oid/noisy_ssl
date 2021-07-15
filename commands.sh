#! /bin/bash

# train moco with resnet-18 -> lightening_logs/moco-resnet18-b512-e100
python training_scripts/train_moco.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1

# train barlowtwins with resnet-18 -> lightening_logs/bt-resnet18-b512-e100
python training_scripts/train_barlowtwins.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1

###############################################
############ Classifier training ##############
############ 	    MoCo         ##############
###############################################
# train classifier on MoCo pretrained ssl resnet-18
python training_scripts/train_classifier.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --checkpoint lightning_logs/moco_resnet18_b512_e100/checkpoints/epoch\=99-step\=9699.ckpt --pretrained-ssl-model moco

# noise sym 0.1
python training_scripts/train_classifier.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --checkpoint lightning_logs/moco_resnet18_b512_e100/checkpoints/epoch\=99-step\=9699.ckpt --pretrained-ssl-model moco --noise-rate 0.1 --noise-type sym

# noise sym 0.3
python training_scripts/train_classifier.py --max-epochs 200 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --checkpoint lightning_logs/moco_resnet18_b512_e100/checkpoints/epoch\=99-step\=9699.ckpt --pretrained-ssl-model moco --noise-rate 0.3 --noise-type sym

# noise sym 0.5
python training_scripts/train_classifier.py --max-epochs 200 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --checkpoint lightning_logs/moco_resnet18_b512_e100/checkpoints/epoch\=99-step\=9699.ckpt --pretrained-ssl-model moco --noise-rate 0.5 --noise-type sym

# noise sym 0.7
python training_scripts/train_classifier.py --max-epochs 200 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --checkpoint lightning_logs/moco_resnet18_b512_e100/checkpoints/epoch\=99-step\=9699.ckpt --pretrained-ssl-model moco --noise-rate 0.7 --noise-type sym

# noise sym 0.9
python training_scripts/train_classifier.py --max-epochs 200 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --checkpoint lightning_logs/moco_resnet18_b512_e100/checkpoints/epoch\=99-step\=9699.ckpt --pretrained-ssl-model moco --noise-rate 0.9 --noise-type sym
###############################################
############ Classifier training ##############
############ 	BarlowTwins      ##############
###############################################
# train classifier on BarlowTwins pretrained ssl resnet-18
python training_scripts/train_classifier.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --checkpoint lightning_logs/bt_resnet18_b512_e100/checkpoints/epoch\=99-step\=9699.ckpt --pretrained-ssl-model barlowtwins

# noise sym 0.1
python training_scripts/train_classifier.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --checkpoint lightning_logs/bt_resnet18_b512_e100/checkpoints/epoch\=99-step\=9699.ckpt --pretrained-ssl-model barlowtwins --noise-rate 0.1 --noise-type sym

###############################################
############ Classifier training ##############
############ 	without SSL      ##############
###############################################

# train classifier on resnet-18 without ssl
python training_scripts/train_classifier.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --pretrained-ssl-model only_supervised

# noise sym 0.1
python training_scripts/train_classifier.py --max-epochs 100 --batch-size 512 --backbone-model resnet-18 --num-fltrs 512 --progress-refresh-rate 1 --pretrained-ssl-model only_supervised --noise-rate 0.1 --noise-type sym
