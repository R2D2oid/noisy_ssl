#! /bin/bash

MAX_EPOCH_SSL=200
MAX_EPOCH_CLF=200
MAX_EPOCH_WO_SSL=1000
BATCH_SIZE=512
NUM_FLTRS=512
DATA_DIR="data"

###############################################
############ Classifier training ##############
###############################################

noise_type="$1"		#"sym"
noise_rate="$2"		#"0.1"
pretrain_type="$3" 	# "barlowtwins" "moco"

source env_ssl/bin/activate

# SSL_MODEL_CHECKPOINT_PATH=lightning_logs/"bt_resnet18_b"$BATCH_SIZE"_e"$MAX_EPOCH_SSL"/checkpoints/epoch\=99-step\=9699.ckpt"
SSL_MODEL_CHECKPOINT_PATH=lightning_logs/"moco_resnet18_b"$BATCH_SIZE"_e"$MAX_EPOCH_SSL"/checkpoints/epoch\=199-step\=19399.ckpt"
EXP_DIR=lightning_logs/"clf_bt_resnet18_b"$BATCH_SIZE"_e"$MAX_EPOCH_CLF"_"$noise_type"_"$noise_rate

if [ -d "$EXP_DIR" ]; then
        echo $EXP_DIR" already exists!"
else
	echo "*** Runing experiment"
	echo $SLURM_JOBID
	echo "python training_scripts/train_classifier.py --max-epochs "$MAX_EPOCH_SSL" --batch-size "$BATCH_SIZE" --backbone-model resnet-18 --num-fltrs "$NUM_FLTRS" --progress-refresh-rate 1 --data "$DATA_DIR" --checkpoint "$SSL_MODEL_CHECKPOINT_PATH" --pretrained-ssl-model "$pretrain_type" --noise-rate "$noise_rate" --noise-type "$noise_type
	### train classifier on BarlowTwins pretrained ssl resnet-18
	python training_scripts/train_classifier.py --max-epochs $MAX_EPOCH_CLF --batch-size $BATCH_SIZE --backbone-model resnet-18 --num-fltrs $NUM_FLTRS --progress-refresh-rate 1 --data $DATA_DIR --checkpoint $SSL_MODEL_CHECKPOINT_PATH --pretrained-ssl-model $pretrain_type --noise-rate $noise_rate --noise-type $noise_type

	mv "lightning_logs/version_"$SLURM_JOBID $EXP_DIR

fi
