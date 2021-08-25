#! /bin/bash

MAX_EPOCH_SSL=200
MAX_EPOCH_CLF=300
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
MAX_EPOCH_CLF=${4:-300}

#module load python/3.6
#virtualenv --system-site-packages -p python3 env_ssl
source env_ssl/bin/activate
#pip install -r requirements.txt

SSL_MODEL_CHECKPOINT_PATH="lightning_logs/"$pretrain_type"_resnet18_b"$BATCH_SIZE"_e"$MAX_EPOCH_SSL"/checkpoints/epoch\=199-step\=19399.ckpt"
EXP_NAME="clf_"$pretrain_type"_resnet18_b"$BATCH_SIZE"_e"$MAX_EPOCH_CLF"_"$noise_type"_"$noise_rate
EXP_DIR="lightning_logs/"$EXP_NAME


if [ -d "$EXP_DIR" ]; then
        echo $EXP_DIR" already exists!"
else
	echo "*** Runing experiment"
	echo $SLURM_JOBID
	echo "python training_scripts/train_classifier.py --max-epochs "$MAX_EPOCH_SSL" --batch-size "$BATCH_SIZE" --backbone-model resnet-18 --num-fltrs "$NUM_FLTRS" --progress-refresh-rate 1 --data "$DATA_DIR" --checkpoint "$SSL_MODEL_CHECKPOINT_PATH" --pretrained-ssl-model "$pretrain_type" --noise-rate "$noise_rate" --noise-type "$noise_type
	python training_scripts/train_classifier.py --max-epochs $MAX_EPOCH_CLF --batch-size $BATCH_SIZE --backbone-model resnet-18 --num-fltrs $NUM_FLTRS --progress-refresh-rate 1 --data $DATA_DIR --checkpoint $SSL_MODEL_CHECKPOINT_PATH --pretrained-ssl-model $pretrain_type --noise-rate $noise_rate --noise-type $noise_type

	mv "lightning_logs/version_"$SLURM_JOBID $EXP_DIR
	mv "slurm-"$SLURM_JOBID".out" "slurm-"$EXP_NAME".out"

fi
