#!/bin/bash
#SBATCH --account=def-mudl
#SBATCH --nodes 1              
#SBATCH --gres=gpu:1           
#SBATCH --mem=30G            
#SBATCH --time=0-12:00   

source env_ssl/bin/activate

export MAX_EPOCH_SSL=200
export MAX_EPOCH_WO_SSL=1000
export BATCH_SIZE=512
export NUM_FLTRS=512
export DATA_DIR="data"

###########################
###### SSL Training #######
###########################

pretrain_type="$1"      # "barlowtwins"/"moco"

EXP_NAME=$pretrain_type"_resnet18_b"$BATCH_SIZE"_e"$MAX_EPOCH_SSL
EXP_DIR="lightning_logs/"$EXP_NAME

if [ -d "$EXP_DIR" ]; then
        echo $EXP_DIR" already exists!"
else
	echo "*** Runing experiment "$SLURM_JOBID"..."

	if [$pretrain_type=="moco"]; then
		echo "python training_scripts/train_moco.py --max-epochs "$MAX_EPOCH_SSL" --batch-size "$BATCH_SIZE" --backbone-model resnet-18 --num-fltrs "$NUM_FLTRS" --progress-refresh-rate 1 --data "$DATA_DIR
		python training_scripts/train_moco.py --max-epochs $MAX_EPOCH_SSL --batch-size $BATCH_SIZE --backbone-model resnet-18 --num-fltrs $NUM_FLTRS --progress-refresh-rate 1 --data $DATA_DIR
	elif [$pretrain_type=="barlowtwins"]; then
		echo "python training_scripts/train_barlowtwins.py --max-epochs "$MAX_EPOCH_SSL" --batch-size "$BATCH_SIZE" --backbone-model resnet-18 --num-fltrs "$NUM_FLTRS" --progress-refresh-rate 1 --data "$DATA_DIR
		python training_scripts/train_barlowtwins.py --max-epochs $MAX_EPOCH_SSL --batch-size $BATCH_SIZE --backbone-model resnet-18 --num-fltrs $NUM_FLTRS --progress-refresh-rate 1 --data $DATA_DIR
	else
		echo "unknown model type "$pretrain_type"!"
	fi

	mv "lightning_logs/version_"$SLURM_JOBID $EXP_DIR
fi
