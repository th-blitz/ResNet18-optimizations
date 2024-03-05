#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=pp2959hpml2
#SBATCH --output=pp2959.out

#SBATCH --mail-type=END
#SBATCH --mail-user=pp2959@nyu.edu

module purge

SINGULARITY_IMAGE=~/hpml-assignment-2/cuda12.2

OVERLAY_FILE=~/hpml-assignment-2/overlay-15

DATASET_PATH='..dataset'

singularity exec --nv --overlay $OVERLAY_FILE:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; conda activate pytorch-0; source lab2.batch $DATASET_PATH"

