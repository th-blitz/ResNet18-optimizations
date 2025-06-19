#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|v100"
#SBATCH --time=08:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=gpus1.out
#SBATCH --output=gpus1.out

#SBATCH --mail-type=END
#SBATCH --mail-user=pp2959@nyu.edu

module purge

SINGULARITY_IMAGE=~/hpml-assignment-2/cuda12.2

OVERLAY_FILE=~/hpml-assignment-2/overlay-15

singularity exec --nv --overlay $OVERLAY_FILE:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; conda activate pytorch-0; source run0.batch"

