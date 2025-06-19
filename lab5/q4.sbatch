#!/bin/bash


#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100|v100|h100"
#SBATCH --time=08:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=q4.out
#SBATCH --output=q4.out

#SBATCH --mail-type=END
#SBATCH --mail-user=pp2959@nyu.edu

module purge

SINGULARITY_IMAGE=~/hpml-assignment-2/cuda12.2

OVERLAY_FILE=~/hpml-assignment-2/overlay-15

singularity exec --nv --overlay $OVERLAY_FILE:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; conda activate pytorch-0; source q4.batch"

