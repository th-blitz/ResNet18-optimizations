# HPML ResNet18 Experiment ( Labs 2 and 5 )
### High Performance Machine Learning ( HPML ) Lab <br>
By Preetham Rakshith (c) 2024 pp2959

A project to profile, benchmark and optimize ResNet-18 DDP training with PyTorch, Slurm, and Singularity scalable across 4 V100 nodes on NYU Greene HPC. 

## Benchmark results
The log files are as follows :<br>
- lab5/gpus1.out : Output for using 1 GPUs for batch sizes of 32, 128, 512, 2048 and 8192 / gpu.<br>
- lab5/gpus2.out : Output for using 2 GPUs for batch sizes of 32, 128, 512, 2048 and 8192 / gpu.<br>
- lab5/gpus4.out : Output for using 4 GPUs for batch sizes of 32, 128, 512, 2048 and 8192 / gpu.<br>
- lab5/q4.out : Output for using 4 GPUs with batch size of 512 / gpu ( best batch size ) for q4.<br>

## Benchmark Summary

<img width="807" alt="Screenshot 2025-06-19 at 12 12 31 PM" src="https://github.com/user-attachments/assets/d7256bf5-74be-4568-b315-1b2ff9085411" />

From the above measurements we are looking at strong scaling behaviour. The problem size which is our total size of the dataset remains unchanged as the number of processes increases which is the number of gpus. With increase in the number of gpus with respect to the batch size the total train times per epoch reduces, thus this is a strong scaling behaviour.

In case of weak scaling, we could double the dataset size for 2-GPUs and keep it the same for 1-GPU. In this case, the total train time will be doubled in theory, that is for 2-GPU with batch size of 512/gpu the train time would be 10.20 x 2 = 20.4 seconds. And for 1-GPU with batch size of 512/gpu the train time will remain the same, 18.07 seconds. Therefore, the speedup will be 0.88 which is less than the speedup of 1.77 with strong scaling. Similarly with 4-GPUs and batch size of 8192/gpu, the train times can be calculated as 5.02 * 4 = 20.08 seconds. With a speedup of 18.34/20.08 = 0.91. Thus, we can observe that speedups with weak scaling would be less than 1 in most cases based on our data. And thus strong scaling is better than weak scaling in this instance.

## Instructions to run the experiments
Required libraries
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install tqdm
```
## To run all Codes from C1 to C8:
### Option (1) : If you are inside a singularity container with an active conda env and pytorch installed: 
```
source lab2.batch '~/path/to/my/dataset/dir'
```
### Option (2) : If you want to submit an sbatch:
open run.sbatch file and replace the 2 variables ```SINGULARITY_IMAGE``` and ```OVERLAY_FILE``` with your appropriate singularity image and overlay file.<br>
Then replace the ```DATASET_PATH``` variable with your dataset path and run the run.sbatch file.
```
sbatch run.sbatch
```
## The outputs will be saved to a pp2959.out file.
---
## lab2.py arguments examples
```
python3 lab2.py --device=cuda --num-of-workers=2 --epochs=5 --optimizer=sgd --batch-size=128 --dataset-path='path/to/dataset'  
```
# and their possible values
## --device ( default : cuda )
```
cuda
```
```
cpu
```
## --optimizer ( default : sgd )
```
sgd
```
```
sgd-nesterov
```
```
adagrad
```
```
adadelta
```
```
adam
```
## To run C3 use:
```
python3 lab2.py --device=cuda --epochs=5 --optimizer=sgd --dataset-path=$DATASET_PATH --run-code-3=true --max-num-of-workers-for-code-3=12 --print-epochs=false 
```
## --run-code-3 ( default : false ) or --print-epochs ( default : true )
```
true
```
```
false
```
## To run C7 use:
```
python3 lab2.py --device=cuda --epochs=5 --optimizer=sgd --num-of-workers=4 --dataset-path=$DATASET_PATH --no-batch-norm=true
```
## --no-batch-norm ( default : false ) 
( to train without batchnorm2d layers in CNNs )
```
true
```
```
false
```
## To run Extra Credits:
```
python3 lab2.py --device=cuda --optimizer=sgd --num-of-workers=4 --dataset-path=$DATASET_PATH --run-profiler=true --trace-file-name=trace
```
## --run-profiler ( default : false )
## --trace-file-name ( pass `file_name` to save trace info in a `file_name`.json file )




