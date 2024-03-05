# HPML-2
### High Performance Machine Learning ( HPML ) Assignment 2 <br>
By Preetham Rakshith (c) 2024 pp2959

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




