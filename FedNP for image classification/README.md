## This sub-project implements experiments involved in "5.2. Image Classification With Non-IID Image Data"

## Major Files 

`dataset/*.py`:  dateset wrapper
`libs/*.py`,: implementations of the frequently used functions
`utils.py`: scripts for partitioning the datasets
`models/*.py`: implementations of NPN and ResNet
`fednp.py`: implementations of FedNP
`fedavg.py`: implementations of FedAvg
`moon.py`: implementations of MOON
`fedprox.py`: implementations of FedProx
`scaffold.py`: implementations of SCAFFOLD
`data`: the folder of datasets


## Usage
The codes run on Python 3.7.10 and PyTorch 1.8.1. 

Quick start:
```
python fednp.py \
    --dataset cifar100 \
    --data_dist noniid \
    --K 10 \
    --local_epochs 10 \
    --epochs 100 \
```

TinyImageNet dataset can be downloaded by 
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

please store the dataset folder in the data/tinyimagenet folder like this:
```
+-- data
|   +-- tinyimagenet
|       +-- train
|       +-- val
|       +-- wnids.txt
|       +-- words.txt
|   +-- cifar10
|   +-- cifar100
```


To run the FedNP and federated learning baselines:
```bash
python [fednp|fedavg|moon|fedprox|scaffold].py \
    --dataset [cifar100|tinyimagenet] \
    --data_dist [noniid|random] \
    --K 10 \
    --local_epochs 10 \
    --epochs 100 \
```

To run the visualization for Figure 2 / 3:
```bash
python viz.py \
    --dataset [cifar100|tinyimagenet] \
    --data_dist [noniid|random] \
    --K 10 \
    --local_epochs 10 \
    --epochs 100 \
    --model_path <path_to_checkpoint>
```