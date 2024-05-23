# ULRL: Unified Neural Backdoor Removal with Only Few Clean Samples through Unlearning and Relearning

This repository is the official implementation of ULRL: Unified Neural Backdoor Removal with Only Few Clean Samples through Unlearning and Relearning. 

### Abstract

The application of deep neural network models in various security-critical applications has raised significant security concerns, particularly the risk of backdoor attacks. Neural backdoors pose a serious security threat as they allow attackers to maliciously alter model behavior. While many defenses have been explored, existing approaches are often bounded by model-specific constraints, or necessitate complex alterations to the training process, or fall short against diverse backdoor attacks. In this work, we introduce a novel method for comprehensive and effective elimination of backdoors, called ULRL (short for <u>U</u>n<u>L</u>earn and <u>R</u>e<u>L</u>earn for backdoor removal). ULRL requires only a small set of clean samples and works effectively against all kinds of backdoors. It first applies unlearning for identifying suspicious neurons and then targeted neural weight tuning for backdoor mitigation (i.e., by promoting significant weight deviation on the suspicious neurons). Evaluated against 12 different types of backdoors, ULRL is shown to significantly outperform state-of-the-art methods in eliminating backdoors whilst preserving the model utility. 

## Installation

You can run the following script to configurate necessary environment

```
cd ULRL
conda create -n ULRL python=3.8
conda activate ULRL
sh ./sh/install.sh
sh ./sh/init_folders.sh
```

## Quick Start

### Defense

This is a script of running ULRL defense on cifar-10 for badnet attack. Before defense you need to place badnet attack on cifar-10 under the 'record' folder at first. Then you use the folder name as result_file.

```
python ./defense/ulrl.py --result_file badnet_0_1 --yaml_path ./config/defense/ulrl/cifar10.yaml --dataset cifar10
```


If you want to change the args, you can both specify them in command line and in corresponding YAML config file (eg. [config.yaml](./config/defense/ulrl/config.yaml)).(They are the defaults we used if no args are specified in command line.)

----
#### Our codes heavily depend on [BackdoorBench](https://github.com/SCLBD/BackdoorBench), *"BackdoorBench: A Comprehensive Benchmark of Backdoor Learning"*. You can download the pre-trained backdoored models from BackdoorBench. 