# DRL_for_Object_Sorting
Deep Reinforcement Learning with Light-Weight Vision Model for Sequential Robotic Object Sorting

# Video Demo
An illustration of the training process of our newly proposed deep reinforcement learning agent (PQCN_MobileNetV3_S_FCN) used for sorting object blocks under varying degrees of complexity. 

<!-- ![Method Overview](method.png?raw=true) -->
<div align="center"><img alt="Animated GIF" src="img/deep_reinforcement_learning_for_object_sorting.gif" width="95%"/></div> 


# Method Overview
This GitHub repository presents an implementation of Pytorch code of deep reinforcement learning (DRL) agents based on Pixelwise Q-valued based Critic Network (PQCN) developed from a modified variant of the Vanilla DQN. The central goal of this project is to perform a comparative analysis on several DRL agents in executing object sorting of several categories of irregular and regular object blocks under a cluttered, occluded, and highly dynamic environment scenario with varying degrees of complexities. In our project, we propose a new variant of a PQCN that factors any of the four kinds of network backbones (MobileNetV3, DenseNet121, DenseNet169, and SqueezeNet1.0) integrated with a custom fully convolutional neural network used for deciding three 
possible optimal actions (pushing, grasping, and placing).
We demonstrate the training and testing of the DRL object sorting policies in simulation environment condition that factors  the utility of a UR5 robot-arm. 

# PQCN-MobileNetV3-S-FCN
<!-- ![Method Overview](method.png?raw=true) -->
<div align="center"><img src="img/MobileNetV3_L.png" width="95%"/></div>

# PQCN-DenseNet121-S-FCN
<!-- ![Method Overview](method.png?raw=true) -->
<div align="center"><img src="img/densen.png" width="95%"/></div>

# PQCN-SqueezeNet-S-FCN
<!-- ![Method Overview](method.png?raw=true) -->
<div align="center"><img src="img/sqn.png" width="95%"/></div>

#### Contact
Please let me know if there are any questions or bugs: [Emmanuel Okafor] emmanuel.okafor@kfupm.edu.sa


#### Instructions

## Installation
We utilized the PyTorch ('2.1.0+cu121')  deep learning framework for implementing and training all our  deep reinforcement learning models on a PC with two NVIDIA GeForce RTX 3090 each having 24GB GPUs  (with CUDA 12.1) and AMD Ryzen Threadripper PRO 3975WX 32-Cores CPU memory capacity running on Ubuntu 22.04.

1. Install [Anaconda](https://www.anaconda.com/) and create virtual environment
```shell
conda create -n object_sorting python=3.9.16 -y
```
2. Install [PyTorch](https://pytorch.org/)
```shell
conda activate object_sorting
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
3. Install python libraries
```shell
pip3 install numpy scipy opencv-python matplotlib
```
4. Install [CoppeliaSim](http://www.coppeliarobotics.com/)) simulation environment

## How to run
#### Prepare simulation environment
Run CoppeliaSim(navigate to your CoppeliaSim directory and run `./coppeliaSim.sh`). From the main menu, select `File` > `Open scene...`, and open the file `DRLSorting/simulation/simulation.ttt` from this repository.

#### Training
```shell
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 10 --push_rewards --experience_replay --random_actions --use_commonsense --explore_rate_decay --future_reward_discount 0.70 --max_iter 50000 --save_visualization
```

#### How to continue training
```shell
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 10 --push_rewards --experience_replay --random_actions --use_commonsense --explore_rate_decay --future_reward_discount 0.70 --max_iter 50000 --save_visualization --load_snapshot --snapshot_file './logs/[USER_FOLDER]/models/snapshot-backup.reinforcement.pth' --continue_logging --logging_directory './logs/USER_FOLDER'
```

#### How to plot the shorter duration experiments for performing 4 object sorting considering three forms of optimization schemes after training for 10000 action steps

```shell
python metric_plot_train_optimization_4obj.py --log_dir './Training_Performance_Results_Logs/Results_4ObjectSorting/plot'
```
#### How to plot the longer duration experiments for executing 4 object sorting considering only SGDM after training for 22000 action steps.

```shell
python metric_plot_train_SGDM_4obj.py --log_dir './Training_Performance_Results_Logs/Results_4ObjectSorting/plot'
```
#### How to plot the experiments of the variants of PQCN during execution of 6 object sorting considering only SGDM after training for 22000 action steps for dual transfer learning and  40000 action steps for single transfer learning.

```shell
python metric_plot_train_6_object_sorting.py --log_dir './Training_Performance_Results_Logs/Results_6_ObjectSorting/plot'
```
#### How to plot the experiments of the variants of PQCN during execution of 10 object sorting considering only SGDM after training for 22000 action steps for dual transfer learning and  50000 action steps for single transfer learning.

```shell
python metric_plot_train_10_object_sorting.py --log_dir './Training_Performance_Results_Logs/Results_10_ObjectSorting/plot/'
```


#### How to evaluate the training  performance
```shell
python metric_eval_train_performance_evaluation.py --log_dir './Training_Performance_Results_Logs/Results_4ObjectSorting/plot/PQCN-DenseNet121-FT-FCN-G-070-SGDM' --object_num  4
```



#### How to evaluate the testing performance
```shell
python3 metric_eval_test.py --log_dir './Testing_Performance_Results_Logs/Testing_fixed_scene_for_4_object_sorting_transition_log/PQCN-DenseNet121-FT-FCN-4objs/transitions1' --object_num 4
```


## Acknowledgement
Our code is based on [VPG](https://github.com/andyzeng/visual-pushing-grasping), [Good Robot!](https://github.com/jhu-lcsr/good_robot), and [DRLSorting](https://github.com/JiatongBao/DRLSorting).


