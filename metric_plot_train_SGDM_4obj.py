import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker
import os


ACTION_TO_ID = {'push': 0, 'grasp': 1, 'place': 2}


def read_txt(path):
    file = open(path, "r")
    list_original = file.readlines()
    list_process = []
    for fields in list_original:
        fields = fields.strip()
        fields = fields.strip("\n")
        fields = int(float(fields))
        list_process.append(fields)

    return list_process


def calcul_action_efficiency(executed_actions_log, place_success_log, window=500):
    length = len(place_success_log)
    actions_efficiency = []
    push_count = executed_actions_log[:, 0] == ACTION_TO_ID['push']
    for i in range(1, length):
        start = max(i - window, 0)
        trial_push_count = push_count[start:i].sum()
        place_success = place_success_log[start:i].count(1)      # 1 -- represent the action of place success.
        ideal_actions_num = place_success * 2
        efficiency = ideal_actions_num / (i - start)

        actions_efficiency.append(efficiency)

    return actions_efficiency


def calcul_grasp_place_suceess_rate(executed_actions_log, grasp_success_log, place_success_log, window=500):
    grasp_success_rate = []
    place_success_rate = []
    length = max(len(grasp_success_log), len(place_success_log))
    grasp_count = executed_actions_log[:, 0] == ACTION_TO_ID['grasp']
    place_count = executed_actions_log[:, 0] == ACTION_TO_ID['place']
    for i in range(1, length):
        start = max(i - window, 0)
        trial_grasp_count = grasp_count[start:i].sum()
        trial_place_count = place_count[start:i].sum()
        trial_grasp_success = grasp_success_log[start:i].count(1)
        trial_place_success = place_success_log[start:i].count(1)
        if trial_grasp_count > 0:
            trial_grasp_success_rate = trial_grasp_success / trial_grasp_count
        else:
            trial_grasp_success_rate = 0
        if trial_place_count > 0:
            trial_place_success_rate = trial_place_success / trial_place_count
        else:
            trial_place_success_rate = 0

        grasp_success_rate.append(trial_grasp_success_rate)
        place_success_rate.append(trial_place_success_rate)

    return grasp_success_rate, place_success_rate


def calcul_sort_success_rate(progress_log, clearance_log, objects_num=4, window=50):
    sort_success_rate = []

    # get the number of objects has been sorted for each episode
    sort_success_num = []
    for i in clearance_log:
        trial_sort_success_num = progress_log[i-1]
        sort_success_num.append(trial_sort_success_num)

    length = len(sort_success_num)
    for i in range(1, length):
        start = max(i - window, 0)
        count_sort_success_num = 0
        for j in sort_success_num[start:i]:
            count_sort_success_num += j
        trial_sort_success_rate = count_sort_success_num / (objects_num * (i - start))

        sort_success_rate.append(trial_sort_success_rate)

    return sort_success_rate


def plot_efficiency(data, x_label, y_label, title, path):
    fig, ax = plt.subplots()
    ax.plot(data[0], color='g', linestyle='-', label='PQCN-DenseNet121-S-FCN-2048-4 objects(G=0.70)-SGDM')
    ax.plot(data[1], color='r', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.60)-SGDM')
    ax.plot(data[2], color='y', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.65)-SGDM')
    ax.plot(data[3], color='c', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.70)-SGDM')
    ax.plot(data[4], color='m', linestyle='-', label='PQCN-MobileNet-V3-S-FCN-1920-4 objects(G=0.70)-SGDM')
    ax.plot(data[5], color='gray', linestyle='-', label='PQCN-MobileNet-V3-FT-FCN-1920-4 objects(G=0.70)-SGDM')
    ax.plot(data[6], color='navy', linestyle='-', label='PQCN-MobileNet-V3-S-FCN-1920-4 objects(G=0.65)-SGDM')
    ax.plot(data[7], color='coral', linestyle='-', label='PQCN-MobileNet-V3-FT-FCN-1920-4 objects(G=0.65)-SGDM')
    ax.legend(loc=4, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xlim([0, 22000])
    ax.set_ylim([-0.70, 0.9])
    #ax.grid()
    plt.savefig(path)
    plt.show()


def plot_grasp_success_rate(grasp_data, x_label, y_label, title, path):
    fig, ax = plt.subplots()
    ax.plot(grasp_data[0], color='g', linestyle='-', label='PQCN-DenseNet121-S-FCN-2048-4 objects(G=0.70)-SGDM')
    ax.plot(grasp_data[1], color='r', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.60)-SGDM')
    ax.plot(grasp_data[2], color='y', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.65)-SGDM')
    ax.plot(grasp_data[3], color='c', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.70)-SGDM')
    ax.plot(grasp_data[4], color='m', linestyle='-', label='PQCN-MobileNet-V3-S-FCN-1920-4 objects(G=0.70)-SGDM')
    ax.plot(grasp_data[5], color='gray', linestyle='-', label='PQCN-MobileNet-V3-FT-FCN-1920-4 objects(G=0.70)-SGDM')
    ax.plot(grasp_data[6], color='navy', linestyle='-', label='PQCN-MobileNet-V3-S-FCN-1920-4 objects(G=0.65)-SGDM')
    ax.plot(grasp_data[7], color='coral', linestyle='-', label='PQCN-MobileNet-V3-FT-FCN-1920-4 objects(G=0.65)-SGDM')
    ax.legend(loc=4, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=10)
    #ax.grid()
    ax.set_xlim([0, 22000])
    ax.set_ylim([-0.1, 0.95])
    plt.savefig(path)
    plt.show()


def plot_place_success_rate(place_data, x_label, y_label, title, path):
    fig, ax = plt.subplots()
    ax.plot(place_data[0], color='g', linestyle='-', label='PQCN-DenseNet121-S-FCN-2048-4 objects(G=0.70)-SGDM')
    ax.plot(place_data[1], color='r', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.60)-SGDM')
    ax.plot(place_data[2], color='y', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.65)-SGDM')
    ax.plot(place_data[3], color='c', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.70)-SGDM')
    ax.plot(place_data[4], color='m', linestyle='-', label='PQCN-MobileNet-V3-S-FCN-1920-4 objects(G=0.70)-SGDM')
    ax.plot(place_data[5], color='gray', linestyle='-', label='PQCN-MobileNet-V3-FT-FCN-1920-4 objects(G=0.70)-SGDM')
    ax.plot(place_data[6], color='navy', linestyle='-', label='PQCN-MobileNet-V3-S-FCN-1920-4 objects(G=0.65)-SGDM')
    ax.plot(place_data[7], color='coral', linestyle='-', label='PQCN-MobileNet-V3-FT-FCN-1920-4 objects(G=0.65)-SGDM')
    ax.legend(loc=4, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=10)
    #ax.grid()
    ax.set_xlim([0, 22000])
    ax.set_ylim([-0.65, 1])
    plt.savefig(path)
    plt.show()


def plot_sort_rate(data, x_label, y_label, title, path):
    fig, ax = plt.subplots()
    ax.plot(data[0], color='g', linestyle='-', label='PQCN-DenseNet121-S-FCN-2048-4 objects(G=0.70)-SGDM')
    ax.plot(data[1], color='r', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.60)-SGDM')
    ax.plot(data[2], color='y', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.65)-SGDM')
    ax.plot(data[3], color='c', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-4 objects(G=0.70)-SGDM')
    ax.plot(data[4], color='m', linestyle='-', label='PQCN-MobileNet-V3-S-FCN-1920-4 objects(G=0.70)-SGDM')
    ax.plot(data[5], color='gray', linestyle='-', label='PQCN-MobileNet-V3-FT-FCN-1920-4 objects(G=0.70)-SGDM')
    ax.plot(data[6], color='navy', linestyle='-', label='PQCN-MobileNet-V3-S-FCN-1920-4 objects(G=0.65)-SGDM')
    ax.plot(data[7], color='coral', linestyle='-', label='PQCN-MobileNet-V3-FT-FCN-1920-4 objects(G=0.65)-SGDM')
    

 
    ax.legend(loc=4, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xlim([0, 2000])
    ax.set_ylim([-0.4, 1.0])
    #ax.grid()
    plt.savefig(path)
    plt.show()


def calculate_mean(list1):
    sum = 0
    for i in list1:
        sum += i
    average = sum / len(list1)

    return average

def main(args):
    log_dir = args.log_dir
    four_random_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-S-FCN-G-070-SGDM/place-success.log.txt'))
    four_random_custom_FT_DQN_060_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-060-SGDM/place-success.log.txt'))
    four_random_custom_FT_DQN_065_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-065-SGDM/place-success.log.txt'))
    four_random_custom_FT_DQN_070_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-070-SGDM/place-success.log.txt'))
    four_random_custom_MOBILENETV3_DQN_070_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-G-070-SGDM/place-success.log.txt'))
    four_random_custom_FT_MOBILENETV3_DQN_070_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-G-070-SGDM/place-success.log.txt'))
    four_random_MNET_065_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-G-065-SGDM/place-success.log.txt'))
    four_random_FT_MNET_065_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-G-065-SGDM/place-success.log.txt'))
   
    
    
    four_random_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-S-FCN-G-070-SGDM/grasp-success.log.txt'))
    four_random_custom_FT_DQN_060_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-060-SGDM/grasp-success.log.txt'))
    four_random_custom_FT_DQN_065_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-065-SGDM/grasp-success.log.txt'))
    four_random_custom_FT_DQN_070_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-070-SGDM/grasp-success.log.txt'))
    four_random_custom_MOBILENETV3_DQN_070_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-G-070-SGDM/grasp-success.log.txt'))
    four_random_custom_FT_MOBILENETV3_DQN_070_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-G-070-SGDM/grasp-success.log.txt'))
    four_random_MNET_065_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-G-065-SGDM/grasp-success.log.txt'))
    four_random_FT_MNET_065_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-G-065-SGDM/grasp-success.log.txt'))
   
    

    four_random_progress_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-S-FCN-G-070-SGDM/progress.log.txt'))
    four_random_custom_FT_DQN_060_progress_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-060-SGDM/progress.log.txt'))
    four_random_custom_FT_DQN_065_progress_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-065-SGDM/progress.log.txt'))
    four_random_custom_FT_DQN_070_progress_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-070-SGDM/progress.log.txt'))
    four_random_custom_MOBILENETV3_DQN_070_progress_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-G-070-SGDM/progress.log.txt'))
    four_random_custom_FT_MOBILENETV3_DQN_070_progress_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-G-070-SGDM/progress.log.txt'))
    four_random_MNET_065_progress_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-G-065-SGDM/progress.log.txt'))
    four_random_FT_MNET_065_progress_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-G-065-SGDM/progress.log.txt'))
   

    four_random_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-S-FCN-G-070-SGDM/clearance.log.txt'))
    four_random_custom_FT_DQN_060_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-060-SGDM/clearance.log.txt'))
    four_random_custom_FT_DQN_065_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-065-SGDM/clearance.log.txt'))
    four_random_custom_FT_DQN_070_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-070-SGDM/clearance.log.txt'))
    four_random_custom_MOBILENETV3_DQN_070_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-G-070-SGDM/clearance.log.txt'))
    four_random_custom_FT_MOBILENETV3_DQN_070_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-G-070-SGDM/clearance.log.txt'))
    four_random_MNET_065_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-G-065-SGDM/clearance.log.txt'))
    four_random_FT_MNET_065_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-G-065-SGDM/clearance.log.txt'))
    
    
    
 
    four_random_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-DenseNet121-S-FCN-G-070-SGDM/executed-action.log.txt'))
    four_random_custom_FT_DQN_060_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-060-SGDM/executed-action.log.txt'))
    four_random_custom_FT_DQN_065_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-065-SGDM/executed-action.log.txt'))
    four_random_custom_FT_DQN_070_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-G-070-SGDM/executed-action.log.txt'))
    four_random_custom_MOBILENETV3_DQN_070_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-G-070-SGDM/executed-action.log.txt'))
    four_random_custom_FT_MOBILENETV3_DQN_070_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-G-070-SGDM/executed-action.log.txt'))
    four_random_MNET_065_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-G-065-SGDM/executed-action.log.txt'))
    four_random_FT_MNET_065_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-G-065-SGDM/executed-action.log.txt'))
   
    
   

    # calculate the action efficiency
    actions_efficiency = []
    four_random_actions_efficiency = calcul_action_efficiency(four_random_executed_actions_log, four_random_place_success_log, window=500)
    actions_efficiency.append(four_random_actions_efficiency)

    four_random_custom_FT_DQN_060_actions_efficiency = calcul_action_efficiency(four_random_custom_FT_DQN_060_executed_actions_log, four_random_custom_FT_DQN_060_place_success_log, window=500)
    actions_efficiency.append(four_random_custom_FT_DQN_060_actions_efficiency)

    four_random_custom_FT_DQN_065_actions_efficiency = calcul_action_efficiency(four_random_custom_FT_DQN_065_executed_actions_log, four_random_custom_FT_DQN_065_place_success_log, window=500)
    actions_efficiency.append(four_random_custom_FT_DQN_065_actions_efficiency)

    four_random_custom_FT_DQN_070_actions_efficiency = calcul_action_efficiency(four_random_custom_FT_DQN_070_executed_actions_log, four_random_custom_FT_DQN_070_place_success_log, window=500)
    actions_efficiency.append(four_random_custom_FT_DQN_070_actions_efficiency)

    four_random_custom_MOBILENETV3_DQN_070_actions_efficiency = calcul_action_efficiency(four_random_custom_MOBILENETV3_DQN_070_executed_actions_log,four_random_custom_MOBILENETV3_DQN_070_place_success_log, window=500)
    actions_efficiency.append(four_random_custom_MOBILENETV3_DQN_070_actions_efficiency)

    four_random_custom_FT_MOBILENETV3_DQN_070_actions_efficiency = calcul_action_efficiency(four_random_custom_FT_MOBILENETV3_DQN_070_executed_actions_log,four_random_custom_FT_MOBILENETV3_DQN_070_place_success_log, window=500)
    actions_efficiency.append(four_random_custom_FT_MOBILENETV3_DQN_070_actions_efficiency)
    
    four_random_MNET_065_actions_efficiency = calcul_action_efficiency(four_random_MNET_065_executed_actions_log, four_random_MNET_065_place_success_log, window=500)
    actions_efficiency.append(four_random_MNET_065_actions_efficiency)
    
    four_random_FT_MNET_065_actions_efficiency = calcul_action_efficiency(four_random_FT_MNET_065_executed_actions_log, four_random_FT_MNET_065_place_success_log, window=500)
    actions_efficiency.append(four_random_FT_MNET_065_actions_efficiency)
    
    
   

    x_label = 'Number of actions'
    y_label = 'Efficiency'
    title = 'Action Efficiency'
    save_path_actions_efficiency = 'plot/actions_efficiency4c.eps'
    plot_efficiency(actions_efficiency, x_label, y_label, title, save_path_actions_efficiency)

    # calculate the success rate of grasp & place
    grasp_success_rate = []
    place_success_rate = []
    
    four_random_grasp_success_rate, four_random_place_success_rate = calcul_grasp_place_suceess_rate(four_random_executed_actions_log, four_random_grasp_success_log, four_random_place_success_log, window=500)
    grasp_success_rate.append(four_random_grasp_success_rate)
    place_success_rate.append(four_random_place_success_rate)

    four_random_custom_FT_DQN_060_grasp_success_rate, four_random_custom_FT_DQN_060_place_success_rate = calcul_grasp_place_suceess_rate(four_random_custom_FT_DQN_060_executed_actions_log, four_random_custom_FT_DQN_060_grasp_success_log, four_random_custom_FT_DQN_060_place_success_log, window=500)
    grasp_success_rate.append(four_random_custom_FT_DQN_060_grasp_success_rate)
    place_success_rate.append(four_random_custom_FT_DQN_060_place_success_rate)

    four_random_custom_FT_DQN_065_grasp_success_rate, four_random_custom_FT_DQN_065_place_success_rate = calcul_grasp_place_suceess_rate(four_random_custom_FT_DQN_065_executed_actions_log, four_random_custom_FT_DQN_065_grasp_success_log, four_random_custom_FT_DQN_065_place_success_log, window=500)
    grasp_success_rate.append(four_random_custom_FT_DQN_065_grasp_success_rate)
    place_success_rate.append(four_random_custom_FT_DQN_065_place_success_rate)

    four_random_custom_FT_DQN_070_grasp_success_rate, four_random_custom_FT_DQN_070_place_success_rate = calcul_grasp_place_suceess_rate(four_random_custom_FT_DQN_070_executed_actions_log, four_random_custom_FT_DQN_070_grasp_success_log, four_random_custom_FT_DQN_070_place_success_log, window=500)
    grasp_success_rate.append(four_random_custom_FT_DQN_070_grasp_success_rate)
    place_success_rate.append(four_random_custom_FT_DQN_070_place_success_rate)

    four_random_custom_MOBILENETV3_DQN_070_grasp_success_rate, four_random_custom_MOBILENETV3_DQN_070_place_success_rate = calcul_grasp_place_suceess_rate(four_random_custom_MOBILENETV3_DQN_070_executed_actions_log, four_random_custom_MOBILENETV3_DQN_070_grasp_success_log, four_random_custom_MOBILENETV3_DQN_070_place_success_log, window=500)
    grasp_success_rate.append(four_random_custom_MOBILENETV3_DQN_070_grasp_success_rate)
    place_success_rate.append(four_random_custom_MOBILENETV3_DQN_070_place_success_rate)
    
    four_random_custom_FT_MOBILENETV3_DQN_070_grasp_success_rate, four_random_custom_FT_MOBILENETV3_DQN_070_place_success_rate = calcul_grasp_place_suceess_rate(four_random_custom_FT_MOBILENETV3_DQN_070_executed_actions_log, four_random_custom_FT_MOBILENETV3_DQN_070_grasp_success_log, four_random_custom_FT_MOBILENETV3_DQN_070_place_success_log, window=500)
    grasp_success_rate.append(four_random_custom_FT_MOBILENETV3_DQN_070_grasp_success_rate)
    place_success_rate.append(four_random_custom_FT_MOBILENETV3_DQN_070_place_success_rate)


    four_random_MNET_065_grasp_success_rate, four_random_MNET_065_place_success_rate =calcul_grasp_place_suceess_rate(four_random_MNET_065_executed_actions_log,four_random_MNET_065_grasp_success_log, four_random_MNET_065_place_success_log, window=500)
    grasp_success_rate.append(four_random_MNET_065_grasp_success_rate)
    place_success_rate.append(four_random_MNET_065_place_success_rate)
    
    four_random_FT_MNET_065_grasp_success_rate, four_random_FT_MNET_065_place_success_rate =calcul_grasp_place_suceess_rate(four_random_FT_MNET_065_executed_actions_log,four_random_FT_MNET_065_grasp_success_log, four_random_FT_MNET_065_place_success_log, window=500)
    grasp_success_rate.append(four_random_FT_MNET_065_grasp_success_rate)
    place_success_rate.append(four_random_FT_MNET_065_place_success_rate)
    

    x_label = 'Number of actions'
    y_label = 'Success rate'
    grasp_title = 'Success Rate of Grasping'
    place_title = 'Success Rate of Placing'
    save_path_grasp_success_rate = 'plot/grasp_success_rate4c.eps'
    save_path_place_success_rate = 'plot/place_success_rate4c.eps'
    plot_grasp_success_rate(grasp_success_rate, x_label, y_label, grasp_title, save_path_grasp_success_rate)
    plot_place_success_rate(place_success_rate, x_label, y_label, place_title, save_path_place_success_rate)

    # calculate the success rate of sorting
    sort_success_rate = []

    four_random_sort_success_rate = calcul_sort_success_rate(four_random_progress_log, four_random_clearance_log, objects_num=4, window=50)
    sort_success_rate.append(four_random_sort_success_rate)

    four_random_custom_FT_DQN_060_sort_success_rate = calcul_sort_success_rate(four_random_custom_FT_DQN_060_progress_log, four_random_custom_FT_DQN_060_clearance_log, objects_num=4, window=50)
    sort_success_rate.append(four_random_custom_FT_DQN_060_sort_success_rate)

    four_random_custom_FT_DQN_065_sort_success_rate = calcul_sort_success_rate(four_random_custom_FT_DQN_065_progress_log, four_random_custom_FT_DQN_065_clearance_log, objects_num=4, window=50)
    sort_success_rate.append(four_random_custom_FT_DQN_065_sort_success_rate)

    four_random_custom_FT_DQN_070_sort_success_rate = calcul_sort_success_rate(four_random_custom_FT_DQN_070_progress_log, four_random_custom_FT_DQN_070_clearance_log, objects_num=4, window=50)
    sort_success_rate.append(four_random_custom_FT_DQN_070_sort_success_rate)

    four_random_custom_MOBILENETV3_DQN_070_sort_success_rate = calcul_sort_success_rate(four_random_custom_MOBILENETV3_DQN_070_progress_log, four_random_custom_MOBILENETV3_DQN_070_clearance_log, objects_num=4, window=50)
    sort_success_rate.append(four_random_custom_MOBILENETV3_DQN_070_sort_success_rate)
    
    four_random_custom_FT_MOBILENETV3_DQN_070_sort_success_rate = calcul_sort_success_rate(four_random_custom_FT_MOBILENETV3_DQN_070_progress_log, four_random_custom_FT_MOBILENETV3_DQN_070_clearance_log, objects_num=4, window=50)
    sort_success_rate.append(four_random_custom_FT_MOBILENETV3_DQN_070_sort_success_rate)
    
    
    four_random_MNET_065_sort_success_rate = calcul_sort_success_rate(four_random_MNET_065_progress_log, four_random_MNET_065_clearance_log, objects_num=4, window=50)
    sort_success_rate.append(four_random_MNET_065_sort_success_rate)

    four_random_FT_MNET_065_sort_success_rate = calcul_sort_success_rate(four_random_FT_MNET_065_progress_log, four_random_FT_MNET_065_clearance_log, objects_num=4, window=50)
    sort_success_rate.append(four_random_FT_MNET_065_sort_success_rate)


    x_label = 'Number of episodes'
    y_label = 'Completion rate'
    title = 'Task Completion Rate'
    save_path_sort_success_rate = 'plot/sort_success_rate4c.eps'
    plot_sort_rate(sort_success_rate, x_label, y_label, title, save_path_sort_success_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing after training or testing.')

    parser.add_argument('--log_dir', dest='log_dir', action='store', default=None,
                           help='directory containing logger should be to process.')

    args = parser.parse_args()
    main(args)
