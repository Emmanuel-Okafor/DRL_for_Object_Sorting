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
    
    ax.plot(data[0], color='g', linestyle='-', label='PQCN-DenseNet121-S-FCN-2048-10 objects(G=0.70)-SGDM')
    ax.plot(data[1], color='c', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-10 objects(G=0.70)-SGDM')
    ax.plot(data[2], color='m', linestyle='-', label='PQCN-MobileNetV3-S-FCN-1920-10 objects(G=0.70)-SGDM')
    ax.plot(data[3], color='gray', linestyle='-', label='PQCN-MobileNetV3-FT-FCN-1920-10 objects(G=0.70)-SGDM')
    ax.legend(loc=4, fontsize=10)
   
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_ylim([-0.25, 0.85])
    #plt.grid()
    plt.savefig(path)
    plt.show()


def plot_grasp_success_rate(grasp_data, x_label, y_label, title, path):
    fig, ax = plt.subplots()
    ax.plot(grasp_data[0], color='g', linestyle='-', label='PQCN-DenseNet121-S-FCN-2048-10 objects(G=0.70)-SGDM')
    ax.plot(grasp_data[1], color='c', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-10 objects(G=0.70)-SGDM')
    ax.plot(grasp_data[2], color='m', linestyle='-', label='PQCN-MobileNetV3-S-FCN-1920-10 objects(G=0.70)-SGDM')
    ax.plot(grasp_data[3], color='gray', linestyle='-', label='PQCN-MobileNetV3-FT-FCN-1920-10 objects(G=0.70)-SGDM')
    ax.legend(loc=4, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=10)
   
   # plt.grid()
    plt.savefig(path)
    plt.show()


def plot_place_success_rate(place_data, x_label, y_label, title, path):
    fig, ax = plt.subplots()
    ax.plot(place_data[0], color='g', linestyle='-', label='PQCN-DenseNet121-S-FCN-2048-10 objects(G=0.70)-SGDM')
    ax.plot(place_data[1], color='c', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-10 objects(G=0.70)-SGDM')
    ax.plot(place_data[2], color='m', linestyle='-', label='PQCN-MobileNetV3-S-FCN-1920-10 objects(G=0.70)-SGDM')
    ax.plot(place_data[3], color='gray', linestyle='-', label='PQCN-MobileNetV3-FT-FCN-1920-10 objects(G=0.70)-SGDM')
    ax.legend(loc=4, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_ylim([-0.25, 0.9])
    #plt.grid()
    plt.savefig(path)
    plt.show()


def plot_sort_rate(data, x_label, y_label, title, path):
    fig, ax = plt.subplots()
    ax.plot(data[0], color='g', linestyle='-', label='PQCN-DenseNet121-S-FCN-2048-10 objects(G=0.70)-SGDM')
    ax.plot(data[1], color='c', linestyle='-', label='PQCN-DenseNet121-FT-FCN-2048-10 objects(G=0.70)-SGDM')
    ax.plot(data[2], color='m', linestyle='-', label='PQCN-MobileNetV3-S-FCN-1920-10 objects(G=0.70)-SGDM')
    ax.plot(data[3], color='gray', linestyle='-', label='PQCN-MobileNetV3-FT-FCN-1920-10 objects(G=0.70)-SGDM')
    ax.legend(loc=4, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_ylim([-0.25, 0.90])
    #plt.grid()
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

    
    ten_random_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-S-FCN-10Objs/place-success.log.txt'))
    ten_random_FT_FCN_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-10Objs/place-success.log.txt'))
    ten_random_MobileNetV3_S_FCN_G070_FCN_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-10Objs/place-success.log.txt'))
    ten_random_MobileNetV3_FT_FCN_G070_FCN_place_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-10Objs/place-success.log.txt'))
   
    

   
    ten_random_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-S-FCN-10Objs/grasp-success.log.txt'))
    ten_random_FT_FCN_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-10Objs/grasp-success.log.txt'))
    ten_random_MobileNetV3_S_FCN_G070_FCN_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-10Objs/grasp-success.log.txt'))
    ten_random_MobileNetV3_FT_FCN_G070_FCN_grasp_success_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-10Objs/grasp-success.log.txt'))
    
    
    ten_random_progress_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-S-FCN-10Objs/progress.log.txt'))
    ten_random_FT_FCN_progress_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-10Objs/progress.log.txt'))
    ten_random_MobileNetV3_S_FCN_G070_FCN_progress_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-10Objs/progress.log.txt'))
    ten_random_MobileNetV3_FT_FCN_G070_FCN_progress_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-10Objs/progress.log.txt'))

     
    ten_random_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-S-FCN-10Objs/clearance.log.txt'))
    ten_random_FT_FCN_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-10Objs/clearance.log.txt'))
    ten_random_MobileNetV3_S_FCN_G070_FCN_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-10Objs/clearance.log.txt'))
    ten_random_MobileNetV3_FT_FCN_G070_FCN_clearance_log = read_txt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-10Objs/clearance.log.txt'))

   
    ten_random_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-DenseNet121-S-FCN-10Objs/executed-action.log.txt'))
    ten_random_FT_FCN_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-DenseNet121-FT-FCN-10Objs/executed-action.log.txt'))
    ten_random_MobileNetV3_S_FCN_G070_FCN_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-MobileNetV3-S-FCN-10Objs/executed-action.log.txt'))
    ten_random_MobileNetV3_FT_FCN_G070_FCN_executed_actions_log = np.loadtxt(os.path.join(log_dir, 'PQCN-MobileNetV3-FT-FCN-10Objs/executed-action.log.txt'))


    # calculate the action efficiency
    actions_efficiency = []
    ten_random_actions_efficiency = calcul_action_efficiency(ten_random_executed_actions_log, ten_random_place_success_log, window=500)
    actions_efficiency.append(ten_random_actions_efficiency)
    
    ten_random_FT_FCN_actions_efficiency = calcul_action_efficiency(ten_random_FT_FCN_executed_actions_log, ten_random_FT_FCN_place_success_log, window=500)
    actions_efficiency.append(ten_random_FT_FCN_actions_efficiency)
    
    ten_random_MobileNetV3_S_FCN_G070_FCN_actions_efficiency = calcul_action_efficiency(ten_random_MobileNetV3_S_FCN_G070_FCN_executed_actions_log, ten_random_MobileNetV3_S_FCN_G070_FCN_place_success_log, window=500)
    actions_efficiency.append(ten_random_MobileNetV3_S_FCN_G070_FCN_actions_efficiency)
    
    ten_random_MobileNetV3_FT_FCN_G070_FCN_actions_efficiency = calcul_action_efficiency(ten_random_MobileNetV3_FT_FCN_G070_FCN_executed_actions_log, ten_random_MobileNetV3_FT_FCN_G070_FCN_place_success_log, window=500)
    actions_efficiency.append(ten_random_MobileNetV3_FT_FCN_G070_FCN_actions_efficiency)
    
    
    print(max(actions_efficiency))

    x_label = 'Number of actions'
    y_label = 'Efficiency'
    title = 'Action Efficiency'
    save_path_actions_efficiency = 'plot/actions_efficiency10_.eps'
    plot_efficiency(actions_efficiency, x_label, y_label, title, save_path_actions_efficiency)

    # calculate the success rate of grasp & place
    grasp_success_rate = []
    place_success_rate = []
    
    ten_random_grasp_success_rate, ten_random_place_success_rate = calcul_grasp_place_suceess_rate(ten_random_executed_actions_log, ten_random_grasp_success_log, ten_random_place_success_log, window=500)
    grasp_success_rate.append(ten_random_grasp_success_rate)
    place_success_rate.append(ten_random_place_success_rate)
    
    ten_random_FT_FCN_grasp_success_rate, ten_random_FT_FCN_place_success_rate = calcul_grasp_place_suceess_rate(ten_random_FT_FCN_executed_actions_log, ten_random_FT_FCN_grasp_success_log, ten_random_FT_FCN_place_success_log, window=500)
    grasp_success_rate.append(ten_random_FT_FCN_grasp_success_rate)
    place_success_rate.append(ten_random_FT_FCN_place_success_rate)
    
    ten_random_MobileNetV3_S_FCN_G070_FCN_grasp_success_rate, ten_random_MobileNetV3_S_FCN_G070_FCN_place_success_rate = calcul_grasp_place_suceess_rate(ten_random_MobileNetV3_S_FCN_G070_FCN_executed_actions_log, ten_random_MobileNetV3_S_FCN_G070_FCN_grasp_success_log, ten_random_MobileNetV3_S_FCN_G070_FCN_place_success_log, window=500)
    grasp_success_rate.append(ten_random_MobileNetV3_S_FCN_G070_FCN_grasp_success_rate)
    place_success_rate.append(ten_random_MobileNetV3_S_FCN_G070_FCN_place_success_rate)
    
    ten_random_MobileNetV3_FT_FCN_G070_FCN_grasp_success_rate, ten_random_MobileNetV3_FT_FCN_G070_FCN_place_success_rate = calcul_grasp_place_suceess_rate(ten_random_MobileNetV3_FT_FCN_G070_FCN_executed_actions_log, ten_random_MobileNetV3_FT_FCN_G070_FCN_grasp_success_log, ten_random_MobileNetV3_FT_FCN_G070_FCN_place_success_log, window=500)
    grasp_success_rate.append(ten_random_MobileNetV3_FT_FCN_G070_FCN_grasp_success_rate)
    place_success_rate.append(ten_random_MobileNetV3_FT_FCN_G070_FCN_place_success_rate)


    x_label = 'Number of actions'
    y_label = 'Success rate'
    grasp_title = 'Success Rate of Grasping'
    place_title = 'Success Rate of Placing'
    save_path_grasp_success_rate = 'plot/grasp_success_rate.eps'
    save_path_place_success_rate = 'plot/place_success_rate.eps'
    plot_grasp_success_rate(grasp_success_rate, x_label, y_label, grasp_title, save_path_grasp_success_rate)
    plot_place_success_rate(place_success_rate, x_label, y_label, place_title, save_path_place_success_rate)

    # calculate the success rate of sorting
    sort_success_rate = []
    ten_random_sort_success_rate = calcul_sort_success_rate(ten_random_progress_log, ten_random_clearance_log, objects_num=10, window=50)
    sort_success_rate.append(ten_random_sort_success_rate)
    
    ten_random_FT_FCN_sort_success_rate = calcul_sort_success_rate(ten_random_FT_FCN_progress_log, ten_random_FT_FCN_clearance_log, objects_num=10, window=50)
    sort_success_rate.append(ten_random_FT_FCN_sort_success_rate)
    
    ten_random_MobileNetV3_S_FCN_G070_FCN_sort_success_rate = calcul_sort_success_rate(ten_random_MobileNetV3_S_FCN_G070_FCN_progress_log, ten_random_MobileNetV3_S_FCN_G070_FCN_clearance_log, objects_num=10, window=50)
    sort_success_rate.append(ten_random_MobileNetV3_S_FCN_G070_FCN_sort_success_rate)
    
    ten_random_MobileNetV3_FT_FCN_G070_FCN_sort_success_rate = calcul_sort_success_rate(ten_random_MobileNetV3_FT_FCN_G070_FCN_progress_log, ten_random_MobileNetV3_FT_FCN_G070_FCN_clearance_log, objects_num=10, window=50)
    sort_success_rate.append(ten_random_MobileNetV3_FT_FCN_G070_FCN_sort_success_rate)
    
    
   

    x_label = 'Number of episodes'
    y_label = 'Completion rate'
    title = 'Task Completion Rate'
    save_path_sort_success_rate = 'plot/sort_success_rate.eps'
    plot_sort_rate(sort_success_rate, x_label, y_label, title, save_path_sort_success_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing after training or testing.')

    parser.add_argument('--log_dir', dest='log_dir', action='store', default=None,
                           help='directory containing logger should be to process.')

    args = parser.parse_args()
    main(args)
