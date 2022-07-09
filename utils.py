import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch.optim as optim


def optimizer_selection(config, train_model, sx, sq):
    optimizer = None
    if config["learn_beta"]:
        if config["optimizer"] == "adam":
            optimizer = optim.AdamW(params=[{'params': train_model.parameters()},
                                            {'params': [sx, sq]}],
                                    lr=config["l_rate"],
                                    weight_decay=config["weight_decay"])
        elif config["optimizer"] == "sgd":
            optimizer = optim.SGD([{'params': train_model.parameters()},
                                   {'params': [sx, sq]}], lr=config["l_rate"],
                                  momentum=config["momentum"],
                                  weight_decay=config["weight_decay"])
        elif config["optimizer"] == "rms":
            optimizer = optim.RMSprop([{'params': train_model.parameters()},
                                       {'params': [sx, sq]}], lr=config["l_rate"],
                                      momentum=config["momentum"],
                                      weight_decay=config["weight_decay"])
        elif config["optimizer"] == "adagrad":
            optimizer = optim.Adagrad([{'params': train_model.parameters()},
                                       {'params': [sx, sq]}], lr=config["l_rate"],
                                      weight_decay=config["weight_decay"])
        else:
            print("ERROR: Optimizer chosen is not valid")
    else:
        if config["optimizer"] == "adam":
            optimizer = optim.AdamW(params=train_model.parameters(), lr=config["l_rate"],
                                    weight_decay=config["weight_decay"])
        elif config["optimizer"] == "sgd":
            optimizer = optim.SGD(params=train_model.parameters(), lr=config["l_rate"],
                                  momentum=config["momentum"],
                                  weight_decay=config["weight_decay"])
        elif config["optimizer"] == "rms":
            optimizer = optim.RMSprop(params=train_model.parameters(), lr=config["l_rate"],
                                      momentum=config["momentum"],
                                      weight_decay=config["weight_decay"])
        elif config["optimizer"] == "adagrad":
            optimizer = optim.Adagrad(params=train_model.parameters(), lr=config["l_rate"],
                                      weight_decay=config["weight_decay"])
        else:
            print("ERROR: Optimizer chosen is not valid")

    return optimizer


def scheduler_selection(config, optimizer):
    scheduler = None
    if config["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"],
                                              gamma=config["gamma"])
    elif config["scheduler"] == "linear":
        scheduler = optim.lr_scheduler.LinearLR(optimizer)
    elif config["scheduler"] == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["gamma"])
    else:
        print("ERROR: Scheduler chosen is not valid")
    return scheduler


def quat_to_euler(q):
    # Convert quaternion in Euler angles
    r = R.from_quat(q)
    return r.as_euler('xyz')


def quat_to_rotmat(q):
    # Convert quaternion in Rotational matrix
    r = R.from_quat(q)
    return r.as_matrix()


def rotmat_to_quat(r):
    # Convert rotational matrix in a quaternion
    rot = R.from_matrix(r)
    return rot.as_quat()


def cal_dist(pred, target):
    # It calculates the Euclidean distance
    return np.linalg.norm(pred - target)


def cal_ori_err(pred, target):
    diff = np.array([0, 0, 0])
    for i in range(3):
        temp = abs(pred[i] - target[i])
        if temp > 180:
            temp = 360 - temp
        diff[i] = temp

    return diff[0], diff[1], diff[2]


def loss_plot(path, model_name, n_epochs, total_loss_training):
    plot_path = path + model_name + "loss_plot"
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(range(n_epochs), total_loss_training, c='red')
    plt.ylabel('Training Loss')
    plt.xlabel('Number of Epochs')
    plt.title("Training Loss Graph")
    plt.grid(True)
    plt.savefig(plot_path)


def trajectory_plot(filepath, target_path, estimated_path):
    s_filepath = filepath + "_1"

    mpl.rc('lines', lw=4)
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    mpl.rc('font', **font)

    targetposes = np.load(target_path)
    estimatedposes = np.load(estimated_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(targetposes[:, 0], targetposes[:, 1], color="red", label="True pose")
    ax.scatter(estimatedposes[:, 0], estimatedposes[:, 1], color="blue", label="Estimated pose")
    plt.ylabel('y')
    plt.xlabel('x')
    ax.set_xlabel('X(m)', fontdict=font)
    ax.set_ylabel('Y(m)', fontdict=font)
    ax.legend(title='2D Trajectory X-Y')
    plt.grid(True)
    # Save the plot as an image
    plt.savefig(s_filepath)

    s_filepath = filepath + "_2"
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(targetposes[:, 0], targetposes[:, 2], color="red", label="True pose")
    ax.scatter(estimatedposes[:, 0], estimatedposes[:, 2], color="blue", label="Estimated pose")
    plt.ylabel('z')
    plt.xlabel('x')
    ax.set_xlabel('X(m)', fontdict=font)
    ax.set_ylabel('Z(m)', fontdict=font)
    ax.legend(title='2D Trajectory X-Z')
    plt.grid(True)
    # Save the plot as an image
    plt.savefig(s_filepath)

    s_filepath = filepath + "_3"
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(targetposes[:, 1], targetposes[:, 2], color="red", label="True pose")
    ax.scatter(estimatedposes[:, 1], estimatedposes[:, 2], color="blue", label="Estimated pose")
    plt.ylabel('z')
    plt.xlabel('y')
    ax.set_xlabel('Y(m)', fontdict=font)
    ax.set_ylabel('Z(m)', fontdict=font)
    ax.legend(title='2D Trajectory Y-Z')
    plt.grid(True)
    # Save the plot as an image
    plt.savefig(s_filepath)


def orientation_plot(filepath, target_path, estimated_path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 15))
    mpl.rc('lines', lw=1.5)
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 12}
    mpl.rc('font', **font)

    targetposes = np.load(target_path)
    estimatedposes = np.load(estimated_path)
    ax1.plot(range(targetposes.shape[0]), targetposes[:, 3], color="red", label="True Orientation")
    ax1.plot(range(targetposes.shape[0]), estimatedposes[:, 3], color="blue", label="Estimated Orientation")
    ax1.set_ylabel('Roll', fontdict=font)
    ax1.grid(True)
    ax2.plot(range(targetposes.shape[0]), targetposes[:, 4], color="red", label="True Orientation")
    ax2.plot(range(targetposes.shape[0]), estimatedposes[:, 4], color="blue", label="Estimated Orientation")
    ax2.set_ylabel('Pitch', fontdict=font)
    ax2.grid(True)
    ax3.plot(range(targetposes.shape[0]), targetposes[:, 5], color="red", label="True Orientation")
    ax3.plot(range(targetposes.shape[0]), estimatedposes[:, 5], color="blue", label="Estimated Orientation")
    ax3.set_ylabel('Yaw', fontdict=font)
    plt.legend(title='Orientation')
    ax3.grid(True)
    # Save the plot as an image
    plt.savefig(filepath)


def orientation_err_plot(filepath, target_path, estimated_path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 15))
    mpl.rc('lines', lw=1.5)
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 12}
    mpl.rc('font', **font)

    targetposes = np.load(target_path)
    estimatedposes = np.load(estimated_path)
    error_ori = np.zeros((targetposes.shape[0], 3))
    for i in range(targetposes.shape[0]):
        temp1 = targetposes[i, 3:]
        temp2 = estimatedposes[i, 3:]
        error_ori[i, 0], error_ori[i, 1], error_ori[i, 2] = cal_ori_err(temp1, temp2)

    ax1.plot(range(targetposes.shape[0]), error_ori[:, 0], color="red")
    ax1.set_ylabel('Roll', fontdict=font)
    ax2.plot(range(targetposes.shape[0]), error_ori[:, 1], color="green")
    ax2.set_ylabel('Pitch', fontdict=font)
    ax3.plot(range(targetposes.shape[0]), error_ori[:, 2], color="blue")
    ax3.set_ylabel('Yaw', fontdict=font)
    # Save the plot as an image
    plt.savefig(filepath)


def position_err_plot(filepath, target_path, estimated_path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 15))
    mpl.rc('lines', lw=1.5)
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 12}
    mpl.rc('font', **font)

    targetposes = np.load(target_path)
    estimatedposes = np.load(estimated_path)
    error_pos = np.zeros((targetposes.shape[0], 3))
    for i in range(targetposes.shape[0]):
        error_pos[i, 0] = cal_dist(targetposes[i, 0], estimatedposes[i, 0])
        error_pos[i, 1] = cal_dist(targetposes[i, 1], estimatedposes[i, 1])
        error_pos[i, 2] = cal_dist(targetposes[i, 2], estimatedposes[i, 2])

    ax1.plot(range(targetposes.shape[0]), error_pos[:, 0], color="red")
    ax1.set_ylabel('X', fontdict=font)
    ax2.plot(range(targetposes.shape[0]), error_pos[:, 1], color="green")
    ax2.set_ylabel('Y', fontdict=font)
    ax3.plot(range(targetposes.shape[0]), error_pos[:, 2], color="blue")
    ax3.set_ylabel('Z', fontdict=font)
    # Save the plot as an image
    plt.savefig(filepath)
