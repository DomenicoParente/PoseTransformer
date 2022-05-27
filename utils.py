import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def quat_to_euler(q):
    # Convert quaternion in Euler angles
    w, x, y, z = q[0], q[1], q[2], q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    roll = np.rad2deg(roll)
    pitch = np.rad2deg(pitch)
    yaw = np.rad2deg(yaw)
    if roll < 0:
        roll = 360 + roll
    if pitch < 0:
        pitch = 360 + pitch
    if yaw < 0:
        yaw = 360 + yaw

    return np.array([roll, pitch, yaw])


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
    fig, ax = plt.subplots(figsize=(10, 10))
    mpl.rc('lines', lw=4)
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}
    mpl.rc('font', **font)

    targetposes = np.load(target_path)
    estimatedposes = np.load(estimated_path)
    ax.scatter(targetposes[:, 0], targetposes[:, 1], color="red", label="True pose")
    ax.scatter(estimatedposes[:, 0], estimatedposes[:, 1], color="blue", label="Estimated pose")
    plt.ylabel('y')
    plt.xlabel('x')
    ax.set_xlabel('X(m)', fontdict=font)
    ax.set_ylabel('Y(m)', fontdict=font)
    ax.legend(title='2D Trajectory')
    plt.grid(True)
    # Save the plot as an image
    plt.savefig(filepath)


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
    error_ori = np.zeros((targetposes.shape[0], 3))
    for i in range(targetposes.shape[0]):
        temp1 = targetposes[i, :3]
        temp2 = estimatedposes[i, :3]
        error_ori[i, 0], error_ori[i, 1], error_ori[i, 2] = cal_ori_err(temp1, temp2)

    ax1.plot(range(targetposes.shape[0]), error_ori[:, 0], color="red")
    ax1.set_ylabel('X', fontdict=font)
    ax2.plot(range(targetposes.shape[0]), error_ori[:, 1], color="green")
    ax2.set_ylabel('Y', fontdict=font)
    ax3.plot(range(targetposes.shape[0]), error_ori[:, 2], color="blue")
    ax3.set_ylabel('Z', fontdict=font)
    # Save the plot as an image
    plt.savefig(filepath)

