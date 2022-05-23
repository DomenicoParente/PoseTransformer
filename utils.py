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
        roll = 360 - roll
    if pitch < 0:
        pitch = 360 - pitch
    if yaw < 0:
        yaw = 360 - yaw

    return np.array([roll, pitch, yaw])


def cal_dist(pred, target):
    # It calculates the Euclidean distance
    return np.linalg.norm(pred - target, 2)


def cal_ori_err(pred, target):
    return abs(pred[0] - target[0]), abs(pred[1] - target[1]), abs(pred[2] - target[2])


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
