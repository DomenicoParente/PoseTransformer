import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch.optim as optim


def optimizer_selection(config, train_model, sx, sq):
    """
    It allows to choose a certain optimizer.
    """
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
    """
    It allows to choose a certain scheduler.
    """
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
    """
    It converts orientation quaternions N x (x, y, z, w) in Euler angles N x (roll, pitch, yaw).
    The scalar values w must be in the last position.
    INPUT:
        q: orientation quaternion.
    OUTPUT:
        x,y,z: pitch, roll and yaw angles in degrees.
    """
    r = R.from_quat(q)
    return r.as_euler('xyz', degrees=True)


def q_conjugate(q):
    """
    It calculates the conjugate quaternion.
    The scalar values w must be in the last position.
    INPUT:
        q: (x, y, z, w) quaternion.
    OUTPUT:
        qinv: conjugate quaternion.
    """
    if q.shape[0] != 4:
        raise Exception("Illegal quaternion format: it is not composed by 4 values.")
    q_c = np.concatenate((-q[:3], q[3:]))
    return q_c


def q_norm(q):
    """
    It calculates the inverse quaternion.
    The scalar values w must be in the last position.
    INPUT:
        q: (x, y, z, w) quaternion.
    OUTPUT:
        qinv: quaternion's norm.
    """
    if q.shape[0] != 4:
        raise Exception("Illegal quaternion format: it is not composed by 4 values.")
    norm = np.linalg.norm(q)
    return norm


def q_inverse(q):
    """
    It calculates the inverse quaternion.
    The scalar values w must be in the last position.
    INPUT:
        q: (x, y, z, w) quaternion.
    OUTPUT:
        q_inv: inverse quaternion if calculable.
    """
    if q.shape[0] != 4:
        raise Exception("Illegal quaternion format: it is not composed by 4 values.")
    if q[0] == 0 and q[1] == 0 and q[2] == 0 and q[3] == 0:
        raise Exception("Invalid quaternion (0, 0, 0, 0)")

    q_inv = q_conjugate(q)/pow(q_norm(q), 2)
    return q_inv

def qmul(q1, q2):
    """
        It multiplies two quaternions.
        The scalar values w must be in the last position.
        INPUT:
            q1, q2: (x, y, z, w) quaternions.
        OUTPUT:
            res: quaternions product normalized.
        """
    w = q1[3] * q2[3] - np.dot(q1[:3], q2[:3])
    v = q2[3] * q1[:3] + q1[3] * q2[:3] + np.cross(q1[:3], q2[:3])
    res = np.concatenate((v, np.array([w])))
    res_norm = q_norm(res)
    return res/res_norm


def quat_to_rotmat(q):
    """
    It converts orientation quaternion (x, y, z, w) in rotational matrix (3x3).
    The scalar values w must be in the last position.
    INPUT:
        q: orientation quaternion.
    OUTPUT:
        3x3 rotational matrix.
    """
    r = R.from_quat(q)
    return r.as_matrix()


def rotmat_to_quat(rmat):
    """
    It converts rotational matrix (3x3) in quaternion (x, y, z, w).
    The scalar values w is in the last position.
    INPUT:
        rmat: 3x3 rotational matrix.
    OUTPUT:
        orientation quaternion (x, y, z, w)
    """
    rot = R.from_matrix(rmat)
    return rot.as_quat()


def pose_to_homcoord(pose):
    """
    It converts 7-values pose to 12-values pose (homogeneous coordinates).
    Firstly it generates the homogeneous coordinates matrix (4x4).
    [ R  t]      R: rotational matrix (3x3)
    [ 0  1]      t: translation vector (3x1)
    After that it is reduced to a 12-values vector.
    INPUT:
        pose: 7-values pose (position + orientation)
    OUTPUT:
        12-values vector that is homogeneous coordinates matrix without the last line squeezed.
    """
    pos = pose[:3]
    ori = pose[3:]
    # it constraints to one hemisphere
    ori *= np.sign(ori[3])
    hc_matrix = np.matrix(np.eye(4))
    hc_matrix[:3, :3] = quat_to_rotmat(ori)
    hc_matrix[:3, 3] = np.matrix(pos).T
    return np.array(hc_matrix[0:3, :]).reshape(1, 12)


def homcoord_vec_to_mat(pose):
    """
    It converts 12-values pose to homogeneous coordinates matrix.
    INPUT:
        pose: 12-values vector that is homogeneous coordinates matrix without the last line squeezed.
    OUTPUT:
        homogeneous coordinates matrix (4x4)
    """
    mat = np.eye(4)
    mat[0:3, :] = pose.reshape(3, 4)
    return np.matrix(mat)


def homcoord_to_pose(hc_pose):
    """
    It converts 12-values pose to 7-values pose.
    INPUT:
        hpose: 12-values vector that is homogeneous coordinates matrix without the last line squeezed.
    OUTPUT:
        7-values pose (position + orientation)
    """
    hc_pose = hc_pose.reshape(3, 4)
    pos = np.array([hc_pose[0][3], hc_pose[1][3], hc_pose[2][3]])
    hc_pose = np.delete(hc_pose, 3, 1)
    ori = rotmat_to_quat(hc_pose)
    return np.concatenate((pos, ori))


def glob_to_rel(pose1, pose2):
    """
    It converts two global poses in a 7-values relative pose.
    INPUT:
        pose1: first global 7-values pose (position + orientation).
        pose2: next global 7-values pose (position + orientation).
    OUTPUT:
        rel_pose: relative 7-values pose between the two global poses given as input.
    """
    t = pose2[:3] - pose1[:3]
    q = qmul(q_inverse(pose1[3:]), pose2[3:])
    rel_pose = np.concatenate((t, q))
    return rel_pose


def rel_to_glob(data):
    """
    It converts the series of relatives pose in global pose using as starting point [0 0 0 0 0 0 1].
    Starting point: position (0, 0, 0) + orientation quaternion ( 0, 0, 0, 1).
    INPUT:
        data: list of all relative 7-values poses.
    OUTPUT:
        all_poses: list of all global 7-values poses.
    """
    data = np.array(data)
    data_size = data.shape[0]
    all_poses = np.zeros((data_size + 1, 7))
    temp = [0, 0, 0, 0, 0, 0, 1]
    all_poses[0, :] = temp
    pose = np.matrix(np.eye(4))
    for i in range(data_size):
        homcoord_mat = homcoord_vec_to_mat(pose_to_homcoord(data[i, :]))
        pose = pose * homcoord_mat
        pose_line = np.array(pose[0:3, :]).reshape(1, 12)
        all_poses[i + 1, :] = homcoord_to_pose(pose_line)
    return all_poses


def cal_dist(pred, target):
    """
    It calculates the Euclidean distance.
    INPUT:
        pred: predicted position (x, y, z)
        target: true position (x, y, z)
    OUTPUT:
        distance between the two points given as input.
    """
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
