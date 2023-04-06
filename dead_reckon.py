import numpy as np
from pr3_utils import *
from scipy.linalg import expm

if __name__ == '__main__':

    # load the measurements
    filename = "/Users/vb/Desktop/pr3/ECE276A_PR3/code/data/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(
        filename)

    # (a) IMU Localization via EKF Prediction

    # vector hatmap creation
    def hatmap(x):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])

    # u hat creation
    def uhat(v, w):
        return np.block([[hatmap(w), v.reshape((3, 1))],
                         [np.zeros((1, 3)), 0]])

    # u curly hat creation
    def ucurly(v, w):
        return np.block([[hatmap(w), hatmap(v)],
                         [np.zeros((3, 3)), hatmap(w)]])

    # create arrays
    pose = np.zeros((4, 4, np.size(t)))
    mu = np.zeros((4, 4, np.size(t)))
    delmu = np.zeros((6, 1, np.size(t)))
    sigma = np.zeros((6, 6, np.size(t)))

    # intialize values
    mu0 = np.identity(4)
    sigma0 = np.zeros((6, 6))
    delmu0 = np.diag(np.random.normal(0, sigma0)).reshape((6, 1))

    pose[:, :, 0] = mu0
    mu[:, :, 0] = mu0
    delmu[:, :, 0] = delmu0
    sigma[:, :, 0] = sigma0

    # noise
    W = np.diag(np.zeros(6))

    # trajectory prediction
    for i in range(t.shape[1] - 1):
        # dynamics parameters
        v = linear_velocity[:, i]
        w = angular_velocity[:, i]
        uh = uhat(v, w)
        uc = ucurly(v, w)
        tau = t[0, i+1] - t[0, i]
        wt = np.diag(np.random.normal(0, W)).reshape((6, 1))

        # mean, delta_mean, variance
        mu[:, :, i+1] = mu[:, :, i] @ expm(tau*uh)
        delmu[:, :, i+1] = expm(-tau*uc) @ delmu[:, :, i] + wt
        sigma[:, :, i+1] = expm(-tau*uc) @ sigma[:, :, i] @ expm(-tau*uc).T + W

        # pose
        delmuh = uhat(delmu[:3, 0, i], delmu[3:, 0, i])
        pose[:, :, i+1] = mu[:, :, i] @ expm(delmuh)

    # visualize trajectory
    visualize_trajectory_2d(pose, path_name="Trajectory", show_ori=True)
