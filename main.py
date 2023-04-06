import numpy as np
from pr3_utils import *
from scipy.linalg import expm


if __name__ == '__main__':

    # load the measurements
    filename = "/Users/vb/Desktop/pr3/ECE276A_PR3/code/data/03.npz"
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
    W = np.diag([0.03, 0.03, 0.03, 0.002, 0.002, 0.002])

    # trajectory prediction
    for i in range(np.size(t) - 1):
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
    # visualize_trajectory_2d(pose, path_name="Unknown", show_ori=True)

    # (b) Landmark Mapping via EKF Update

    # get required number of best features from all features
    def get_best_features(req):
        best = np.zeros((features.shape[0], req, features.shape[2]))
        count = np.zeros(features.shape[1])

        # check the number of time each feature is observed
        for i in range(1, np.size(t)):
            reqind = np.sum(features[:, :, i], axis=0) != -4
            count[reqind] += 1

        # get the required number of features observed
        ind = np.sort(np.flip(np.argsort(count))[:req])
        best = features[:, ind, :]
        return best

    m = 1000
    best_features = get_best_features(m)

    # get landmark coordinates in 3D from the features in 2D using stereo camera model
    def get_landmark_3D(ul, vl, ur, vr):
        avg = (vl + vr) / 2
        z = K[0, 0] * b / (ul - ur)
        x = z * (ul - K[0, 2]) / K[0, 0]
        y = z * (avg - K[1, 2]) / K[1, 1]
        return np.array([x, y, z, 1]).reshape(4, 1)

    # create array
    landmark_world = np.zeros(np.shape(best_features))

    # landmarks in world frame
    for i in range(np.size(t) - 1):
        for j in range(m):
            if sum(best_features[:, j, i]) != -4:
                ul, vl, ur, vr = best_features[:, j, i]
                point = get_landmark_3D(ul, vl, ur, vr)
                # get landmark coordinates in world frame from the optical frame
                landmark_world[:, j, i] = (
                    pose[:, :, i] @ imu_T_cam @ point).reshape(4)

    # mu_feature and mu_flat
    mu_feature = np.zeros(np.shape(landmark_world))
    mu_flat = np.zeros((3*m, 1))
    for i in range(m):
        reqind = np.where(landmark_world[3, i, :] == 1)[0]
        avg = np.sum(landmark_world[:, i, :], axis=1)/len(reqind)
        mu_feature[:, i, reqind] = avg.reshape(4, 1)
        mu_flat[3*i:3*(i+1)] = avg[:3].reshape(3, 1)
    sigma_feature = 0.01 * np.identity(3*m)

    # parameters
    checker = np.ones(m)
    cam_T_imu = np.linalg.inv(imu_T_cam)
    P_trans = np.vstack((np.identity(3), np.zeros(3)))
    Ks = np.array([[K[0, 0], 0, K[0, 2], 0],
                   [0, K[1, 1], K[1, 2], 0],
                   [K[0, 0], 0, K[0, 2], -K[0, 0] * b],
                   [0, K[1, 1], K[1, 2], 0]])

    # compute dpi/dq
    def diff_pi(q):
        mat = np.identity(4)
        mat[:, 2] = np.array([-q[0]/q[2], -q[1] /
                             q[2], 0, -q[3]/q[2]])
        return mat/q[2]

    # s dot creation
    def sdot(s):
        return np.block([[np.identity(3), -hatmap(s)],
                         [np.zeros((1, 3)), np.zeros((1, 3))]])

    # run time loop
    for i in range(np.size(t)):
        # parameters
        ind = np.where(mu_feature[3, :, i] == 1)[0]
        Nt = len(ind)
        z_pose_trans = cam_T_imu @ np.linalg.inv(pose[:, :, i])
        H_pose_trans = z_pose_trans @ P_trans

        # create arrays
        zt = np.zeros((4*Nt, 1))
        z = np.zeros((4*Nt, 1))
        H = np.zeros((4*Nt, 3*m))
        IV = 5*np.identity(4*Nt)

        # compute predicted observation and Jacobian
        for j in range(Nt):
            if checker[ind[j]]:
                arg = z_pose_trans @ mu_feature[:, ind[j], i]
                pi = arg / arg[2]
                zt[4*j:4*(j+1), 0] = Ks @ pi
                z[4*j:4*(j+1), 0] = best_features[:, ind[j], i]
                H[4*j:4*(j+1), ind[j]:ind[j] +
                  3] = Ks @ diff_pi(arg) @ H_pose_trans
                checker[ind[j]] = 0

        # compute Kalman gain
        Kgain = sigma_feature @ H.T @ np.linalg.inv(
            (H @ sigma_feature @ H.T) + IV)

        # update mean and variance
        mu_flat += Kgain @ (z - zt)
        sigma_feature = (np.identity(3*m) - Kgain @ H) @ sigma_feature

        # (c) Visual-Inertial SLAM

        # create arrays
        zt_new = np.zeros((4*Nt, 1))
        H_new = np.zeros((4*Nt, 6))

        # compute predicted observation and Jacobian for VI-SLAM
        for j in range(Nt):
            mj = mu_flat[3*ind[j]:3*(ind[j]+1), 0]
            mj_ = np.concatenate((mj, np.array([1])), axis=0)
            arg_new = z_pose_trans @ mj_
            pi_new = arg_new / arg_new[2]
            zt_new[4*j:4*(j+1), 0] = Ks @ pi_new
            dot_term = sdot(np.linalg.inv(pose[:, :, i]) @ mj_)
            H_new[4*j:4*(j+1), :] = - \
                Ks @ diff_pi(arg_new) @ cam_T_imu @ dot_term

        # compute Kalman gain for VI-SLAM
        Kgain_new = sigma[:, :, i] @ H_new.T @ np.linalg.inv(
            (H_new @ sigma[:, :, i] @ H_new.T) + IV)

        # update mean and variance for VI-SLAM
        exp_arg = Kgain_new @ (z - zt_new)
        exp_term = expm(uhat(exp_arg[:3], exp_arg[3:]))
        pose[:, :, i] = pose[:, :, i] @ exp_term
        mult_term = np.identity(6) - Kgain_new @ H_new
        add_term = Kgain_new @ IV @ Kgain_new.T
        sigma[:, :, i] = mult_term @ sigma[:, :, i] @ mult_term.T + add_term

    landmark_points = mu_flat.reshape((m, 3))
    # visualize trajectory
    visualize_trajectory_2d(
        pose, landmark_points[:, 0], landmark_points[:, 1], path_name="Unknown")
