import itertools
import numpy as np
import os
import pykitti
from kalmans import *
from geo_transforms import *
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def pykitti_dataset():
    # Change this to the directory where you store KITTI data
    curr_dir_path = os.getcwd()
    basedir = curr_dir_path + '/kitti_data'

    # Specify the dataset to load
    date = '2011_09_30'
    drive = '0033'

    # Load the data.
    dataset = pykitti.raw(basedir, date, drive)

    return dataset

def confidence_ellipse(mean_x, mean_y, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def KF(gt_trajectory_xyz, obs_trajectory_xyz, xy_obs_noise_std, N, ts,obs_forward_velocities, obs_yaw_rates,gt_yaws, gt_east_velocities , gt_north_velocities):

    # Initial state
    mu = np.matrix( [obs_trajectory_xyz[0, 0],
                    obs_trajectory_xyz[1, 0],
                    3.0,
                    3.0] ).T

    # Initial Covariance matrix
    P = np.array([
        [3.**2, 0., 0., 0.],
        [0., 3.**2, 0., 0.],
        [0., 0., 7**2, 0.],
        [0., 0., 0., 7**2]
    ])

    # Measurement noise matrix
    Q = np.matrix([[xy_obs_noise_std**2, 0],
                   [0, xy_obs_noise_std**2]])

    result = []
    cov_result = []
    dr_results = []

    dr_results.append(mu.tolist())
    result.append(mu.tolist())
    cov_result.append(P.tolist())

    # array to store estimated error variance
    var_x = [P[0, 0], ]
    var_y = [P[1, 1], ]
    var_vx = [P[2, 2], ]
    var_vy = [P[3, 3], ]

    sig_a = 1.6

    t_last = 0.


    for t_idx in range(1, N):

        # get control input `u = [v, omega] + noise`
        u = np.array([
            obs_forward_velocities[t_idx],
            obs_yaw_rates[t_idx]
        ])

        t = ts[t_idx]
        dt = t - t_last
        z = (obs_trajectory_xyz[0][t_idx],obs_trajectory_xyz[1][t_idx])

        mu, P = KalmanFilter(mu, P, z, Q, dt, sig_a)

        if t > 5.0:
            dead_reckoning = True
            v, omega = u
            r = v / omega  # turning radius

            dtheta = omega * dt
            theta = gt_yaws[t_idx] + np.random.normal(0, 0.5)

            dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
            dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)

            dr_results.append(( np.array(dr_results[-1]) + np.matrix([dx, dy, 0.0, 0.0]).T).tolist())
        else:
            dr_results.append((mu).tolist())

        result.append((mu).tolist())
        cov_result.append((P).tolist())

        var_x.append(P[0, 0])
        var_y.append(P[1, 1])
        var_vx.append(P[2, 2])
        var_vy.append(P[3, 3])

        t_last = t


    mu_x, mu_y, mu_vx, mu_vy = zip(*result)
    dr_x, dr_y, _, _ = zip(*dr_results)

    mu_x = np.array(mu_x).squeeze()
    mu_y = np.array(mu_y).squeeze()
    mu_vx = np.array(mu_vx).squeeze()
    mu_vy = np.array(mu_vy).squeeze()
    dr_x = np.array(dr_x).squeeze()
    dr_y = np.array(dr_y).squeeze()

    xs_gt, ys_gt, _ = gt_trajectory_xyz

    # Evaluation
    e_x = xs_gt[100:] - mu_x[100:]
    e_y = ys_gt[100:] - mu_y[100:]

    maxE = max(abs(e_x) + abs(e_y))
    indx = np.argmax(abs(e_x) + abs(e_y))

    N = len(e_x) - 99
    RMSE = np.sqrt((1/N) * (e_x[100:] ** 2 + e_y[100:] ** 2).sum())

    print("maxE: " + str(maxE) + "   RMSE: " + str(RMSE)+ "   indx: " + str(indx))


    if True:
        fig, ax = plt.subplots(1, 4, figsize=(9, 6))

        ax[0].plot(ts, gt_trajectory_xyz[0], lw=2, label='ground-truth')
        # ax.plot(ts, obs_trajectory_xyz[0], lw=0, marker='.', markersize=3, alpha=0.4, label='observed')
        ax[0].plot(ts, mu_x, lw=1, label='estimated', color='r')

        ax[0].set_xlabel('time elapsed [sec]')
        ax[0].set_ylabel('X [m]')
        ax[0].legend()

        ax[1].plot(ts, gt_trajectory_xyz[1], lw=2, label='ground-truth')
        # ax.plot(ts, obs_trajectory_xyz[0], lw=0, marker='.', markersize=3, alpha=0.4, label='observed')
        ax[1].plot(ts, mu_y, lw=1, label='estimated', color='r')

        ax[1].set_xlabel('time elapsed [sec]')
        ax[1].set_ylabel('Y [m]')
        ax[1].legend()

        ax[2].plot(ts, gt_east_velocities, lw=2, label='ground-truth')
        # ax.plot(ts, obs_trajectory_xyz[0], lw=0, marker='.', markersize=3, alpha=0.4, label='observed')
        ax[2].plot(ts, mu_vx, lw=1, label='estimated', color='r')

        ax[2].set_xlabel('time elapsed [sec]')
        ax[2].set_ylabel('v_x [m/s]')
        ax[2].legend()

        ax[3].plot(ts, gt_north_velocities, lw=2, label='ground-truth')
        # ax.plot(ts, obs_trajectory_xyz[0], lw=0, marker='.', markersize=3, alpha=0.4, label='observed')
        ax[3].plot(ts, mu_vy, lw=1, label='estimated', color='r')

        ax[3].set_xlabel('time elapsed [sec]')
        ax[3].set_ylabel('v_x [m/s]')
        ax[3].legend()

        plt.show()

    if True:
        fig, ax = plt.subplots(4, 1, figsize=(15, 8))

        ax[0].plot(ts, mu_x - gt_trajectory_xyz[0], lw=1.5, label='estimation error')
        ax[0].plot(ts, np.sqrt(var_x), lw=1.5, label='estimated 1-sigma interval', color='darkorange')
        ax[0].plot(ts, -np.sqrt(var_x), lw=1.5, label='', color='darkorange')

        ax[0].set_xlabel('time elapsed [sec]')
        ax[0].set_ylabel('X estimation error [m]')
        ax[0].legend()

        ax[1].plot(ts, mu_y - gt_trajectory_xyz[1], lw=1.5, label='estimation error')
        ax[1].plot(ts, np.sqrt(var_y), lw=1.5, label='estimated 1-sigma interval', color='darkorange')
        ax[1].plot(ts, -np.sqrt(var_y), lw=1.5, label='', color='darkorange')

        ax[1].set_xlabel('time elapsed [sec]')
        ax[1].set_ylabel('Y estimation error [m]')
        # ax[1].legend()

        ax[2].plot(ts, mu_vx - gt_east_velocities, lw=1.5, label='estimation error')
        ax[2].plot(ts, np.sqrt(var_vx), lw=1.5, label='estimated 1-sigma interval', color='darkorange')
        ax[2].plot(ts, -np.sqrt(var_vx), lw=1.5, label='', color='darkorange')

        ax[2].set_ylim(-5, 5)

        ax[2].set_xlabel('time elapsed [sec]')
        ax[2].set_ylabel('V_x estimation error [m/s]')
        # ax[2].legend()

        ax[3].plot(ts, mu_vy - gt_north_velocities, lw=1.5, label='estimation error')
        ax[3].plot(ts, np.sqrt(var_vy), lw=1.5, label='estimated 1-sigma interval', color='darkorange')
        ax[3].plot(ts, -np.sqrt(var_vy), lw=1.5, label='', color='darkorange')

        ax[3].set_ylim(-5, 5)

        ax[3].set_xlabel('time elapsed [sec]')
        ax[3].set_ylabel('V_y estimation error [m/s]')
        # ax[3].legend()

        plt.show()

    if True:
        xs, ys, _ = obs_trajectory_xyz
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.plot(xs_gt, ys_gt, lw=2, label='ground-truth trajectory')
        ax.plot(xs, ys, lw=0, marker='.', markersize=4, alpha=1., label='observed trajectory')
        ax.plot(mu_x, mu_y, lw=1, label='estimated trajectory', color='r')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid()
        plt.show()

    if True:
        from matplotlib.animation import FuncAnimation
        xs_obs, ys_obs, _ = obs_trajectory_xyz
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        inx = 0
        ax.plot(xs_gt[:inx], ys_gt[:inx], lw=2, color='k', label='gt')
        ax.plot(xs_obs[inx], ys_obs[inx], lw=0, marker='.', markersize=4, alpha=1., color='orange', label='measurement')
        ax.plot(mu_x[:inx], mu_y[:inx], lw=1, color='r', label='KF')
        ax.plot(dr_x[:inx], dr_y[:inx], lw=1, color='g', label='DR')

        for x, y in zip(mu_x, mu_y):
            ax.plot(xs_gt[:inx], ys_gt[:inx], lw=2,color='k')
            ax.plot(xs_obs[inx], ys_obs[inx], lw=0, marker='.', markersize=4, alpha=1.,color='orange')
            ax.plot(mu_x[:inx], mu_y[:inx], lw=1, color='r')
            ax.plot(dr_x[:inx], dr_y[:inx], lw=1, color='g')
            confidence_ellipse(x, y, np.array(cov_result[inx]), ax, edgecolor='blue')

            inx += 1

            if False:
                # windowed plot, i dont really like it
                ax.set_xlim(mu_x[inx]- 50, mu_x[inx] + 50)
                ax.set_ylim(mu_y[inx]- 50, mu_y[inx] + 50)

            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.grid()
            ax.legend()
            # plt.savefig(str(inx))

            plt.pause(0.005)

def KF_constant_acc(gt_trajectory_xyz, obs_trajectory_xyz, xy_obs_noise_std, N, ts,obs_forward_velocities, obs_yaw_rates,gt_yaws, gt_east_velocities , gt_north_velocities):

    # v , y , vx , vy ,ax , ay
    mu = np.matrix( [obs_trajectory_xyz[0, 0],
                    obs_trajectory_xyz[1, 0],
                    3.0,
                    3.0,
                     0,
                     0,] ).T

    # Initial Covariance matrix
    P = np.array([
        [5.**2, 0., 0., 0., 0.0, 0.0],
        [0., 5.**2, 0., 0., 0.0, 0.0],
        [0., 0., 10**2, 0., 0.0, 0.0],
        [0., 0., 0., 10**2, 0.0, 0.0],
        [0., 0., 0., 0.0, 2 ** 2, 0.0],
        [0., 0., 0., 0., 0.0, 2 ** 2]
    ])

    # Measurement noise matrix
    Q = np.matrix([[xy_obs_noise_std**2, 0],
                   [0, xy_obs_noise_std**2]])

    result = []
    cov_result = []
    dr_results = []

    result.append(mu.tolist())
    cov_result.append(P.tolist())
    dr_results.append(mu.tolist())

    # array to store estimated error variance
    var_x = [P[0, 0], ]
    var_y = [P[1, 1], ]
    var_vx = [P[2, 2], ]
    var_vy = [P[3, 3], ]
    var_ax = [P[4, 5], ]
    var_ay = [P[5, 5], ]

    sig_a = 1.8

    t_last = 0.



    for t_idx in range(1, N):

        # get control input `u = [v, omega] + noise`
        u = np.array([
            obs_forward_velocities[t_idx],
            obs_yaw_rates[t_idx]
        ])

        t = ts[t_idx]
        dt = t - t_last
        z = (obs_trajectory_xyz[0][t_idx],obs_trajectory_xyz[1][t_idx])

        mu, P = KalmanFilter(mu, P, z, Q, dt, sig_a, 'constant_acc')

        if  t > 5.0:
            v, omega = u
            r = v / omega  # turning radius

            dtheta = omega * dt
            theta = gt_yaws[t_idx] + np.random.normal(0, 0.5)

            dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
            dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)

            dr_results.append(( np.array(dr_results[-1]) + np.matrix([dx, dy, 0.0, 0.0, 0.0 , 0.0]).T).tolist())
        else:
            dr_results.append((mu).tolist())

        result.append((mu).tolist())
        cov_result.append((P).tolist())

        var_x.append(P[0, 0])
        var_y.append(P[1, 1])
        var_vx.append(P[2, 2])
        var_vy.append(P[3, 3])

        t_last = t


    mu_x, mu_y, mu_vx, mu_vy, mu_ax, mu_ay = zip(*result)

    mu_x = np.array(mu_x).squeeze()
    mu_y = np.array(mu_y).squeeze()

    xs_gt, ys_gt, _ = gt_trajectory_xyz

    # Evaluation
    e_x = xs_gt[100:] - mu_x[100:]
    e_y = ys_gt[100:] - mu_y[100:]

    maxE = max(abs(e_x) + abs(e_y))
    indx = np.argmax(abs(e_x) + abs(e_y))

    N = len(e_x) - 99
    RMSE = np.sqrt((1/N) * (e_x[100:] ** 2 + e_y[100:] ** 2).sum())

    print("maxE: " + str(maxE) + "   RMSE: " + str(RMSE)+ "   indx: " + str(indx))



    if True:
        xs, ys, _ = obs_trajectory_xyz
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.plot(xs_gt, ys_gt, lw=2, label='ground-truth trajectory')
        ax.plot(xs, ys, lw=0, marker='.', markersize=4, alpha=1., label='observed trajectory')
        ax.plot(mu_x, mu_y, lw=1, label='estimated trajectory', color='r')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid()
        plt.show()

def EKF(obs_trajectory_xyz, obs_forward_velocities, xy_obs_noise_std, forward_velocity_noise_std,obs_yaw_rates, yaw_rate_noise_std, N, ts):

    initial_yaw_std = np.pi
    initial_yaw = gt_yaws[0] + np.random.normal(0, initial_yaw_std)

    mu_0 = np.array([
        obs_trajectory_xyz[0, 0],
        obs_trajectory_xyz[1, 0],
        initial_yaw
    ])
    # covariance for initial state estimation error (Sigma_0)

    P = np.array([
        [4 ** 2., 0., 0.],
        [0., 4 ** 2., 0.],
        [0., 0.,(initial_yaw_std/2) ** 2.]
    ])

    # measurement error covariance Q
    Q = np.array([
        [xy_obs_noise_std ** 2., 0.],
        [0., xy_obs_noise_std ** 2.]
    ])

    #  state transition noise covariance R

    R = np.array([[forward_velocity_noise_std ** 2., 0.],
                  [0., yaw_rate_noise_std ** 2.]])

    # initialize Kalman filter
    kf = ExtendedKalmanFilter(mu_0, P)

    # array to store estimated 2d pose [x, y, theta]
    mu_x = [mu_0[0], ]
    mu_y = [mu_0[1], ]
    mu_theta = [mu_0[2], ]

    # array to store estimated 2d pose [x, y, theta]
    dr_x = [mu_0[0], ]
    dr_y = [mu_0[1], ]
    dr_theta = [mu_0[2], ]

    # array to store estimated error variance of 2d pose
    var_x = [P[0, 0], ]
    var_y = [P[1, 1], ]
    var_theta = [P[2, 2], ]

    t_last = 0.
    for t_idx in range(1, N):
        t = ts[t_idx]
        dt = t - t_last

        # get control input `u = [v, omega] + noise`
        u = np.array([
            obs_forward_velocities[t_idx],
            obs_yaw_rates[t_idx]
        ])

        # propagate!
        kf.propagate(u, dt, R)

        # get measurement `z = [x, y] + noise`
        z = np.array([
            obs_trajectory_xyz[0, t_idx],
            obs_trajectory_xyz[1, t_idx]
        ])

        # update!
        kf.update(z, Q)

        if t > 5.0:
            # propagate state x
            v, omega = u
            r = v / omega  # turning radius
            theta = mu_theta[t_idx-1]

            dtheta = omega * dt
            dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
            dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)

            # save estimated state to analyze later
            dr_x.append(dr_x[-1] + dx)
            dr_y.append(dr_y[-1] + dy)
            dr_theta.append(dr_theta[-1] + normalize_angles(dtheta))
        else:
            dr_x.append(kf.mu[0])
            dr_y.append(kf.mu[1])
            dr_theta.append(normalize_angles(kf.mu[2]))

        # save estimated state to analyze later
        mu_x.append(kf.mu[0])
        mu_y.append(kf.mu[1])
        mu_theta.append(normalize_angles(kf.mu[2]))

        # save estimated variance to analyze later
        var_x.append(kf.P[0, 0])
        var_y.append(kf.P[1, 1])
        var_theta.append(kf.P[2, 2])

        t_last = t

    mu_x = np.array(mu_x)
    mu_y = np.array(mu_y)
    mu_theta = np.array(mu_theta)

    dr_x = np.array(dr_x)
    dr_y = np.array(dr_y)
    dr_theta = np.array(dr_theta)

    var_x = np.array(var_x)
    var_y = np.array(var_y)
    var_theta = np.array(var_theta)

    cov_results_xy = [np.array([[vrx,0.0],[0.0,vry]]) for vrx,vry in  zip(var_x,var_y)]

    xs_gt, ys_gt, _ = gt_trajectory_xyz
    xs_obs, ys_obs, _ = obs_trajectory_xyz


    # Evaluation
    e_x = xs_gt[100:] - np.array(mu_x).squeeze()[100:]
    e_y = ys_gt[100:] - np.array(mu_y).squeeze()[100:]

    maxE = max(abs(e_x) + abs(e_y))
    indx = np.argmax(abs(e_x) + abs(e_y))

    N = len(e_x)
    RMSE = np.sqrt((1/N) * (e_x[100:] ** 2 + e_y[100:] ** 2).sum() )

    print("maxE: " + str(maxE) + "   RMSE: " + str(RMSE)+ "   indx: " + str(indx))

    if False:
        fig, ax = plt.subplots(1, 3, figsize=(9, 6))

        ax[0].plot(ts, gt_trajectory_xyz[0], lw=2, label='ground-truth')
        # ax.plot(ts, obs_trajectory_xyz[0], lw=0, marker='.', markersize=3, alpha=0.4, label='observed')
        ax[0].plot(ts, mu_x, lw=1, label='estimated', color='r')

        ax[0].set_xlabel('time elapsed [sec]')
        ax[0].set_ylabel('X [m]')
        ax[0].legend()

        ax[1].plot(ts, gt_trajectory_xyz[1], lw=2, label='ground-truth')
        # ax.plot(ts, obs_trajectory_xyz[0], lw=0, marker='.', markersize=3, alpha=0.4, label='observed')
        ax[1].plot(ts, mu_y, lw=1, label='estimated', color='r')

        ax[1].set_xlabel('time elapsed [sec]')
        ax[1].set_ylabel('Y [m]')

        ax[2].plot(ts, gt_yaws, lw=2, label='ground-truth')
        # ax.plot(ts, obs_trajectory_xyz[0], lw=0, marker='.', markersize=3, alpha=0.4, label='observed')
        ax[2].plot(ts, mu_theta, lw=1, label='estimated', color='r')

        ax[2].set_xlabel('time elapsed [sec]')
        ax[2].set_ylabel('Theta [rad]')

        plt.show()

    if False:
        fig, ax = plt.subplots(3, 1, figsize=(9, 6))

        ax[0].plot(ts, mu_x - gt_trajectory_xyz[0], lw=1.5, label='estimation error')
        ax[0].plot(ts, np.sqrt(var_x), lw=1.5, label='estimated 1-sigma interval', color='darkorange')
        ax[0].plot(ts, -np.sqrt(var_x), lw=1.5, label='', color='darkorange')
        ax[0].set_ylim(-5, 5)
        ax[0].set_xlabel('time elapsed [sec]')
        ax[0].set_ylabel('X estimation error [m]')
        ax[0].legend();

        ax[1].plot(ts, mu_y - gt_trajectory_xyz[1], lw=1.5, label='estimation error')
        ax[1].plot(ts, np.sqrt(var_y), lw=1.5, label='estimated 1-sigma interval', color='darkorange')
        ax[1].plot(ts, -np.sqrt(var_y), lw=1.5, label='', color='darkorange')
        ax[1].set_ylim(-5, 5)
        ax[1].set_xlabel('time elapsed [sec]')
        ax[1].set_ylabel('Y estimation error [m]')

        ax[2].plot(ts, normalize_angles(mu_theta - gt_yaws), lw=1.5, label='estimation error')
        ax[2].plot(ts, np.sqrt(var_theta), lw=1.5, label='estimated 1-sigma interval', color='darkorange')
        ax[2].plot(ts, -np.sqrt(var_theta), lw=1.5, label='', color='darkorange')

        ax[2].set_ylim(-0.5, 0.5)

        ax[2].set_xlabel('time elapsed [sec]')
        ax[2].set_ylabel('yaw estimation error [rad]')

        plt.show()

    if False:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.plot(xs_gt, ys_gt, lw=2, label='ground-truth trajectory')
        ax.plot(xs_obs, ys_obs, lw=0, marker='.', markersize=4, alpha=1., label='observed trajectory')
        ax.plot(mu_x, mu_y, lw=2, label='estimated trajectory', color='r')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid()
        plt.show()

    if True:
        from matplotlib.animation import FuncAnimation
        from matplotlib.animation import FuncAnimation, PillowWriter
        import cv2

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        inx = 0

        ax.plot(xs_gt[:inx], ys_gt[:inx], lw=2, color='k', label='gt')
        ax.plot(xs_obs[inx], ys_obs[inx], lw=0, marker='.', markersize=4, alpha=1., color='orange', label='measurement')
        ax.plot(mu_x[:inx], mu_y[:inx], lw=1, color='r', label='EKF')
        ax.plot(dr_x[:inx], dr_y[:inx], lw=1, color='g', label='DR')

        for x, y in zip(mu_x, mu_y):

            ax.plot(xs_gt[:inx], ys_gt[:inx], lw=2,color='k')
            ax.plot(xs_obs[inx], ys_obs[inx], lw=0, marker='.', markersize=4, alpha=1.,color='orange')
            ax.plot(mu_x[:inx], mu_y[:inx], lw=1, color='r')
            ax.plot(dr_x[:inx], dr_y[:inx], lw=1, color='g')

            confidence_ellipse(x, y, np.array(cov_results_xy[inx]), ax, edgecolor='blue')
            inx += 1

            if False:
                # windowed plot, i dont really like it
                ax.set_xlim(mu_x[inx]- 50, mu_x[inx] + 50)
                ax.set_ylim(mu_y[inx]- 50, mu_y[inx] + 50)

            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.grid()
            ax.legend()

            plt.pause(0.005)


if __name__ == "__main__":


    ##############
    #### 1.A #####
    ##############
    dataset = pykitti_dataset()
    gt_trajectory_lla = []          # [longitude(deg), latitude(deg), altitude(meter)] x N
    gt_yaws = []                    # [yaw_angle(rad),] x N
    gt_yaw_rates = []               # [vehicle_yaw_rate(rad/s),] x N
    gt_forward_velocities = []      # [vehicle_forward_velocity(m/s),] x N
    gt_east_velocities = []         # [gt_east_velocities(m/s),] x N
    gt_north_velocities = []        # [gt_north_velocities(m/s),] x N

    ##############
    #### 1.B #####
    ##############

    for oxts_data in dataset.oxts:
        packet = oxts_data.packet
        gt_trajectory_lla.append([
            packet.lon,
            packet.lat,
            packet.alt
        ])
        gt_yaws.append(packet.yaw)
        gt_yaw_rates.append(packet.wz)
        gt_forward_velocities.append(packet.vf)
        gt_east_velocities.append(packet.ve)
        gt_north_velocities.append(packet.vn)

    gt_trajectory_lla = np.array(gt_trajectory_lla).T
    gt_yaws = np.array(gt_yaws)
    gt_yaw_rates = np.array(gt_yaw_rates)
    gt_forward_velocities = np.array(gt_forward_velocities)
    gt_east_velocities = np.array(gt_east_velocities)
    gt_north_velocities = np.array(gt_north_velocities)

    timestamps = np.array(dataset.timestamps)
    elapsed = np.array(timestamps) - timestamps[0]
    ts = [t.total_seconds() for t in elapsed]
    deltas = np.array(ts)[1:] - np.array(ts)[:-1]

    ##############
    #### 1.C.a ###
    ##############

    lons, lats, _ = gt_trajectory_lla

    if True:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(lons, lats)
        ax.set_xlabel('longitude [deg]')
        ax.set_ylabel('latitude [deg]')
        ax.grid()
        plt.show()

    ##############
    #### 1.C.b ###
    ##############
    origin = gt_trajectory_lla[:, 0]  # set the initial position to the origin
    gt_trajectory_xyz = lla_to_enu(gt_trajectory_lla, origin)

    if True:
        xs, ys, _ = gt_trajectory_xyz
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(xs, ys)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid()
        plt.show()


    ##############
    #### 1.D #####
    ##############
    N = len(ts)  # number of data point

    xy_obs_noise_std = 3.0  # standard deviation of observation noise of x and y in meter

    xy_obs_noise = np.random.normal(0.0, xy_obs_noise_std, (2, N))  # gen gaussian noise
    obs_trajectory_xyz = gt_trajectory_xyz.copy()
    obs_trajectory_xyz[:2, :] += xy_obs_noise[:, :]  # add the noise to ground-truth positions
    xs, ys, _ = gt_trajectory_xyz
    xs_w_noise, ys_w_noise, _ = obs_trajectory_xyz

    if True:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.plot(xs, ys, lw=2, label='ground-truth trajectory')
        ax.plot(xs_w_noise, ys_w_noise, lw=0, marker='.', markersize=5, alpha=0.4, label='observed trajectory')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid()
        plt.show()



    ##############
    #### 2. #####
    ##############

    forward_velocity_noise_std = 2. # standard deviation of forward velocity in m/s
    forward_velocity_noise = np.random.normal(0.0, forward_velocity_noise_std, (N,))  # gen gaussian noise

    # Add noise to yaw rates
    yaw_rate_noise_std = 0.2  # standard deviation of yaw rate in rad/s

    yaw_rate_noise = np.random.normal(0.0, yaw_rate_noise_std, (N,))  # gen gaussian noise
    obs_yaw_rates = gt_yaw_rates.copy()

    obs_yaw_rates += yaw_rate_noise  # add the noise to ground-truth positions

    if False:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        ax.plot(ts, gt_yaw_rates, lw=1, label='ground-truth')
        ax.plot(ts, obs_yaw_rates, lw=0, marker='.', alpha=0.4, label='observed')

        ax.set_xlabel('time elapsed [sec]')
        ax.set_ylabel('yaw rate [rad/s]')
        ax.legend()
        plt.show()

    obs_forward_velocities = gt_forward_velocities.copy()
    obs_forward_velocities += forward_velocity_noise  # add the noise to ground-truth positions

    if False:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        ax.plot(ts, gt_forward_velocities, lw=1, label='ground-truth')
        ax.plot(ts, obs_forward_velocities, lw=0, marker='.', alpha=0.4, label='observed')

        ax.set_xlabel('time elapsed [sec]')
        ax.set_ylabel('forward velocity [m/s]')
        ax.legend()
        plt.show()


    ##############
    #### 1.E #####
    ##############


    KF(gt_trajectory_xyz,
       obs_trajectory_xyz,
       xy_obs_noise_std,
       N,
       ts,
       obs_forward_velocities,
       obs_yaw_rates,
       gt_yaws,
       gt_east_velocities,
       gt_north_velocities)

    ##############
    #### 1.G #####
    ##############

    # KF_constant_acc(gt_trajectory_xyz,
    #    obs_trajectory_xyz,
    #    xy_obs_noise_std,
    #    N,
    #    ts,
    #    obs_forward_velocities,
    #    obs_yaw_rates,
    #    gt_yaws,
    #    gt_east_velocities,
    #    gt_north_velocities)

    ##############
    #### 2.E #####
    ##############

    # EKF(obs_trajectory_xyz,
    #     obs_forward_velocities,
    #     xy_obs_noise_std,
    #     forward_velocity_noise_std,
    #     obs_yaw_rates,
    #     yaw_rate_noise_std,
    #     N,
    #     ts)