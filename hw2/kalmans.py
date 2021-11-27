import matplotlib.pyplot as plt
from math import sin, cos, pi,tan, atan2,log
import numpy as np
from scipy import linalg
import time
from scipy.spatial import cKDTree as KDTree
import sympy as sym

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    trans_iimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %trans_iimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def KalmanFilter(mu, P, z, Q, delta, sig_a, model='constant_vel'):

    '''
    Parameters:
    mu: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    z: observed position (same shape as C*mu)
    Q: z noise (same shape as C)
    R: motion noise (same shape as P)
    A: next state function: mu_next = A*mu
    C: z function: position = C*mu

    Return: the updated and predicted new values for (mu, P)
    '''

    # UPDATE mu, P based on z m
    # distance between measured and current position-belief
    if model == 'constant_vel':
        R = np.matrix([[0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., delta, 0.],
                       [0., 0., 0., delta]]) * sig_a ** 2


        C = np.matrix([[1., 0., 0., 0.],
                       [0., 1., 0., 0.]])

        A = np.matrix([[1., 0., delta, 0.],
                       [0., 1., 0., delta],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])

    if model == 'constant_acc':
        R = np.matrix([[0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., delta, 0.],
                       [0., 0., 0., 0., 0., delta]]) * sig_a ** 2

        B = np.matrix('0. 0. 0. 0. 0. 0.').T

        C = np.matrix([[1., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0.]])

        A = np.matrix([[1., 0., delta, 0., 0.5*delta**2, 0.],
                       [0., 1., 0., delta, 0., 0.5*delta**2],
                       [0., 0., 1., 0., delta, 0.],
                       [0., 0., 0., 1., 0. , delta],
                       [0., 0., 0., 0., 1. , 0.],
                       [0., 0., 0., 0., 0. , 1.]])

    # predict mu, P based on motion
    mu = A * mu
    P = A * P * A.T + R

    # get Kalman gain
    S = C * P * C.T + Q  # residual convariance
    K = P * C.T * S.I    # Kalman gain
    y = np.matrix(z).T - C * mu

    # Apply correction
    mu = mu + K*y
    I = np.matrix(np.eye(A.shape[0]))
    P = (I - K*C)*P

    return mu, P


class ExtendedKalmanFilter:
    """Kalman Filter

    """
    def __init__(self, mu, P):
        """
        Args:
            mu (numpy.array): state to estimate: [x_, y_, theta]^T
            P (numpy.array): estimation error covariance
        """
        self.mu = mu  #  [3,]
        self.P = P  #  [3, 3]

    def update(self, z, Q):
        """update mu and P based on observation of (x_, y_)
        Args:
            z (numpy.array): obsrervation for [x_, y_]^T
            Q (numpy.array): observation noise covariance
        """
        # compute Kalman gain
        H = np.array([
            [1., 0., 0.],
            [0., 1., 0.]
        ])  # Jacobian of observation function

        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + Q)

        # update state x
        x, y, theta = self.mu
        z_ = np.array([x, y])  # expected observation from the estimated state
        self.mu = self.mu + K @ (z - z_)

        # update covariance P
        self.P = self.P - K @ H @ self.P

    def propagate(self, u, dt, R):
        """propagate x and P based on state transition model defined as eq. (5.9) in [1]
        Args:
            u (numpy.array): control input: [v, omega]^T
            dt (float): time interval in second
            R (numpy.array): state transition noise covariance
        """
        # propagate state x
        x, y, theta = self.mu
        v, omega = u
        r = v / omega  # turning radius

        dtheta = omega * dt
        dx = - r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = + r * np.cos(theta) - r * np.cos(theta + dtheta)

        self.mu += np.array([dx, dy, dtheta])

        V = np.array(
            [[(-1/omega) * np.sin(theta) + (1/omega)* np.sin(theta + dtheta),
             (v/omega**2) * np.sin(theta) - (v/omega**2)* np.sin(theta + dtheta) +
              r * np.cos(theta + dtheta)*dt],
            [(1 / omega) * np.cos(theta) - (1 / omega) * np.cos(theta + dtheta),
             (-v / omega ** 2) * np.cos(theta) + (v / omega ** 2) * np.cos(theta + dtheta) +
             r * np.sin(theta + dtheta)*dt],
            [0, dt]])


        G = np.array([
            [1., 0., - r * np.cos(theta) + r * np.cos(theta + dtheta)],
            [0., 1., - r * np.sin(theta) + r * np.sin(theta + dtheta)],
            [0., 0., 1.]
        ])  # Jacobian of state transition function

        self.P = G @ self.P @ G.T + V @ R @ V.T