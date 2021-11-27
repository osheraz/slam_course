
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def angle_diff(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

def error_ellipse(position, sigma):

    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    error_ellipse = Ellipse(xy=[position[0],position[1]], width=width, height=height, angle=angle/np.pi*180)
    error_ellipse.set_alpha(0.25)

    return error_ellipse


def plot_state(particles, landmarks, gt,avg_list_x,avg_list_y,best_list_x,best_list_y):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and 
    # estimated mean landmark positions and covariances.

    draw_mean_landmark_poses = False

    map_limits = [-7.5, 17.5, -7.5, 17.5]
    
    #particle positions
    xs = []
    ys = []

    #landmark mean positions
    lxs = []
    lys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])


    # ground truth landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # average over particles
    avg_list_x.append(np.mean(np.array(xs)))
    avg_list_y.append(np.mean(np.array(ys)))

    # best particle
    estimated = best_particle(particles)
    robot_x = estimated['x']
    robot_y = estimated['y']
    robot_theta = estimated['theta']

    # estimated traveled path of best particle at each iteration.
    best_list_x.append(robot_x)
    best_list_y.append(robot_y)

    # estimated traveled path of best particle
    hist = estimated['history']
    hx = []
    hy = []

    for pos in hist:
        hx.append(pos[0])
        hy.append(pos[1])

    # plot filter state
    plt.clf()

    #particles
    plt.plot(xs, ys, 'r.')

    if draw_mean_landmark_poses:
        # estimated mean landmark positions of each particle
        plt.plot(lxs, lys, 'b.')

    # estimated traveled path of best particle
    plt.plot(hx, hy, 'r-', label='current best-history')

    # estimated traveled path of average
    plt.plot(avg_list_x, avg_list_y, 'ko', label='avg', markersize=1)

    # estimated traveled path of average
    plt.plot(best_list_x, best_list_y, 'go', label='best-at-each-itr', markersize=1)

    # gt traveled path of best particle
    plt.plot(np.array(gt)[:,0], np.array(gt)[:,1], 'bo' , label='gt', markersize=1)

    # true landmark positions
    plt.plot(lx, ly, 'b+',markersize=10)


    # draw pose of best particle
    plt.quiver(robot_x, robot_y, np.cos(robot_theta), np.sin(robot_theta), angles='xy',scale_units='xy')
    plt.axis(map_limits)
    plt.axis('equal')
    plt.legend()
    plt.pause(0.000000001)
    # plt.show()

    return [avg_list_x, avg_list_y], [hx,hy] , gt

def best_particle(particles):
    #find particle with highest weight 

    highest_weight = 0

    best_particle = None
    
    for particle in particles:
        if particle['weight'] > highest_weight:
            best_particle = particle
            highest_weight = particle['weight']

    return best_particle
