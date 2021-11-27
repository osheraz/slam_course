from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy
import scipy.stats


def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(sigma)
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def initialize_particles(num_particles):
    # initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        r = 10 * np.random.uniform(0, 1)
        particle['theta'] = 0
        angle = np.pi * np.random.uniform(0, 2)
        particle['x'] = r * np.cos(angle) + 5
        particle['y'] = r * np.sin(angle) + 5

        # initial weight
        particle['weight'] = 1.0 / num_particles

        # particle history aka all visited poses
        particle['history'] = []

        # add particle to set
        particles.append(particle)

    return particles

def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise

    gt_u = [odometry['r1'], odometry['t'], odometry['r2'] ]
    noise = [0.1, 0.2, 0.1]
    # noise = [0.01, 0.04, 0.01]

    for p in particles:

        obs_u = [gt_u[0] + np.random.normal(0, noise[0]),
                 gt_u[1] + np.random.normal(0, noise[1]),
                 gt_u[2] + np.random.normal(0, noise[2]), ]

        p['history'].append([p['x'], p['y'], p['theta']])
        p['x'] = p['x'] + obs_u[1] * np.cos(p['theta'] + obs_u[0])
        p['y'] = p['y'] + obs_u[1] * np.sin(p['theta'] + obs_u[0])
        p['theta'] = p['theta'] + obs_u[0] + obs_u[2]

def eval_sensor_model(gt, landmarks, particles):
    # Correct landmark poses with a measurement and
    # calculate particle weight
    Q_t = np.array([[1, 0], [0, 1]])

    # update landmarks and calculate weight for each particle
    for particle in particles:

        px = particle['x']
        py = particle['y']
        ptheta = particle['theta']

        # calculate expected range measurement
        for i in range(len(landmarks)):

            lx = landmarks[i + 1][0]
            ly = landmarks[i + 1][1]

            meas_range = np.sqrt((lx - gt[0]) ** 2 + (ly - gt[1]) ** 2)
            meas_bearing = math.atan2(ly - gt[1], lx - gt[0]) - gt[2]
            z = np.array([meas_range, meas_bearing])

            meas_range_exp = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)
            meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

            delta = np.array([meas_range_exp, meas_bearing_exp])

            weight = norm_pdf_multivariate(delta, z, Q_t)

            particle['weight'] = weight * particle['weight']

    # normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer
    pass

def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle
    # weights.

    new_particles = []
    step = 1.0/len(particles)
    rand = np.random.rand(1) * step
    c = particles[0]['weight']
    # index of weight container and corresponding particle
    i = 0
    #loop over all particle weights
    for m in range(len(particles)):

    #go through the weights until you find the particle to which the pointer points
        U = rand + m * step

        while U > c:
            i = i + 1
            c = c + particles[i]['weight']

        new_particle = copy.deepcopy(particles[i])
        # new_particle['weight'] = step
        new_particles.append(new_particle)

    return new_particles

def eval(gt,avg,best):
    # Evaluation
    gt = np.array(gt)
    avg = np.array(avg)
    best = np.array(best)

    e_x_avg = gt[1:,0] - avg[0,1:]
    e_y_avg =  gt[1:,1] - avg[1,1:]

    e_x_best = gt[1:,0] - best[0,1:]
    e_y_best =  gt[1:,1] - best[1,1:]

    maxE_avg = max(abs(e_x_avg) + abs(e_y_avg))
    maxE_best = max(abs(e_x_best) + abs(e_y_best))

    N = len(e_x_avg)
    RMSE_AVG = np.sqrt((1/N) * (e_x_avg ** 2 + e_y_avg ** 2).sum())
    RMSE_BEST = np.sqrt((1/N) * (e_x_best ** 2 + e_y_best ** 2).sum())

    print("maxE_avg: " + str(round(maxE_avg,3)) + "   RMSE_avg: " + str(round(RMSE_AVG,3)), end ="   ")
    print("maxE_best: " + str(round(maxE_best,3)) + "   RMSE_BEST: " + str(round(RMSE_BEST,3)))


def main():

    sensor_readings = read_sensor_data("odometry.dat")
    landmarks = read_world("landmarks_EX1.csv")

    # update gt using sample motion model
    gt = [[0, 0, 0], ]

    for i in range(len(sensor_readings)):
        odometry = sensor_readings[i, 'odometry']
        gt_u = [odometry['r1'], odometry['t'], odometry['r2']]

        gt.append([gt[-1][0] + gt_u[1] * np.cos(gt[-1][2] + gt_u[0]),  # x
                   gt[-1][1] + gt_u[1] * np.sin(gt[-1][2] + gt_u[0]),  # y
                   gt[-1][2] + gt_u[0] + gt_u[2]])  # theta

    # for i in range(len(landmarks)):
    #     plt.plot(landmarks[i+1][0],landmarks[i+1][1], 'bo')
    #
    # plt.plot(np.array(gt)[:,0], np.array(gt)[:,1], 'k-', label='avg')
    # plt.show()

    avg_list_x = []
    avg_list_y = []
    best_list_x = []
    best_list_y = []

    num_particles = 10

    # create particle set
    particles = initialize_particles(num_particles)

    # run PF
    for timestep in range(len(sensor_readings)):
        # predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep, 'odometry'], particles)

        # evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(gt[timestep], landmarks, particles)

        avg_path, best_path, gt_path = plot_state(particles,
                                                  landmarks,
                                                  gt[:timestep+1],
                                                  avg_list_x,
                                                  avg_list_y,
                                                  best_list_x,
                                                  best_list_y)

        # calculate new set of equally weighted particles
        particles = resample_particles(particles)

    eval(gt_path, avg_path, best_path)
    plt.savefig('wow')


if __name__ == "__main__":
    main()