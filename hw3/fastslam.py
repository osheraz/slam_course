from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy


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

def initialize_particles(num_particles, num_landmarks):
    # initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        # initialize pose: at the beginning, robot is certain it is at [0,0,0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        # initial weight
        particle['weight'] = 1.0 / num_particles

        # particle history aka all visited poses
        particle['history'] = []

        # initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            # initialize the landmark mean and covariance
            landmark['mu'] = [0, 0]
            landmark['sigma'] = np.zeros([2, 2])
            landmark['observed'] = False

            landmarks[i + 1] = landmark

        # add landmarks to particle
        particle['landmarks'] = landmarks

        # add particle to set
        particles.append(particle)

    return particles


def sample_motion_model(odometry, particles, gt):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise

    gt_u = [odometry['r1'], odometry['t'], odometry['r2'] ]
    noise = [0.01, 0.02, 0.01]

    for p in particles:

        obs_u = [gt_u[0] + np.random.normal(0, noise[0]),
                 gt_u[1] + np.random.normal(0, noise[1]),
                 gt_u[2] + np.random.normal(0, noise[2]), ]

        p['history'].append([p['x'], p['y'], p['theta']])
        p['x'] = p['x'] + obs_u[1] * np.cos(p['theta'] + obs_u[0])
        p['y'] = p['y'] + obs_u[1] * np.sin(p['theta'] + obs_u[0])
        p['theta'] = p['theta'] + obs_u[0] + obs_u[2]
        # ground truth
    gt.append([gt[-1][0] + gt_u[1] * np.cos(gt[-1][2] + gt_u[0]), # x
               gt[-1][1] + gt_u[1] * np.sin(gt[-1][2] + gt_u[0]), # y
               gt[-1][2] + gt_u[0] + gt_u[2]]) # theta

def measurement_model(particle, landmark):
    # Compute the expected measurement for a landmark
    # and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    # calculate expected range measurement
    meas_range_exp = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h
    # wrt the landmark location

    H = np.zeros((2, 2))
    H[0, 0] = (lx - px) / h[0]
    H[0, 1] = (ly - py) / h[0]
    H[1, 0] = (py - ly) / (h[0] ** 2)
    H[1, 1] = (lx - px) / (h[0] ** 2)

    return h, H


def eval_sensor_model(sensor_data, particles):
    # Correct landmark poses with a measurement and
    # calculate particle weight

    # sensor noise
    Q_t = np.array([[1.0, 0], [0, 0.1]])

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']
    # update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']
        px = particle['x']
        py = particle['y']
        ptheta = particle['theta']

        # loop over observed landmarks
        for i in range(len(ids)):

            # current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]

            # measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time

                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                landmark['mu'] = [px + meas_range * np.cos(ptheta + meas_bearing),
                                  py + meas_range * np.sin(ptheta + meas_bearing)]

                h, H = measurement_model(particle, landmark)
                landmark['sigma'] = np.linalg.inv(H) @ Q_t @ np.linalg.inv(H).T
                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above.
                # calculate particle weight: particle['weight'] = ...
                h, H = measurement_model(particle, landmark) # measurement prediction
                delta = np.array([meas_range - h[0], angle_diff(meas_bearing, h[1])])
                Q = H @ landmark['sigma'] @ H.T + Q_t # measurement covariance
                K = landmark['sigma'] @ H.T @ np.linalg.inv(Q) # calculate Kalman gain
                landmark['mu'] = landmark['mu'] + K @ delta # update mean
                landmark['sigma'] = (np.eye(np.shape(landmark['sigma'])[0]) - K @ H) @ landmark['sigma'] # update covariance
                weight = norm_pdf_multivariate(delta, np.array([0, 0]), Q)
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

    e_x_avg = gt[:,0] - avg[0,:]
    e_y_avg =  gt[:,1] - avg[1,:]

    e_x_best = gt[:-1,0] - best[0,:]
    e_y_best =  gt[:-1,1] - best[1,:]

    maxE_avg = max(abs(e_x_avg) + abs(e_y_avg))
    maxE_best = max(abs(e_x_best) + abs(e_y_best))

    N = len(e_x_avg)
    RMSE_AVG = np.sqrt((1/N) * (e_x_avg ** 2 + e_y_avg ** 2).sum())
    RMSE_BEST = np.sqrt((1/N) * (e_x_best ** 2 + e_y_best ** 2).sum())

    print("maxE_avg: " + str(round(maxE_avg,3)) + "   RMSE_avg: " + str(round(RMSE_AVG,3)), end ="   ")
    print("maxE_best: " + str(round(maxE_best,3)) + "   RMSE_BEST: " + str(round(RMSE_BEST,3)))


def main():
    landmarks = read_world("world.dat")
    sensor_readings = read_sensor_data("sensor_data.dat")
    avg_list_x = [0.0, ]
    avg_list_y = [0.0, ]
    best_list_x = [0.0, ]
    best_list_y = [0.0, ]

    num_particles = 100
    num_landmarks = len(landmarks)

    gt = [[0, 0, 0], ]
    # create particle set
    particles = initialize_particles(num_particles, num_landmarks)

    # run FastSLAM
    for timestep in range(int(len(sensor_readings) / 2)):
        # predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep, 'odometry'], particles, gt)

        # evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)


        avg_path, best_path, gt_path = plot_state(particles,
                                                  landmarks,
                                                  gt,
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