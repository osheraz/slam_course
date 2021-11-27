import matplotlib.pyplot as plt
from math import sin, cos, pi,tan, atan2,log
import numpy as np
from scipy import linalg
import time
from scipy.spatial import cKDTree as KDTree

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


def ICP_based_SVD(cur, next, save=False):
    HTM = np.identity(4)
    rot = np.identity(3)
    trans = [0.0, 0.0, 0.0]
    kdtree = KDTree(cur)
    n = np.size(next, 0)
    err = 1000
    for i in range(100):
        last_error = err
        # Using tree search, get the nearest point for each point in the next pcl to the cur pcl
        d, idx = kdtree.query(next, k=1)
        # calc the norm of each point
        err = np.mean(d ** 2)
        # find the center of the next point
        mu_p = np.mean(next, 0)
        # find the corresponding center of mass
        x_i_sum = np.array([0.0, 0.0, 0.0])
        for j in range(np.size(idx, 0)):
            x_i_sum = np.add(x_i_sum, np.array(cur[idx[j]]), out=x_i_sum, casting='unsafe')
        mu_x = x_i_sum / np.size(idx, 0)

        # Apply singular value decomposition to find W
        T = np.asanyarray([cur[p] for p in idx])
        W = np.dot(np.transpose(next - mu_p), T - mu_x)
        U, _, V = np.linalg.svd(W, full_matrices=False)
        # Compute the rotation matrix R = UV.T
        rot_i = np.dot(V.T, U.T)
        # Compute the translation vector t = m_x -Rm_p
        trans_i = mu_x - np.dot(rot_i, mu_p)

        next = (rot_i.dot(next.T)).T # apply rotation
        next = np.add(next, trans_i) # add translation

        rot = np.dot(rot_i, rot) # update rot
        trans = np.add(np.dot(rot_i, trans), trans_i) # update trans
        if save:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(cur[0:, 0], cur[0:, 1], cur[0:, 2], c='r', marker='o', s=1)
            ax.scatter(next[0:, 0], next[0:, 1], next[0:, 2], c='b', marker='x', s=1)
            ax.set_ylabel('Y [m]')
            ax.set_xlabel('X [m]')
            ax.view_init(90.0, -180.0), plt.xlim([-10, 10]), plt.ylim([-10, 10]), ax.set_zticks([])
            filename = str(i) + 'plot.png'
            fig.savefig('icp/' + str(filename))
            plt.close(fig)
        # Check convergence
        if abs(last_error - err) < 0.0001:
            HTM[0:3, 0:3] = rot[0:, 0:]
            HTM[0:3, 3] = trans[:]
            break

    return HTM, next


class Occupancy_Map():
    def __init__(self, x_size, y_size, grid_size):
        
        self.x_size = x_size+2 
        self.y_size = y_size+2
        self.grid_size = grid_size
        self.res = int(1/self.grid_size)
        self.max_range = 30.0

        self.pose = np.zeros((2,1))
        self.LogOddMap = np.zeros((self.x_size, self.y_size))
        self.ThresholdMap = np.zeros((self.x_size, self.y_size))
        self.origin = np.ceil(np.array([self.x_size, self.y_size]) / 2)
        
        # Coordinates of all cells
        self.grid_indexes = np.array([np.tile(np.arange(0, self.x_size, 1)[:,None], (1, self.y_size)),
                                   np.tile(np.arange(0, self.y_size, 1)[:,None].T, (self.x_size, 1))])

        self.alpha = 0.2*self.res  # thickness of obstacles
        self.beta = 0.5*np.pi/180.0  # width of the laser beam
        self.height_diff = 0.3

        # Log Probabilities to add or remove from the map
        self.l_occ = self.logit(0.7)
        self.l_free = self.logit(0.4)
        self.prior = self.logit(0.5)
        self.thresh_occ = self.logit(0.6)
        self.thresh_free = self.logit(0.4)
        self.max_logodd = self.logit(0.95)
        self.min_logodd = self.logit(0.05)

    def logit(self,p):
        """
        simple logit implementation
        input p (1,)
        output logg odd value ld (1,)
        """
        return log(p/(1-p))

    def get_relevant(self,pcl, grid_size):
        """
        relevant pixel extraction
        (only one who is hit at least twice in vertical direction or mean pixel height is above the Threshold)

        inputs:
        pcl - (nx3) array, where n is the number of measurements
        grid_size (1,) - required grid size
        output relevant points from the pcl (nx3)
        """
        # Sort the point cloud
        non_empty_cell, indices, nb_pts_per_cell = np.unique(((pcl - np.min(pcl, axis=0)) // grid_size).
                                                          astype(int), axis=0, return_inverse=True,return_counts=True)
        idx_pts_vox_sorted = np.argsort(indices)  # Sort the indexes so we can loop over
        cell_grid = {}
        interest = []
        by_height = []
        last_seen = 0
        LIDAR_HEIGHT = 1.7

        for i, cell in enumerate(non_empty_cell):
            # loop over the non empty cells and append all the points that inside
            cell_grid[tuple(cell)] = pcl[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_cell[i]]]
            # Calc the mean
            mean = np.mean(cell_grid[tuple(cell)], axis=0)
            # Calc the distance w.r.t the origin
            r = np.sqrt(mean[0]**2 + mean[1]**2)
            # Check whether there is at least two hit in the same celll
            if len(cell_grid[tuple(cell)]) > 1 and r <= self.max_range:
                interest.append(mean)
            if mean[2] > -LIDAR_HEIGHT and r <= self.max_range:
                by_height.append(mean)
            last_seen += nb_pts_per_cell[i]

        return np.array(interest)

    def update(self, z, x):
        """
        implementation of a occupancy map update at each car position using the inverse sensor model
        inputs:
        z - (nx3) array, where n is the number of measurements
        x (1,3) - current car position
        """
        # coordinate of robot in metric map,
        self.pose = np.ceil([self.res * x[0] + int(self.origin[0])/2, self.res * x[1]+ int(self.origin[1])/2 ])
        # Get the coordinates of all cells
        grid_indexes = self.grid_indexes.copy()
        # A matrix of all the x coordinates of the cell
        grid_indexes[0, :, :] -= int(self.pose[0]) 
        grid_indexes[1, :, :] -= int(self.pose[1]) 
        # A matrix of all the angles w.r.t the robot pose
        grid_angles = np.arctan2(grid_indexes[1, :, :], grid_indexes[0, :, :])
        # Clip to +pi / - pi
        grid_angles[grid_angles > np.pi] = grid_angles[grid_angles > np.pi] - 2. * np.pi
        grid_angles[grid_angles < -np.pi] = grid_angles[grid_angles < -np.pi] + 2. * np.pi
        
        self.grid_distances = linalg.norm(grid_indexes, axis=0) # matrix of distance from all cells w.r.t the current robot pose

        tic()
        for i in range(len(z)):
            z_x = np.ceil(self.res * (z[i][0]))
            z_y = np.ceil(self.res * (z[i][1]))

            r = np.sqrt(z_x**2 + z_y**2)

            if r > self.max_range * self.res:
                continue

            b = np.arctan2(z_y, z_x)

            free_cond1 = (np.abs(grid_angles - b) <= self.beta/2.0)
            free_cond2 = (self.grid_distances < (r - self.alpha/2.0))
            free = np.logical_and(free_cond1, free_cond2)
            occ_cond1 = (np.abs(grid_angles - b) <= self.beta/2.0)
            occ_cond2 = (np.abs(self.grid_distances - r) <= self.alpha/2.0)
            occ = np.logical_and(occ_cond1, occ_cond2)

            self.LogOddMap[occ] += self.l_occ
            self.LogOddMap[free] += self.l_free

        # Clip to Saturation values
        self.LogOddMap = np.where(self.LogOddMap > self.max_logodd, self.max_logodd, self.LogOddMap)
        self.LogOddMap = np.where(self.LogOddMap < self.min_logodd, self.min_logodd, self.LogOddMap)

        # Generate the map
        self.ThresholdMap = np.where(self.LogOddMap > self.thresh_occ, 1, self.ThresholdMap)
        self.ThresholdMap = np.where(self.LogOddMap < self.thresh_free, 0, self.ThresholdMap)

        toc()

    def update_each_mi(self, z, x):
        """
        implementation of a occupancy map update at each car position using the inverse sensor model
        inputs:
        z - (nx3) array, where n is the number of measurements
        x (1,3) - current car position
        Note: Slow performance compared to update function above.
        """
        # coordinate of robot in metric map
        self.pose = np.ceil([self.res * x[0]+ int(self.origin[0]) , self.res * x[1]+ int(self.origin[0]) / 2])
        # tensor of coordinates of all cells
        grid_indexes = self.grid_indexes.copy()
        # A matrix of all the x coordinates of the cell
        grid_indexes[0, :, :] -= int(self.pose[0])
        grid_indexes[1, :, :] -= int(self.pose[1])
        # A matrix of all the angles w.r.t the robot pose
        grid_angles = np.arctan2(grid_indexes[1, :, :], grid_indexes[0, :, :])
        # Clip to +pi / - pi
        grid_angles[grid_angles > np.pi] = grid_angles[grid_angles > np.pi] - 2. * np.pi
        grid_angles[grid_angles < -np.pi] = grid_angles[grid_angles < -np.pi] + 2. * np.pi

        grid_distances = linalg.norm(grid_indexes, axis=0) # matrix of distance from all cells w.r.t the current robot pose


        for x in range(self.LogOddMap.shape[0]):
            for y in range(self.LogOddMap.shape[1]):

                r = grid_distances[x,y]

                if r > self.max_range * self.res:
                    continue

                z_x = np.ceil(self.res * (z[:,0]))
                z_y = np.ceil(self.res * (z[:,1]))
                r_z = np.sqrt(z_x**2 + z_y**2)
                b = np.arctan2(z_y, z_x)

                ind = np.argmin(np.abs(grid_angles[x,y] - b))
                r_ztk = r_z[ind]

                # Update only the relevant area, same as checking because of l0 = 0..
                if r_ztk > self.max_range * self.res:
                    continue

                if (r > r_ztk + self.alpha/2.0) or (np.abs(grid_angles[x,y] - b[ind]) >= self.beta/2.0):
                    self.LogOddMap[x,y] += self.prior
                elif abs(r - r_ztk) < self.alpha/2.0:
                    self.LogOddMap[x,y] +=self.l_occ
                elif r < r_ztk:
                    self.LogOddMap[x, y] += self.l_free


        # Clip to Saturation values
        self.LogOddMap = np.where(self.LogOddMap > self.max_logodd, self.max_logodd, self.LogOddMap)
        self.LogOddMap = np.where(self.LogOddMap < self.min_logodd, self.min_logodd, self.LogOddMap)

        # Generate the map
        self.ThresholdMap = np.where(self.LogOddMap > self.thresh_occ, 1, self.ThresholdMap)
        self.ThresholdMap = np.where(self.LogOddMap < self.thresh_free, 0, self.ThresholdMap)



class Visualizer():

    def __init__(self, dataset,occ_map, pose_list):
        # Simple Visualizer.

        self.dataset = dataset
        self.x_size = occ_map.x_size
        self.y_size = occ_map.y_size
        self.map = occ_map
        self.pose = occ_map.pose
        self.pics = list(dataset.cam2)
        self.fig = plt.figure(figsize=(8, 8))
        self.pose_list = pose_list
        import matplotlib.animation as animation

    def crop(self,image, x1, x2, y1, y2):
        """
        Return the cropped image at the x1, x2, y1, y2 coordinates
        """
        if x2 == -1:
            x2 = image.shape[1] - 1
        if y2 == -1:
            y2 = image.shape[0] - 1

        mask = np.zeros(image.shape)
        mask[y1:y2 + 1, x1:x2 + 1] = 1
        m = mask > 0

        return image[m].reshape((y2 + 1 - y1, x2 + 1 - x1))

    def visualize(self, i, pcl, params):
        # Plot
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        pcl = np.array(pcl)[:,:-1]
        self.fig.clf()

        ##### Camera view #####
        ax = self.fig.add_subplot(2, 1, 1)
        ax.imshow(self.pics[i])
        ax.grid(False)
        ax.xaxis.set_visible(False)  # Do not draw axis tick marks
        ax.yaxis.set_visible(False)  # Do not draw axis tick marks
        ax.set_title('Camera view')

        ##### PCL view #####
        ax = self.fig.add_subplot(223, projection='3d')
        f = ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2],c=plt.cm.jet(pcl[:, 2]/max(pcl[:, 2])), s=1)
        ax.view_init(90.0, -180.0), plt.xlim([-80, 80]), plt.ylim([-80, 80]),ax.set_zticks([])
        box = ax.get_position()
        ax.set_position([box.x0-0.09, box.y0-0.09, box.width * 1.5, box.height * 1.5])
        ax.set_ylabel('X [m]')
        ax.set_xlabel('Y [m]')

        ##### Occupancy map view #####
        ax = self.fig.add_subplot(2, 2, 4)
        extent = [-self.x_size / 2, self.x_size / 2, -self.y_size / 2, self.y_size / 2]
        # which map to plot
        if params['mask']:
            image = self.map.ThresholdMap
        else:
            image = self.map.LogOddMap

        # Interactive vs constant map

        if params['interactive']:
            # Same as bellow
            # x_ = np.arange(int(self.map.pose[0]) - int(self.x_size / 4), int(self.map.pose[0]) + int(self.x_size / 4))
            # y_ = np.arange(int(self.map.pose[1]) - int(self.y_size / 4), int(self.map.pose[1]) + int(self.y_size / 4))
            # xx, yy = np.meshgrid(x_, y_)
            x,y = np.where(self.map.grid_distances == 0)
            x1 = int(y) - int(self.x_size / 4)
            x2 = int(y) + int(self.x_size / 4)
            y1 = int(x) - int(self.y_size / 4)
            y2 = int(x) + int(self.y_size / 4)
            image = self.crop(image,x1,x2,y1,y2)

        ax.imshow(np.rot90(image), interpolation='none', cmap='binary', extent=extent)
        plt.xticks([]), plt.yticks([]),ax.set_title('Occupancy map')
        ax.set_xlabel('Map resolution '+ str(self.map.grid_size) + 'x'+ str(self.map.grid_size)+ '[m]\n'
                      +'Map size ' + str(int(self.map.x_size*self.map.grid_size)) + 'x'+ str(int(self.map.y_size*self.map.grid_size))+ '[m]')

        ##### Car trajectory map view #####
        # Cannot be together due to zooming
        if not params['interactive']:
            path = np.zeros((i,2))
            for j in range(i):
                path[j,:] = self.map.res*self.pose_list[j,:] - [int(self.map.origin[0])/2, int(self.map.origin[1])/2]

            ax.plot(path[:,0],path[:,1],'r')
            l = 4
            w = h = 1.5
            x = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
            y = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
            z = np.array([0,0,0,0,h,h,h,h])
            car = np.vstack((x,y,z, np.ones((1,len(x)))))
            R = np.dot(self.dataset.calib.T_velo_imu, car)
            transform_car = np.dot(np.dot(np.linalg.inv(self.dataset.oxts[0].T_w_imu),self.dataset.oxts[i].T_w_imu), R)
            for f in range(transform_car.shape[1]):
                transform_car[:2, f] = transform_car[:2, f] * self.map.res - [int(self.map.origin[0]) / 2,
                                                                              int(self.map.origin[1]) / 2]

            ax.plot([transform_car[0, 0], transform_car[0, 1]], [transform_car[1, 0], transform_car[1, 1]], 'k')
            ax.plot([transform_car[0, 0], transform_car[0, 3]], [transform_car[1, 0], transform_car[1, 3]], 'k')
            ax.plot([transform_car[0, 2], transform_car[0, 1]], [transform_car[1, 2], transform_car[1, 1]], 'k')
            ax.plot([transform_car[0, 2], transform_car[0, 3]], [transform_car[1, 2], transform_car[1, 3]], 'k')

        ##### Display\Save\Pause #####
        #
        # plt.savefig('logoddmapplot/' +str(i)+'.png')

        plt.pause(0.00005)

    def visualize_map(self, i, pcl, params):

        self.fig.clf()


        ##### Occupancy map view #####
        ax = self.fig.add_subplot(1, 1, 1)
        extent = [-self.x_size / 2, self.x_size / 2, -self.y_size / 2, self.y_size / 2]

        if params['mask']:
            image = self.map.ThresholdMap
        else:
            image = self.map.LogOddMap

        # Interactive vs constant map
        if params['interactive']:
            # same
            # x_ = np.arange(int(self.map.pose[0]) - int(self.x_size / 4), int(self.map.pose[0]) + int(self.x_size / 4))
            # y_ = np.arange(int(self.map.pose[1]) - int(self.y_size / 4), int(self.map.pose[1]) + int(self.y_size / 4))
            # xx, yy = np.meshgrid(x_, y_)
            x,y = np.where(self.map.grid_distances == 0)
            x1 = int(y) - int(self.x_size / 4)
            x2 = int(y) + int(self.x_size / 4)
            y1 = int(x) - int(self.y_size / 4)
            y2 = int(x) + int(self.y_size / 4)
            image = self.crop(image,x1,x2,y1,y2)

        ax.imshow(np.rot90(image), interpolation='none', cmap='binary',extent=extent)
        plt.xticks([]), plt.yticks([]),ax.set_title('Occupancy map')
        ax.set_xlabel('Map resolution '+ str(self.map.grid_size) + 'x'+ str(self.map.grid_size)+ '[m]\n'
                      +'Map size ' + str(int(self.map.x_size*self.map.grid_size)) + 'x'+ str(int(self.map.y_size*self.map.grid_size))+ '[m]')

        ##### Car trajectory map view #####
        if not params['interactive']:
            path = np.zeros((i,2))
            for j in range(i):
                path[j,:] = self.map.res*self.pose_list[j,:] - [int(self.map.origin[0])/2, int(self.map.origin[1])/2]
            ax.plot(path[:,0],path[:,1],'r')
            l = 4
            w = h = 1.5
            x = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
            y = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
            z = np.array([0,0,0,0,h,h,h,h])
            car = np.vstack((x,y,z, np.ones((1,len(x)))))
            R = np.dot(self.dataset.calib.T_velo_imu, car)
            transform_car = np.dot(np.dot(np.linalg.inv(self.dataset.oxts[0].T_w_imu),self.dataset.oxts[i].T_w_imu), R)
            for f in range(transform_car.shape[1]):
                transform_car[:2,f] = transform_car[:2,f] * self.map.res - [int(self.map.origin[0])/2, int(self.map.origin[1])/2]


            ax.plot([transform_car[0, 0], transform_car[0, 1]], [transform_car[1, 0], transform_car[1, 1]], 'k')
            ax.plot([transform_car[0, 0], transform_car[0, 3]], [transform_car[1, 0], transform_car[1, 3]], 'k')
            ax.plot([transform_car[0, 2], transform_car[0, 1]], [transform_car[1, 2], transform_car[1, 1]], 'k')
            ax.plot([transform_car[0, 2], transform_car[0, 3]], [transform_car[1, 2], transform_car[1, 3]], 'k')

        ##### Display\Save\Pause #####
        # if i==140:
            # plt.savefig('logoddmapplot/' +str(i)+'.png')
        plt.pause(0.00005)


    def update(self,occ_map):
        self.map = occ_map
        self.pose = occ_map.pose
