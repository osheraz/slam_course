import itertools
import numpy as np
import os
import pykitti
from occupancy_grid import *
import math


def latToScale(lat):
    return math.cos(lat * math.pi / 180.0)

def latlonToMercator(lat,lon,scale):
    er = 6378137
    mx = scale * lon * math.pi * er / 180
    my = scale * er * math.log(math.tan((90 + lat) * math.pi / 360))
    return mx,my

def convertOxtsToPose(oxts):
    # lat:   latitude of the oxts-unit (deg)
    # lon:   longitude of the oxts-unit (deg)
    # alt:   altitude of the oxts-unit (m)
    scale = latToScale(oxts[1].packet.lat)
    pose = []
    Tr_0_inv = []

    for o in oxts:
        mx,my = latlonToMercator(o.packet.lat, o.packet.lon,scale)
        mz = o.packet.alt
        t = np.array([mx,my,mz])
        rx = o.packet.roll
        ry = o.packet.pitch
        rz = o.packet.yaw
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0],[math.sin(rz), math.cos(rz), 0], [0, 0 ,1]])
        R  = np.dot(np.dot(Rz,Ry),Rx)

        m = np.c_[R, t.T]
        m = np.vstack((m, np.array([0, 0, 0, 1]).T))

        if len(Tr_0_inv)==0:
            Tr_0_inv = np.linalg.inv(m)

        pose.append(np.dot(Tr_0_inv,m))

    return pose

def pykitti_dataset():
    # Change this to the directory where you store KITTI data
    curr_dir_path = os.getcwd()
    basedir = curr_dir_path + '/kitti_data'

    # Specify the dataset to load
    date = '2011_09_26'
    drive = '0013'

    # Load the data.
    dataset = pykitti.raw(basedir, date, drive)

    return dataset

if __name__ == "__main__":

    ###################
    ### PARAMETERS
    ###################

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', '-gs', type=float, default=0.5)      # grid size
    parser.add_argument('--x_size', '-xs', type=int, default=100)           # width of the map
    parser.add_argument('--y_size', '-ys', type=int, default=100)           # height of the map
    parser.add_argument('--display', '-d', type=bool, default=True)         # display the map
    parser.add_argument('--pos_by_vel', '-v', type=bool, default=False)     # pose extraction by IMU
    parser.add_argument('--ICP', '-icp', type=bool, default=False)          # allow ICP correction
    parser.add_argument('--save_icp', '-sicp', type=bool, default=False)    # save icp during each iteration
    parser.add_argument('--interactive', '-i', type=bool, default=False)    # interactive map plot ( zooming on the car)
    parser.add_argument('--only_map', '-om', type=bool, default=False)       # whether to plot all the scene
    parser.add_argument('--mask', '-m', type=bool, default=False)            # display threshold map

    params = vars(parser.parse_args())

    point = np.array([0,0,0,1])
    pcl_list = []
    dataset = pykitti_dataset()
    velodyne_measurement = list(dataset.velo)  # Load Velodyne measurement
    T_imu_velo = dataset.calib.T_velo_imu  # Load the rigid transformation from IMU to velodyne


    # Using pykitti, dataset.oxts already convert LLY to ENU, so
    # I wrote another implementation using the Matlab Devkit ConvertOxtsToPose, in case of banning Pykitti. (NED)
    # v = [oxt for oxt in dataset.oxts]
    # pose_R = convertOxtsToPose(v)
    # pose_list = [np.dot(np.array(v), point) for v in pose_R]

    # normalize translation and rotation (start at 0/0/0)

    Tr_0_inv = np.linalg.inv(dataset.oxts[0].T_w_imu)
    pose_R = [np.dot(Tr_0_inv, o.T_w_imu) for o in dataset.oxts]
    pose_list = [np.dot(np.array(p), point) for p in pose_R]  # Get x,y,z in world coordinate
    lidar_pose = [np.dot(T_imu_velo,p) for p in pose_R]  # Get x,y,z of the velodyine in the world coordinate


    if params['pos_by_vel']:
        ###################
        ### POSE USING IMU
        ###################
        # Compute the pose by integrating the velocity in each time step
        time_stamps = list(dataset.timestamps)
        time_intervals = np.array([ [d.minute*60 + d.second + d.microsecond*(10**-6)] for d in time_stamps])
        delta_time = time_intervals[1:] - time_intervals[:-1]
        velocity_list = np.array([[o.packet.ve,o.packet.vn] for o in dataset.oxts]) # Ve, then Vn because of ENU
        pos_by_vel = np.array([v*dt for v, dt in zip(velocity_list,delta_time)])
        pos_by_vel = np.array([[np.sum(pos_by_vel[:i,0]), np.sum(pos_by_vel[:i,1]), 1] for i in range(pos_by_vel.shape[0])])
        pose_R = [o.T_w_imu for o in dataset.oxts]
        for i in range(len(pose_R)-1):
            pose_R[i][0,-1] = pos_by_vel[i,0]
            pose_R[i][1,-1] = pos_by_vel[i,1]
        pose_R = [np.dot(Tr_0_inv, p) for p in pose_R]
        pose_list = [np.dot(np.array(p), point) for p in pose_R]

    # Init the occupancy map and visulaizer
    map = Occupancy_Map(int(params['x_size']/params['grid_size']),
                        int(params['y_size']/params['grid_size']),
                        params['grid_size'])


    vis = Visualizer(dataset, map, np.array(pose_list)[:,:-2])


    # for each pose of the car
    for i in range(len(pose_list)):
        # Acquire the pose by IMU\GPS

        pose = pose_list[i][:-1]

        pcl = []
        # Convert pcl to the world coordinate
        for v in velodyne_measurement[i][::20]:
            p = np.dot(lidar_pose[i], v)
            pcl.append(p)

        pcl = map.get_relevant(np.array(pcl)[:, :-1], params['grid_size'])  # Sample interesting pixels
        pcl_list.append(pcl)  # Keep track in case of using ICP

        if params['ICP'] and i >= 1:
            ###################
            ### ICP
            ###################
            HTM, _ = ICP_based_SVD(np.array(pcl_list[i-1]), np.array(pcl), params['save_icp'])
            pose += np.dot(np.array(HTM), point)[:-1]

        map.update(pcl, pose)  # Perform update loop using inverse sensor model
        vis.update(map)  # Update the visualizer for plotting purposes

        if params['display']:
            if params['only_map']:
                vis.visualize_map(i,velodyne_measurement[i][::20], params)  # display
            else:
                vis.visualize(i,velodyne_measurement[i][::20], params)  # display




