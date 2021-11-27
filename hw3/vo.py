import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join


class Dataset():
    def __init__(self, dataset_path, dataset_calib_path, dataset_pose_path):
        self.dataset_path = dataset_path
        self.dataset_calib_path = dataset_calib_path
        self.dataset_pose_path = dataset_pose_path

    def read_dataset(self):
        pics_list = [self.dataset_path + f for f in listdir(self.dataset_path) if isfile(join(self.dataset_path, f))]
        pics_list.sort()
        return pics_list

    def read_K_matrix(self):
        # camera calibration for intrinsic parameters
        file = open(self.dataset_calib_path, "r")
        lines = file.readlines()
        k = [lines[0].split(' ')][0][1:]
        k = [float(i) for i in k]
        k = np.array(k).reshape((3, 4))[:, :-1]
        file.close()
        return k

    def read_gt_trajectory(self):
        file = open(self.dataset_pose_path, "r")
        lines = file.readlines()
        x, y, z = [], [], []
        for i in lines:  # flatten transformation matrix
            x.append(i.split(' ')[3])
            y.append(i.split(' ')[11])
            z.append(i.split(' ')[7])
        file.close()
        return np.stack((x, y, z)).astype(np.float32)



def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


class VO():

    def __init__(self):
        # i dont really tried other methods, but we can pass them directly to the generator
        self.detection_type = ''
        self.detection_method = ''
        self.essential_extraction_method = ''


    def generate_features_to_track(self, f1):
        det = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        p1 = det.detect(f1)
        p1 = np.array([x.pt for x in p1], dtype=np.float32)
        return p1


    def track(self, image_ref, image_cur, px_ref):
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
        lk_params = dict(winSize=(21, 21),
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        # calculate optical flow
        p2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
        st = st.reshape(st.shape[0])
        # Select good points
        p1 = px_ref[st == 1]
        p2 = p2[st == 1]
        return p1, p2

    def calc_rot_and_tran(self, p2, p1 , k):
        # find the rotation and translation matrix using ransac
        E, mask = cv2.findEssentialMat(p2, p1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        pts, R, t, mask = cv2.recoverPose(E, p2, p1, k)
        return pts, R, t, mask

    def get_scale(self, gt_trajectory, vo_trajectory, frame_id, method):
        if method == 'class':
            magnitudes_gt = np.sqrt((gt_trajectory[0, :frame_id] ** 2 +
                                     gt_trajectory[1, :frame_id]).sum())

            magnitudes_vo = np.sqrt((vo_trajectory[0, :frame_id] ** 2
                                     + vo_trajectory[1, :frame_id] ** 2).sum())

            mag = np.median(magnitudes_gt / magnitudes_vo)
            if mag < 1000:
                scale_factor = mag
            else:
                scale_factor = 1  # for first iteration

        elif method == 'delta_gt':
            scale_factor = np.sqrt((gt_trajectory[0, frame_id] - gt_trajectory[0, frame_id - 1]) ** 2 +
                                   (gt_trajectory[1, frame_id] - gt_trajectory[1, frame_id - 1]) ** 2)

        else:
            scale_factor = 1

        return scale_factor

def main():
    import cv2
    # import imageio
    frames = []

    dataset_path = 'dataset/sequences/07/image_0/'
    dataset_pose_path = "dataset/poses/07.txt"
    dataset_calib_path = "dataset/sequences/07/calib.txt"

    dataset = Dataset(dataset_path, dataset_calib_path, dataset_pose_path)
    kMinNumFeature = 3000

    vo = VO()

    map = 255 * np.ones((500, 500, 3), dtype=np.uint8)
    error = []
    x_vo, y_vo = [], []

    pic_list = dataset.read_dataset()
    k = dataset.read_K_matrix()
    gt_trajectory = dataset.read_gt_trajectory()

    first_frame = cv2.imread(pic_list[0], 0)
    second_frame = cv2.imread(pic_list[1], 0)

    ## first frame
    p1 = vo.generate_features_to_track(first_frame)
    p1, p2 = vo.track(first_frame, second_frame, p1)
    pts, R, t, mask = vo.calc_rot_and_tran(p2, p1, k)
    p1 = p2
    cur_frame = second_frame

    for i in range(len(pic_list)):

        new_frame = cv2.imread(pic_list[i], 0)

        p1, p2 = vo.track(cur_frame, new_frame, p1)  # track feature movement
        pts, R_i, t_i, mask = vo.calc_rot_and_tran(p2, p1, k)

        # get scale - 2 methods
        if i > 2:
            # 'delta_gt' / 'class'
            scale = vo.get_scale(gt_trajectory, np.array([x_vo, y_vo]), i, 'delta_gt')
        else:
            scale = 1

        t += scale * R @ t_i
        R = R_i @ R

        # if num.features decreases
        if (p1.shape[0] < kMinNumFeature):
            p2 = vo.generate_features_to_track(new_frame)

        # update
        p1 = p2
        cur_frame = new_frame
        x, z, y = t[0][0], t[1][0], t[2][0]  # switch z-y because of the camera direction
        x_vo.append(x)
        y_vo.append(y)

        #####################
        ##      Draw       ##
        #####################

        cv2.circle(map, (int(x) + 300, int(y) + 200), 1, (0, 0, 0), 1)
        cv2.circle(map, (int(gt_trajectory[0, i]) + 300, int(gt_trajectory[1, i]) + 200), 1, (0, 255, 0), 1)

        for l in range(0, p1.shape[0], 10):
            cv2.circle(new_frame, (int(p1[l, 0]), int(p1[l, 1])), 5, (255, 255, 255))

        text1 = "VO: x=%.2fm y=%.2fm   (Black line)" % (x, y)
        text2 = "GT: x=%.2fm y=%.2fm   (Green line)" % (gt_trajectory[0, i], gt_trajectory[1, i])
        e = [np.sqrt((x - gt_trajectory[0, i]) ** 2), np.sqrt((y - gt_trajectory[1, i]) ** 2)]
        text3 = "Error:  x=%.2fm y=%.2fm" % (e[0], e[1])
        error.append(e)
        cv2.rectangle(map, (0, 400), (500, 500), (255, 255, 255), -1)
        cv2.putText(map, text1, (20, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 4)
        cv2.putText(map, text2, (20, 440), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 8)
        cv2.putText(map, text3, (20, 460), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 8)

        cv2.imshow('Camera', new_frame)
        cv2.imshow('Trajectory', map)
        # frames.append(np.array(map))

        # Close the frame
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # Release and Destroy
    cv2.destroyAllWindows()

    # print("Saving GIF file")
    # imageio.mimsave('movie.mp4', frames)

    e = np.mean(error)
    plt.plot(np.array(error)[:, 0], label="x - error ")
    plt.plot(np.array(error)[:, 1], label="y - error ")
    plt.grid()
    plt.show()
    print(' Mean error: ' + str(e))

    plt.figure(figsize=(8, 8), dpi=100)
    plt.title("Trajectory Comparison")
    plt.ylabel("X")
    plt.xlabel("Y")
    plt.plot(x_vo, y_vo, label="VO")
    plt.plot(gt_trajectory[0], gt_trajectory[1], label="GT")
    plt.legend()
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()


if __name__ == "__main__":
    main()