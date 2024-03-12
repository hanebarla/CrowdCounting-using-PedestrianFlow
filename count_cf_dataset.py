import os
import numpy as np
from scipy import ndimage
from tqdm import trange
import cv2


def make_npz():
    mode = "add"
    root = "/groups1/gca50095/aca10350zi/TUBCrowdFlow/"
    frame_num_list = [300, 300, 250, 300, 450]

    FFdataset_path = os.path.join(root, "ff_{}".format(mode))
    if not os.path.isdir(FFdataset_path):
        os.mkdir(FFdataset_path)

    for i in range(5):
        tmp_nums = []
        scene_dir = "IM0{}".format(i+1)
        print(scene_dir)
        if not os.path.isdir(os.path.join(FFdataset_path, scene_dir)):
            os.makedirs(os.path.join(FFdataset_path, scene_dir), exist_ok=True)

        frame = frame_num_list[i]

        scene_pos = np.loadtxt(
            os.path.join(
                root,
                "gt_trajectories",
                scene_dir,
                "personTrajectories.csv"),
            delimiter=",")

        staticff_label = np.zeros((720, 1280))
        staticff_label_360x640 = np.zeros((360, 640))
        staticff_label_180x320 = np.zeros((180, 320))
        staticff_label_90x160 = np.zeros((90, 160))
        staticff_label_45x80 = np.zeros((45, 80))
        for fra in trange(frame):
            # x = np.load(os.path.join(FFdataset_path, scene_dir, "{}_45x80.npz".format(fra)))["x"]
            x = cv2.imread(os.path.join(root ,"gt_trajectories", scene_dir, "PersonTrajectories", "PersonTrajectories_frame_{:04}.png".format(fra)), 0)
            x = np.array(x, dtype="float32")
            # print(x.max())
            x /= np.max(x)
            x = cv2.resize(x, (80, 45), interpolation=cv2.INTER_CUBIC)
            tmp_nums.append(x.sum())
            # raise ValueError
            # np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_360x640.npz".format(fra)), x=anno_input_same)
            # np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_180x320.npz".format(fra)), x=anno_input_half)
            # np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_90x160.npz".format(fra)), x=anno_input_quarter)
            # np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_45x80.npz".format(fra)), x=anno_input_eighth)

        np_tmp_nums = np.array(tmp_nums)
        print("Scene {}, MIN: {}, MAX: {}, AVE: {:.1f}, STD: {:.1f}".format(i+1, int(np_tmp_nums.min()), int(np_tmp_nums.max()), np.mean(np_tmp_nums), np.std(np_tmp_nums)))

if __name__ == "__main__":
    make_npz()
