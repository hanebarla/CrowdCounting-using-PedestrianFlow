import os
import numpy as np
from scipy import ndimage
from tqdm import trange
import h5py
import csv
import cv2

def make_npz():
    mode = "add"
    root = "/groups1/gca50095/aca10350zi/FDST/our_dataset/test_data"
    frame_num_list = [150 for _ in range(40)]

    for i in range(40):
        tmp_nums = []
        scene_dir = str(int(i/2)*5+4) if i%2 == 0 else str(int((i+1)/2)*5)
        print(scene_dir)

        frame = frame_num_list[i]

        staticff_label = np.zeros((720, 1280))
        staticff_label_360x640 = np.zeros((360, 640))
        staticff_label_180x320 = np.zeros((180, 320))
        staticff_label_90x160 = np.zeros((90, 160))
        staticff_label_45x80 = np.zeros((45, 80))
        for fra in trange(frame):
            gt_path = os.path.join(root, scene_dir, "{:03}_resize_add.h5".format(fra+1))
            # print(gt_path)
            gt_file = h5py.File(gt_path)
            x = np.asarray(gt_file['density'])
            # x = cv2.resize(x, (80, 45), interpolation=cv2.INTER_CUBIC)
            # print(x.shape)
            gt_file.close()
            tmp_nums.append(x.sum())
            # raise ValueError
            # np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_360x640.npz".format(fra)), x=anno_input_same)
            # np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_180x320.npz".format(fra)), x=anno_input_half)
            # np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_90x160.npz".format(fra)), x=anno_input_quarter)
            # np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_45x80.npz".format(fra)), x=anno_input_eighth)

        np_tmp_nums = np.array(tmp_nums)
        print("Scene {}, MIN: {}, MAX: {}, AVE: {}, STD: {}".format(scene_dir, np_tmp_nums.min(), np_tmp_nums.max(), np.mean(np_tmp_nums), np.std(np_tmp_nums)))

        with open("fdst_data.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(["Scene{}".format(scene_dir), "150", "{}".format(int(np_tmp_nums.min())), "{}".format(int(np_tmp_nums.max())), "{:.1f}".format(np.mean(np_tmp_nums)), "{:.1f}".format(np.std(np_tmp_nums))])

if __name__ == "__main__":
    make_npz()
