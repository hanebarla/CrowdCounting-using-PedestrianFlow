import os
import numpy as np
from scipy import ndimage
from tqdm import trange


def person_annotate_img_generator(person_pos, frame, mode="add"):
    """
    Gaussian Spot Annotation
    ------
        (txtfile, image) --> (annotated image)
    """
    anno = np.zeros((720, 1280))
    anno_input_same = np.zeros((360, 640))
    anno_input_half = np.zeros((180, 320))
    anno_input_quarter = np.zeros((90, 160))
    anno_input_eighth = np.zeros((45, 80))

    frame_per_pos = np.array(
        person_pos[:, 2 * frame: 2 * (frame + 1)], dtype=np.uint32)
    shape = frame_per_pos.shape

    for i in range(shape[0]):
        pos = frame_per_pos[i, :]
        if pos[0] == 0 and pos[1] == 0:
            continue
        elif pos[0] >= 720 or pos[1] >= 1280:
            continue
        elif pos[0] < 0 or pos[1] < 0:
            continue
        anno[pos[0], pos[1]] = 1.0

        if mode == "add":
            anno_input_same[int(pos[0]/2), int(pos[1]/2)] += 1.0
            anno_input_half[int(pos[0]/4), int(pos[1]/4)] += 1.0
            anno_input_quarter[int(pos[0]/8), int(pos[1]/8)] += 1.0
            anno_input_eighth[int(pos[0]/16), int(pos[1]/16)] += 1.0
        elif mode == "once":
            anno_input_same[int(pos[0]/2), int(pos[1]/2)] = 1.0
            anno_input_half[int(pos[0]/4), int(pos[1]/4)] = 1.0
            anno_input_quarter[int(pos[0]/8), int(pos[1]/8)] = 1.0
            anno_input_eighth[int(pos[0]/16), int(pos[1]/16)] = 1.0
        else:
            raise ValueError

    anno = ndimage.filters.gaussian_filter(anno, 3)
    anno_input_same = ndimage.filters.gaussian_filter(anno_input_same, 3)
    anno_input_half = ndimage.filters.gaussian_filter(anno_input_half, 3)
    anno_input_quarter = ndimage.filters.gaussian_filter(anno_input_quarter, 3)
    anno_input_eighth = ndimage.filters.gaussian_filter(anno_input_eighth, 3)

    return anno, anno_input_same, anno_input_half, anno_input_quarter, anno_input_eighth


def make_npz():
    mode = "once"
    root = "/groups1/gca50095/aca10350zi/TUBCrowdFlow"
    frame_num_list = [300, 300, 250, 300, 450]

    FFdataset_path = os.path.join(root, "ff_{}".format(mode))
    if not os.path.isdir(FFdataset_path):
        os.mkdir(FFdataset_path)

    for i in range(5):
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
            anno, anno_input_same, anno_input_half, anno_input_quarter, anno_input_eighth = person_annotate_img_generator(scene_pos, fra, mode=mode)
            staticff_label += anno
            staticff_label_360x640 += anno_input_same
            staticff_label_180x320 += anno_input_half
            staticff_label_90x160 += anno_input_quarter
            staticff_label_45x80 += anno_input_eighth
            np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_720x1280.npz".format(fra)), x=anno)
            np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_360x640.npz".format(fra)), x=anno_input_same)
            np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_180x320.npz".format(fra)), x=anno_input_half)
            np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_90x160.npz".format(fra)), x=anno_input_quarter)
            np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "{}_45x80.npz".format(fra)), x=anno_input_eighth)

        staticff_label[staticff_label>1] = 1.0
        staticff_label_360x640[staticff_label_360x640>1] = 1.0
        staticff_label_180x320[staticff_label_180x320>1] = 1.0
        staticff_label_90x160[staticff_label_90x160>1] = 1.0
        staticff_label_45x80[staticff_label_45x80>1] = 1.0

        np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "staticff_720x1280.npz"), x=staticff_label)
        np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "staticff_360x640.npz"), x=staticff_label_360x640)
        np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "staticff_180x320.npz"), x=staticff_label_180x320)
        np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "staticff_90x160.npz"), x=staticff_label_90x160)
        np.savez_compressed(os.path.join(FFdataset_path, scene_dir, "staticff_45x80.npz"), x=staticff_label_45x80)


if __name__ == "__main__":
    make_npz()
