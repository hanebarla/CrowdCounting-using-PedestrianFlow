import os
import h5py
import cv2
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage.filters import gaussian_filter
from lib.plot import plot_staticflow
import json
import numpy as np

dirs = []

for i in range(100):
    if (i+1) % 5 == 4 or (i+1) % 5 == 0:
        dirs.append(i+1)

root = "/groups1/gca50095/aca10350zi/FDST/our_dataset/test_data"

output_path_dict = {}
for d in dirs:
    scene_dir = os.path.join(root, str(d))
    print(scene_dir)

    tmp_h5s = [os.path.join(scene_dir,f) for f in os.listdir(scene_dir) if ".h5" in f]
    tmp_h5s.sort()
    print(tmp_h5s)
    # raise ValueError
    output_path_dict[d] = tmp_h5s

    staticff = None
    for n, h5file in enumerate(tmp_h5s):
        print('\r{}'.format(n), end='')
        # print(h5file)
        gt_file = h5py.File(h5file)
        target = np.asarray(gt_file['density'])
        target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
        target = gaussian_filter(target, 3)
        # print(target.max())
        gt_file.close()

        if staticff is None:
            staticff = target
        else:
            staticff += target
        # raise ValueError

    staticff[staticff>1] = 1.0
    staticff_path = os.path.join(scene_dir, "staticff_test.pickle")
    with open(staticff_path, "wb") as f:
        pickle.dump(staticff, f)

    fig = cv2.imread(tmp_h5s[0].replace("_resize.h5", ".jpg"))
    input_num = cv2.cvtColor(fig, cv2.COLOR_BGR2RGB)
    plot_filename = os.path.join(scene_dir, "staticff.png")
    plot_staticflow(input_num, staticff, plot_filename)

with open("FDST_ff_test.json", 'w') as f:
    json.dump(output_path_dict, f)

