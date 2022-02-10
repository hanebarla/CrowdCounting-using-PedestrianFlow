import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def OptFlow(c_prev_img, c_img):
    prev_img = cv2.cvtColor(c_prev_img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(c_img)
    hsv[..., 1] = 255

    flow = cv2.calcOpticalFlowFarneback(prev_img, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return hsv

if __name__ == "__main__":
    # json file contains the test images
    test_json_path = './test.json'

    # the folder to output density map and flow maps
    output_folder = './plot'

    with open(test_json_path, 'r') as outfile:
        img_paths = json.load(outfile)

    for i in range(2):
        img_path = img_paths[i]

        img_folder = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        index = int(img_name.split('.')[0])

        prev_index = int(max(1,index-5))
        prev_img_path = os.path.join(img_folder,'%03d.jpg'%(prev_index))

        c_img = cv2.imread(img_path)
        c_img = cv2.resize(c_img, (640, 360))

        c_prev_img = cv2.imread(prev_img_path)
        c_prev_img = cv2.resize(c_prev_img, (640, 360))

        hsv = OptFlow(c_prev_img, c_img)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        save_img = np.concatenate([c_prev_img, rgb], 0)

        cv2.imwrite('opticalflow/opticalflow_{}.png'.format(i), save_img)
