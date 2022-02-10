import json
from make_csv import main
from os.path import join
import os
import random


def del_exp(filen):
    return filen[:-4]


if __name__ == '__main__':
    root = "/home/data/venice/"
    gt_d = "ground-truth"
    img_d = "images"
    train_all_path = []
    test_all_path = []

    train_img_d = join(root, "train_data", img_d)
    train_gt_d = join(root, "train_data", gt_d)
    test_img_d = join(root, "test_data", img_d)
    test_gt_d = join(root, "test_data", gt_d)

    train_img_files = sorted(os.listdir(train_img_d))
    train_gt_files = sorted(os.listdir(train_gt_d))
    test_img_files = sorted(os.listdir(test_img_d))
    test_gt_files = sorted(os.listdir(test_gt_d))
    print(list(map(del_exp, train_img_files)) == list(map(del_exp, train_gt_files)))
    print(list(map(del_exp, test_img_files)) == list(map(del_exp, test_gt_files)))

    for i in range(len(train_img_files) - 2):
        train_pathes = {}
        train_pathes["prev"] = join(train_img_d, train_img_files[i])
        train_pathes["now"] = join(train_img_d, train_img_files[i+1])
        train_pathes["next"] = join(train_img_d, train_img_files[i+2])
        train_pathes["target"] = join(train_gt_d, train_gt_files[i+1])
        train_all_path.append(train_pathes)

    with open("venice_train.json", "w") as f:
        json.dump(train_all_path, f)

    test_img_4895 = []
    test_gt_4895 = []
    test_img_4898 = []
    test_gt_4898 = []
    test_img_4901 = []
    test_gt_4901 = []

    for i, p in enumerate(test_img_files):
        if "4895" in p:
            test_img_4895.append(p)
            test_gt_4895.append(test_gt_files[i])
        elif "4898" in p:
            test_img_4898.append(p)
            test_gt_4898.append(test_gt_files[i])
        else:
            test_img_4901.append(p)
            test_gt_4901.append(test_gt_files[i])
    print(len(test_gt_files) == (len(test_img_4895) + len(test_img_4898) + len(test_img_4901)))

    for i in range(len(test_img_4895) - 2):
        test_pathes = {}
        test_pathes["prev"] = join(test_img_d, test_img_4895[i])
        test_pathes["now"] = join(test_img_d, test_img_4895[i+1])
        test_pathes["next"] = join(test_img_d, test_img_4895[i+2])
        test_pathes["target"] = join(test_gt_d, test_gt_4895[i+1])
        test_all_path.append(test_pathes)

    for i in range(len(test_img_4898) - 2):
        test_pathes = {}
        test_pathes["prev"] = join(test_img_d, test_img_4898[i])
        test_pathes["now"] = join(test_img_d, test_img_4898[i+1])
        test_pathes["next"] = join(test_img_d, test_img_4898[i+2])
        test_pathes["target"] = join(test_gt_d, test_gt_4898[i+1])
        test_all_path.append(test_pathes)

    for i in range(len(test_img_4901) - 2):
        test_pathes = {}
        test_pathes["prev"] = join(test_img_d, test_img_4901[i])
        test_pathes["now"] = join(test_img_d, test_img_4901[i+1])
        test_pathes["next"] = join(test_img_d, test_img_4901[i+2])
        test_pathes["target"] = join(test_gt_d, test_gt_4901[i+1])
        test_all_path.append(test_pathes)

    with open("venice_test.json", "w") as f:
        json.dump(test_all_path, f)
