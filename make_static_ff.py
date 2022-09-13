import pickle
import os
from lib.model import CANNet2s
from lib.utils import save_checkpoint, fix_model_state_dict
from lib.plot import plot_staticflow

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.nn.functional as F

import numpy as np
import argparse
import json
import cv2
from lib import dataset
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage.filters import gaussian_filter

parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')
parser.add_argument('--dataset', default="FDST")
parser.add_argument('--load_model', default="checkpoint.pth.tar")
parser.add_argument('--activate', default="leaky")

dloss_on = True


def main():
    global args, best_prec1

    best_prec1 = 200
    args = parser.parse_args()
    args.lr = 1e-4
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 1e-3
    args.start_epoch = 0
    args.epochs = 200
    args.workers = 8
    args.seed = int(time.time())
    args.print_freq = 10
    args.pretrained = True

    if args.dataset == "FDST":
        with open(args.val_json, 'r') as outfile:
            val_list = json.load(outfile)
    else:
        val_list = args.val_json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(args.seed)

    model = CANNet2s(load_weights=False, activate=args.activate)
    if args.pretrained:
        checkpoint = torch.load(str(args.load_model))
        model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
        try:
            best_prec1 = checkpoint['val']
        except KeyError:
            print("No Key: val")

    if torch.cuda.device_count() > 1:
        print("You can use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = nn.MSELoss(size_average=False)

    staticff = validate(val_list, model, criterion, device)

    picklefile = os.path.join(os.path.dirname(args.load_model), "staticff.pickle")
    with open(picklefile, "wb") as f:
        pickle.dump(staticff, f)


def validate(val_list, model, criterion, device):
    global args

    print('begin val')
    val_dataset = dataset.get_test_dataset(args.dataset, val_list)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    model.eval()

    whole_target_num = []
    for i, (prev_img, img, post_img, target) in enumerate(val_loader):
        target_num = target.detach().numpy()
        target_num_gauss = gaussian_filter(target_num, 3)
        whole_target_num.append(target_num_gauss)

    target_ave = np.sum(np.concatenate(whole_target_num), axis=0)
    target_ave[target_ave>1] = 1.0
    static_k = 1
    # staticff = np.exp(static_k*target_ave)
    staticff = static_k*target_ave

    input_num = prev_img[0, :, :, :].detach().cpu().numpy()
    input_num = input_num.transpose((1, 2, 0))
    input_num = input_num * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    plot_filename = os.path.join(os.path.dirname(args.load_model), "staticff.png")
    plot_staticflow(input_num, staticff, plot_filename)

    return staticff

if __name__ == "__main__":
    main()
