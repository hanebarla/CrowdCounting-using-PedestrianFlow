import csv
import os
from lib.model import CANNet2s
import pickle
from lib.utils import save_checkpoint, fix_model_state_dict

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.nn.functional as F

import numpy as np
import argparse
import json
from lib import dataset
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage.filters import gaussian_filter

parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

parser.add_argument('val_json', metavar='VAL',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')
parser.add_argument('--dataset', default="FDST")
parser.add_argument('--load_model', default="checkpoint.pth.tar")
parser.add_argument('--activate', default="leaky")
parser.add_argument('--bn', default=0, type=int)
parser.add_argument('--do_rate', default=0.0, type=float)
parser.add_argument('--DynamicFF', default=0, type=int)
parser.add_argument('--StaticFF', default=0, type=int)

dloss_on = True

# Dynamic Floor Field
K_D = 0.5
BETA = 0.9
DELTA = 0.5


def dataset_factory(dlist, arguments, mode="train"):
    if arguments.dataset == "FDST":
        if mode == "train":
            return dataset.listDataset(
                dlist,
                shuffle=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ]),
                train=True,
                batch_size=args.batch_size,
                num_workers=args.workers
            )
        else:
            return dataset.listDataset(
                dlist,
                shuffle=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ]),
                train=False
            )
    elif arguments.dataset == "CrowdFlow":
        return dataset.CrowdDatasets(
            dlist,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            )
    elif arguments.dataset == "venice":
        return dataset.VeniceDataset(
            dlist,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        )
    elif arguments.dataset == "points":
        return dataset.PointsDataset(
            dlist,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )
    else:
        raise ValueError


def main():
    global args, best_prec1

    best_prec1 = 200

    args = parser.parse_args()
    args.lr = 1e-4
    args.batch_size = 1
    args.decay = 1e-3
    args.start_epoch = 0
    args.epochs = 200
    args.workers = 8
    args.seed = int(time.time())
    args.print_freq = 10
    args.pretrained = True
    savefolder = os.path.dirname(args.load_model)

    if args.StaticFF == 1 and args.DynamicFF == 1:
        savefilename = 'BothFF_result'
    elif args.StaticFF == 1:
        savefilename = 'StaticFF_result'
    elif args.DynamicFF == 1:
        savefilename = 'DynamicFF_result'
    else:
        savefilename = 'noFF_result'

    # choose dataset
    if args.dataset == "FDST":
        with open(args.val_json, 'r') as outfile:
            val_list = json.load(outfile)
        with open(args.test_json, 'r') as outfile:
            test_list = json.load(outfile)
    elif args.dataset == "CrowdFlow":
        val_list = args.val_json
        test_list = args.test_json
    elif args.dataset == "venice":
        val_list = args.val_json
        test_list = args.test_json
    elif args.dataset == "other":
        val_list = args.val_json
        test_list = args.test_json
    elif args.dataset == "points":
        val_list = None
        test_list = args.test_json
    else:
        raise ValueError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(args.seed)

    # load model
    if args.bn != 0 or args.do_rate > 0.0:
        load_weight = True
    else:
        load_weight = False
    model = CANNet2s(load_weights=load_weight, activate=args.activate, bn=args.bn, do_rate=args.do_rate)
    if args.pretrained:
        checkpoint = torch.load(str(args.load_model))
        model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
        try:
            best_prec1 = checkpoint['val']
        except KeyError:
            print("No Key: val")

    # multi gpu
    if torch.cuda.device_count() > 1:
        print("You can use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    staticff_file = os.path.join(os.path.dirname(args.load_model), "staticff.pickle")
    with open(staticff_file, mode="rb") as f:
        staticff_num = pickle.load(f)
    staticff = torch.from_numpy(staticff_num.astype(np.float32)).clone()
    staticff = staticff.to(device)

    torch.backends.cudnn.benchmark = True

    mae, rmse, pix_mae, pix_rmse = validate(val_list, model, staticff, device)
    print(' * best MAE {mae:.3f}, pix MAE {pix_mae:.5f} \n best RMSE {rsme:.3f}, pix RMSE {pix_rmse:.5f}'
          .format(mae=mae, pix_mae=pix_mae, rsme=rmse, pix_rmse=pix_rmse))
    with open(os.path.join(savefolder, '{}_val.csv'.format(savefilename)), mode='w') as f:
        writer = csv.writer(f)
        writer.writerow([mae, rmse, pix_mae, pix_rmse])

    mae, rmse, pix_mae, pix_rmse = validate(test_list, model, staticff, device, savefolder)
    print(' * best MAE {mae:.3f}, pix MAE {pix_mae:.5f} \n best RMSE {rsme:.3f}, pix RMSE {pix_rmse:.5f}'
          .format(mae=mae, pix_mae=pix_mae, rsme=rmse, pix_rmse=pix_rmse))
    with open(os.path.join(savefolder, '{}_test.csv'.format(savefilename)), mode='w') as f:
        writer = csv.writer(f)
        writer.writerow([mae, rmse, pix_mae, pix_rmse])


def validate(val_list, model, staticff, device, savefolder=None):
    global args
    print('begin val')
    val_dataset = dataset_factory(val_list, args, mode="val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1)

    model.eval()

    mae = 0
    rmse = 0
    pix_mae = []
    pix_rmse = []

    pred = []
    gt = []

    past_output = None

    for i, (prev_img, img, post_img, target) in enumerate(val_loader):
        # only use previous frame in inference time, as in real-time application scenario, future frame is not available
        prev_img = prev_img.to(device, dtype=torch.float)
        prev_img = Variable(prev_img)

        img = img.to(device, dtype=torch.float)
        img = Variable(img)

        with torch.no_grad():
            prev_flow = model(prev_img, img)
            prev_flow_inverse = model(img, prev_img)

        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0,:] = 1.0
        mask_boundry[-1,:] = 1.0
        mask_boundry[:,0] = 1.0
        mask_boundry[:,-1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())

        reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry

        overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).type(torch.FloatTensor)

        target = target.detach().numpy().copy()
        pred_sum = overall.detach().numpy().copy()

        if args.StaticFF == 1:
            prev_flow *= staticff
            prev_flow_inverse *= staticff

            reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
            reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry

            overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).type(torch.FloatTensor)

        if args.DynamicFF == 1 and past_output is not None:
            d_t_prev = gaussian_filter(past_output, 3)
            past_output = BETA * overall + (1 - DELTA) * d_t_prev
            # print("past output {}".format(torch.sum(past_output)))
            overall *= gaussian_filter(past_output, 3)

        if past_output is None:
            past_output = BETA * overall.detach().numpy().copy()

        pred_sum = overall.sum().detach().numpy().copy()
        if savefolder is not None:
            np.savez_compressed(os.path.join(savefolder, "output", "{}.npz".format(i)), x=pred_sum)

        pix_mae.append(mean_absolute_error(target.squeeze(), overall.detach().numpy().copy()))
        pix_rmse.append(np.sqrt(mean_squared_error(target.squeeze(), overall.detach().numpy().copy())))

        pred.append(pred_sum)
        gt.append(np.sum(target))


    #print("pred: {}".format(np.array(pred)))
    #print("target: {}".format(np.array(gt)))
    mae = mean_absolute_error(pred,gt)
    rmse = np.sqrt(mean_squared_error(pred,gt))
    pix_mae_val = np.mean(np.array(pix_mae))
    pix_rmse_val = np.mean(np.array(pix_rmse))

    return mae, rmse, pix_mae_val, pix_rmse_val

if __name__ == "__main__":
    main()
