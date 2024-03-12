import csv
from genericpath import isfile
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
import matplotlib.pyplot as plt
import argparse
import json
from lib import dataset
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

parser.add_argument('val_json', metavar='VAL',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')
parser.add_argument('--dataset', default="FDST")
parser.add_argument('--data_mode', default='once')  # once or add
parser.add_argument('--load_model', default="checkpoint.pth.tar")
parser.add_argument('--activate', default="leaky")
parser.add_argument('--bn', default=0, type=int)
parser.add_argument('--do_rate', default=0.0, type=float)
parser.add_argument('--DynamicFF', default=0, type=int)
parser.add_argument('--StaticFF', default=0, type=int)
parser.add_argument('--pix', default=0, type=int)

dloss_on = True


def dataset_factory(dlist, arguments, mode="train", scene=None, add=False):
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
            ]),
            add=add
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
    elif arguments.dataset == "CityStreet":
         if mode == "train":
             return dataset.CityStreetDataset(
                 dlist,
                 data_type=arguments.data_mode,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                     )
                 ]),
                 scene=scene
             )
         else:
             return dataset.CityStreetDataset(
                 dlist,
                 data_type=arguments.data_mode,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                     )
                 ]),
                 scene=scene
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

    param_path = os.path.join(savefolder, "images", savefilename.replace("result", "Demo"), 'ff_param_pix.csv' if args.pix==1 else 'ff_param.csv')
    if os.path.isfile(param_path):
        print(param_path)
        with open(param_path) as f:
            reader = csv.reader(f)
            params = [row for row in reader]
            print(params)
            static_param, temperature_param, beta_param, delta_param = float(params[0][0]), float(params[0][1]), float(params[0][2]), float(params[0][3])
            # static_param, dynamic_param = float(params[0][0]), 20.0
    else:
        static_param, temperature_param, beta_param, delta_param = None, None, None, None

    result_per_data_path = os.path.join(savefolder, "images", savefilename.replace("result", "Demo"), 'res_pix.csv' if args.pix==1 else 'res.csv')
    if(os.path.isfile(result_per_data_path)):
        os.remove(result_per_data_path)

    # choose dataset
    scene = None
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
    elif args.dataset == "CityStreet":
        val_list = "/groups1/gca50095/aca10350zi/CityStreet/GT_density_maps/camera_view/train/"
        test_list = "/groups1/gca50095/aca10350zi/CityStreet/GT_density_maps/camera_view/test/"
        scene = None
        if "view1" in args.test_json:
            scene = "view1"
        elif "view2" in args.test_json:
            scene = "view2"
        elif "view3" in args.test_json:
            scene = "view3"
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
        checkpoint = torch.load(str(args.load_model), torch.device(device))
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

    staticff_file = os.path.join(os.path.dirname(args.load_model), "staticff_test.pickle")
    if args.StaticFF == 1:
        with open(staticff_file, mode="rb") as f:
            # staticff_num = pickle.load(f)
            staticff = pickle.load(f)
    else:
        staticff = None
    # staticff = torch.from_numpy(staticff_num.astype(np.float32)).clone()
    # staticff = staticff.to(device)

    # staticff_file = os.path.join(os.path.dirname(args.load_model), "staticff_val.pickle")
    # with open(staticff_file, mode="rb") as f:
    #     staticff_val = pickle.load(f)

    torch.backends.cudnn.benchmark = True

    # mae, rmse, pix_mae, pix_rmse = validate(val_list, model, staticff_val, device, savefolder, static_param=static_param, temperature_param=temperature_param, beta_param=beta_param, delta_param=delta_param, mode="_val")
    # print(' best MAE {mae:.3f}, pix MAE {pix_mae:.5f} \n best RMSE {rsme:.3f}, pix RMSE {pix_rmse:.5f}'
    #       .format(mae=mae, pix_mae=pix_mae, rsme=rmse, pix_rmse=pix_rmse))
    # with open(os.path.join(savefolder, '{}_val.csv'.format(savefilename)), mode='w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([mae, rmse, pix_mae, pix_rmse])

    mae, mae_std, rmse, rmse_std, var, pix_mae, pix_mae_std, pix_rmse, pix_rmse_std, pix_var = validate(test_list, model, staticff, device, savefolder, tested_num=0, static_param=static_param, temperature_param=temperature_param, beta_param=beta_param, delta_param=delta_param, scene=scene, result_per_data_path=result_per_data_path)
    print(' best MAE {mae:.3f} ({mae_std:.3f}), pix MAE {pix_mae:.5f} ({pix_mae_std:.5f}), \n best RMSE {rsme:.3f} ({rmse_std:.3f}), pix RMSE {pix_rmse:.5f} ({pix_rmse_std:.5f}) \n best var {var:.3f}, pix var {pix_var:.5f}'
          .format(mae=mae, pix_mae=pix_mae, rsme=rmse, pix_rmse=pix_rmse, var=var, pix_var=pix_var, mae_std=mae_std, pix_mae_std=pix_mae_std, rmse_std=rmse_std, pix_rmse_std=pix_rmse_std))
    with open(os.path.join(savefolder, '{}_test.csv'.format(savefilename)), mode='w') as f:
        writer = csv.writer(f)
        writer.writerow([mae, rmse, pix_mae, pix_rmse, mae_std, rmse_std, pix_mae_std, pix_rmse_std])


def validate(val_list, model, staticff, device, savefolder=None, tested_num=None,  static_param=1.0, temperature_param=1.0, beta_param=0.5, delta_param=0.5, mode="", scene=None, result_per_data_path=None):
    global args
    add_mode = "add" in args.test_json
    print('begin val, (ADD: {})'.format("True" if add_mode else "False"))
    val_dataset = dataset_factory(val_list, args, mode=args.data_mode, scene=scene, add=add_mode)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1)

    model.eval()

    var = 0
    mae = 0
    rmse = 0
    pix_mae = []
    pix_rmse = []
    pix_var = []

    pred_scene = []
    gt = []

    past_output = None

    for i, (prev_img, img, post_img, target) in enumerate(val_loader):
        if tested_num is not None:
            if i < tested_num:
                continue
        # only use previous frame in inference time, as in real-time application scenario, future frame is not available
        prev_img = prev_img.to(device, dtype=torch.float)
        prev_img = Variable(prev_img)

        img = img.to(device, dtype=torch.float)
        img = Variable(img)

        # print(os.path.join(savefolder, "output{}".format(mode), "{}.npz".format(i)))
        if os.path.isfile(os.path.join(savefolder, "output{}".format(mode), "{}.npz".format(i))):
            # print(i)
            pred = np.load(os.path.join(savefolder, "output{}".format(mode), "{}.npz".format(i)))["x"]
        else:
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

            pred = overall.detach().numpy().copy()
        target = target.detach().numpy().copy()

        debug_hist_path = os.path.join("/home/aca10350zi/habara/CrowdCounting-using-PedestrianFlow/data/debug")
        debug_hist = {}
        debug_hist["original"] = pred
        if args.StaticFF == 1:
            debug_hist["staticff"] = staticff
            pred *= static_param*staticff
            debug_hist["pred_with_staticff"] = pred

        pred_g = gaussian_filter(pred, 3)
        if args.DynamicFF == 1 and past_output is not None:
            debug_hist["t-1_dynamicff"] = past_output
            d_t_prev = gaussian_filter(past_output, 3)
            debug_hist["t-1_dynamicff_g"] = d_t_prev
            past_output = beta_param * pred_g + (1 - delta_param) * d_t_prev
            # print("past output {}".format(torch.sum(past_output)))
            past_output_g = gaussian_filter(past_output, 3)
            height, width = past_output_g.shape
            deno = np.sum(np.exp(past_output_g/temperature_param)) / (height * width)
            nume = np.exp(past_output_g/temperature_param)
            dynamicff = nume / deno
            # dynamicff = past_output
            debug_hist["t_dynamicff"] = dynamicff
            pred *= dynamicff
            # rate = 0.005
            # pred = (pred + rate * dynamicff) / (1 + rate)
            debug_hist["pred_with_dynamicff"] = pred

        if args.DynamicFF == 1 and past_output is None:
            past_output = beta_param * pred_g

        pred_sum = np.sum(pred)
        save_dir = os.path.join(savefolder, "output{}".format(mode))
        if not os.path.isfile(os.path.join(save_dir, "{}.npz".format(i))):
            print(save_dir, i, "saved")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savez_compressed(os.path.join(save_dir, "{}.npz".format(i)), x=overall.detach().numpy().copy())

        # pix_mae.append(mean_absolute_error(target.squeeze(), pred))
        pix_mae.append(np.nanmean(np.abs(target.squeeze()-pred)))
        pix_rmse.append(np.sqrt(np.nanmean(np.square(target.squeeze()-pred))))
        pix_var.append(np.var(pred))

        # print(np.sum(target), pred_sum)
        pred_scene.append(pred_sum)
        gt.append(np.sum(target))

        with open(result_per_data_path, mode="a") as f:
            writer = csv.writer(f)
            target_sum = np.sum(target)
            mae_per_data = abs(target_sum - pred_sum)
            pix_mae_per_data = np.nanmean(np.abs(target.squeeze()-pred))
            pix_rmse_per_data = np.sqrt(np.nanmean(np.square(target.squeeze()-pred)))
            writer.writerow([i, target_sum, pred_sum, mae_per_data, pix_mae_per_data, pix_rmse_per_data])

        debug_ = False
        if debug_ and i%50 == 1:
            hist_nums = len(debug_hist.keys())
            fig, axes = plt.subplots(2, hist_nums, figsize=(hist_nums*2, 4), tight_layout=True)
            for j, k in enumerate(debug_hist.keys()):
                # print(j%3)
                if hist_nums >= 2:
                    axes[0, j].imshow(debug_hist[k])
                    axes[0, j].set_title(k)
                    axes[1, j].hist(debug_hist[k].ravel())
                    axes[1, j].set_ylim(0, 200)
                else:
                    axes[0].imshow(debug_hist[k])
                    axes[0].set_title(k)
                    axes[1].hist(debug_hist[k].ravel())
                    axes[1].set_ylim(0, 200)

            fig.savefig("/home/aca10350zi/habara/CrowdCounting-using-PedestrianFlow/data/debug/{}{}.png".format(i, mode), dpi=300)


    #print("pred: {}".format(np.array(pred)))
    #print("target: {}".format(np.array(gt)))
    abs_diff = np.abs(np.array(pred_scene)-np.array(gt))
    mae = np.nanmean(abs_diff)
    # mae = mean_absolute_error(pred_scene, gt)
    mae_std = np.nanstd(abs_diff)
    var = np.var(np.array(pred_scene))
    squared_diff = np.square(np.array(pred_scene)-np.array(gt))
    # rmse = np.sqrt(mean_squared_error(pred_scene, gt))
    rmse = np.sqrt(np.array(np.nanmean(squared_diff)))
    rmse_std = np.sqrt(np.array(np.nanstd(squared_diff)))
    pix_mae_val = np.nanmean(np.array(pix_mae))
    pix_mae_val_std = np.nanstd(np.array(pix_mae))
    pix_rmse_val = np.nanmean(np.array(pix_rmse))
    pix_rmse_val_std = np.nanstd(np.array(pix_rmse))
    pix_var_val = np.nanmean(np.array(pix_var))

    return mae, mae_std, rmse, rmse_std, var, pix_mae_val, pix_mae_val_std, pix_rmse_val, pix_rmse_val_std, pix_var_val

if __name__ == "__main__":
    main()
