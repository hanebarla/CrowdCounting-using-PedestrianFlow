from genericpath import isfile
import os
import json
import cv2
import PIL.Image as Image
import numpy as np
import torch
import torchvision
import argparse
import csv
import pickle
import random

from lib.utils import *
from lib.model import CANNet2s
from lib import dataset
from torchvision import transforms
from torch.autograd import Variable
import scipy.io
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error


def reconstruction_forward(prev_flow, device):
    prev_flow = prev_flow.to(device)
    mask_boundry = torch.zeros(prev_flow.shape[2:]).to(device)
    mask_boundry[0,:] = 1.0
    mask_boundry[-1,:] = 1.0
    mask_boundry[:,0] = 1.0
    mask_boundry[:,-1] = 1.0

    reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry

    return reconstruction_from_prev

def search(args, scene_index, scene_num):
    print("Scene Num:", scene_num)
    if args.StaticFF == 1 and args.DynamicFF == 1:
        savefilename = 'BothFF_Demo'
    elif args.StaticFF == 1:
        savefilename = 'StaticFF_Demo'
    elif args.DynamicFF == 1:
        savefilename = 'DynamicFF_Demo'
    else:
        savefilename = 'noFF_Demo'
    savefolder = os.path.join('/groups1/gca50095/aca10350zi/habara_exp/CityStreet_{}/'.format(args.data_mode), str(args.penalty), 'no_change', str(scene_index), savefilename)
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    # if os.path.isfile(os.path.join(savefolder, 'ff_param.csv')):
    #     with open(os.path.join(savefolder, 'ff_param.csv')) as f:
    #         reader = csv.reader(f)
    #         static_param, dynamic_param = reader[0][0], reader[0][1]
    #         return static_param, dynamic_param

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    past_output = None

    with open(os.path.join("/groups1/gca50095/aca10350zi/CityStreet/GT_density_maps/camera_view/test/staticff_{}.pickle".format(scene_index)), "rb") as f:
        staticff = pickle.load(f)
        staticff = np.squeeze(staticff)

    with h5py.File("/groups1/gca50095/aca10350zi/CityStreet/GT_density_maps/camera_view/test/ff_{}.h5".format(scene_index)) as f:
        targets = np.array(f["density_{}".format(args.data_mode)])  # data_mode: once or add

    target_nums = []
    static_param = 0
    static_param_pix = 0
    beta_param = 0
    beta_param_pix = 0
    delta_param = 0
    delta_param_pix = 0
    temperature_param = 0
    temperature_param_pix = 0
    mae = 1000
    pix_mae = 1000

    # dynamic_params = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    # dynamic_params = np.logspace(1, 2, 10).tolist()
    # print(dynamic_params)
    static_params = [0.8, 0.9, 1.0, 1.1, 1.2]
    beta_params = [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5]
    delta_params = [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5]
    temperature_params = [0.1, 0.5, 1., 5., 10., 50., 100., 500., 1000.]

    for i_s, s in enumerate(static_params):
        if args.StaticFF != 1 and i_s > 0:
            continue
        for i_t, temperature in enumerate(temperature_params):
            if args.DynamicFF != 1 and i_t > 0:
                continue
            for i_b, beta in enumerate(beta_params):
                if args.DynamicFF != 1 and i_b > 0:
                    continue
                for i_d, delta in enumerate(delta_params):
                    if args.DynamicFF != 1 and i_d > 0:
                        continue
                    print("===== ", s, temperature, beta, delta, " =====")
                    tmp_output_nums = []
                    _pix_mae = []
                    through_path = False

                    for i in range(scene_num):
                        # prev_img, img, target = img_paths[i]
                        # target_num = np.array(target)
                        target_num = targets[i]
                        if len(target_nums) < scene_num:
                            target_nums.append(target_num.sum())

                        normal_dense = np.load(os.path.join("/groups1/gca50095/aca10350zi/habara_exp/CityStreet_{}".format(args.data_mode), str(args.penalty), "no_change", scene_index, "{}.npz".format(i)))["x"]
                        height, width = normal_dense.shape

                        if args.StaticFF == 1:
                            normal_dense = normal_dense*s*staticff

                        normal_dense_gauss = gaussian_filter(normal_dense, 3)

                        if args.DynamicFF == 1 and past_output is not None:
                            d_t_prev = gaussian_filter(past_output, 3)
                            past_output = beta * normal_dense_gauss + (1 - delta) * d_t_prev
                            past_output = gaussian_filter(past_output, 3)
                            # print(past_output.sum())
                            exp_sum = np.sum(np.exp(past_output/temperature)) + 1e-5
                            # print(exp_sum)
                            dynamicff = height * width * np.exp(past_output/temperature) / exp_sum
                            # normal_dense *= gaussian_filter(dynamic_coefficient * past_output, 3)
                            normal_dense *= dynamicff

                        if past_output is None:
                            past_output = beta * normal_dense_gauss

                        # _pix_mae.append(mean_absolute_error(np.squeeze(target_num), normal_dense))
                        _pix_mae.append(np.nanmean(np.abs(np.squeeze(target_num)-normal_dense)))
                        tmp_output_nums.append(normal_dense.sum())

                    tmp_mae = mean_absolute_error(target_nums, tmp_output_nums)
                    print(tmp_mae)

                    tmp_pix_mae = np.mean(np.array(_pix_mae))
                    print(tmp_pix_mae)
                    # print(tmp_pix_mae)
                    if tmp_mae < mae:
                        mae = tmp_mae
                        static_param = s
                        beta_param = beta
                        delta_param = delta
                        temperature_param = temperature

                    if tmp_pix_mae < pix_mae:
                        pix_mae = tmp_pix_mae
                        static_param_pix = s
                        beta_param_pix = beta
                        delta_param_pix = delta
                        temperature_param_pix = temperature

    with open(os.path.join(savefolder, 'ff_param.csv'), mode='w') as f:
        writer = csv.writer(f)
        writer.writerow([static_param, temperature_param, beta_param, delta_param])

    with open(os.path.join(savefolder, 'ff_param_pix.csv'), mode='w') as f:
        writer = csv.writer(f)
        writer.writerow([static_param_pix, temperature_param_pix, beta_param_pix, delta_param_pix])

    print("With MAE, best StaticFF param: {}, best temperature param: {}, best beta param: {}, best delta param: {}".format(static_param, temperature_param, beta_param, delta_param))
    print("With pix-MAE, best StaticFF param: {}, best temperature param: {}, best beta param: {}, best delta param: {}".format(static_param_pix, temperature_param_pix, beta_param_pix, delta_param_pix))

    return static_param, temperature_param, beta_param, delta_param, savefolder

def val(args, scene_index, scene_num, pix=False):
    if args.StaticFF == 1 and args.DynamicFF == 1:
        savefilename = 'BothFF_Demo'
    elif args.StaticFF == 1:
        savefilename = 'StaticFF_Demo'
    elif args.DynamicFF == 1:
        savefilename = 'DynamicFF_Demo'
    else:
        savefilename = 'noFF_Demo'

    savefolder = os.path.join('/groups1/gca50095/aca10350zi/habara_exp/CityStreet_{}/'.format(args.data_mode), str(args.penalty), 'no_change', str(scene_index), savefilename)
    param_path = os.path.join(savefolder, "ff_param.csv" if not pix else "ff_param_pix.csv")
    if os.path.isfile(param_path):
        print(param_path)
        with open(param_path) as f:
            reader = csv.reader(f)
            params = [row for row in reader]
            print(params)
            static_param, temperature_param, beta_param, delta_param = float(params[0][0]), float(params[0][1]), float(params[0][2]), float(params[0][3])

    with open(os.path.join("/groups1/gca50095/aca10350zi/CityStreet/GT_density_maps/camera_view/test/staticff_{}.pickle".format(scene_index)), "rb") as f:
        staticff = pickle.load(f)
        staticff = np.squeeze(staticff)

    with h5py.File("/groups1/gca50095/aca10350zi/CityStreet/GT_density_maps/camera_view/test/ff_{}.h5".format(scene_index)) as f:
        targets = np.array(f["density_{}".format(args.data_mode)])  # data_type: once or add

    pred_dir = os.path.join("/groups1/gca50095/aca10350zi/habara_exp/CityStreet_{}".format(args.data_mode), str(args.penalty), "no_change", "view1")
    files = os.listdir(os.path.join(pred_dir))
    pred_files = [os.path.join(pred_dir, f) for f in files if (os.path.isfile(os.path.join(pred_dir, f)) and (".npz" in f))]
    pred_length = len(pred_files)
    print(pred_length)

    result_per_data_path = os.path.join(savefolder, 'res_pix.csv' if pix else 'res.csv')
    if(os.path.isfile(result_per_data_path)):
        os.remove(result_per_data_path)

    var = 0
    mae = 0
    rmse = 0
    pix_mae = []
    pix_rmse = []
    pix_var = []

    pred_scene = []
    gt = []

    past_output = None
    for i, p in enumerate(pred_files):
        if i < scene_num:
            continue

        pred = np.load(os.path.join(pred_dir, "{}.npz".format(i)))["x"]

        target = targets[i]

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
            dynamicff = height * width * np.exp(past_output_g/temperature_param) / np.sum(np.exp(past_output_g/temperature_param))
            # dynamicff = past_output
            debug_hist["t_dynamicff"] = dynamicff
            pred *= dynamicff
            # rate = 0.005
            # pred = (pred + rate * dynamicff) / (1 + rate)
            debug_hist["pred_with_dynamicff"] = pred

        if past_output is None:
            past_output = beta_param * pred_g

        pred_sum = np.sum(pred)
        # print(pred_sum)
        pix_mae.append(mean_absolute_error(target.squeeze(), pred))
        pix_rmse.append(np.sqrt(mean_squared_error(target.squeeze(), pred)))
        pix_var.append(np.var(pred))

        pred_scene.append(pred_sum)
        gt.append(np.sum(target))

        with open(result_per_data_path, mode="a") as f:
            writer = csv.writer(f)
            target_sum = np.sum(target)
            mae_per_data = abs(target_sum - pred_sum)
            pix_mae_per_data = np.nanmean(np.abs(target.squeeze()-pred))
            pix_rmse_per_data = np.sqrt(np.nanmean(np.square(target.squeeze()-pred)))
            writer.writerow([i, target_sum, pred_sum, mae_per_data, pix_mae_per_data, pix_rmse_per_data])

        debug_ = True
        if debug_ and i%25 == 1:
            hist_nums = len(debug_hist.keys())
            fig, axes = plt.subplots(2, hist_nums, figsize=(hist_nums*2, 4), tight_layout=True)
            for j, k in enumerate(debug_hist.keys()):
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

            fig.savefig(os.path.join(savefolder, "{}.png".format(i)), dpi=300)

    abs_diff = np.abs(np.array(pred_scene)-np.array(gt))
    mae = np.mean(abs_diff)
    mae_std = np.std(abs_diff)

    squared_diff = np.square(np.array(pred_scene)-np.array(gt))
    rmse = np.sqrt(np.array(np.mean(squared_diff)))
    rmse_std = np.sqrt(np.array(np.std(squared_diff)))

    pix_mae_val = np.mean(np.array(pix_mae))
    pix_mae_val_std = np.std(np.array(pix_mae))

    pix_rmse_val = np.mean(np.array(pix_rmse))
    pix_rmse_val_std = np.std(np.array(pix_rmse))

    return mae, mae_std, rmse, rmse_std, pix_mae_val, pix_mae_val_std, pix_rmse_val, pix_rmse_val_std


def main(args, start, end, static_param, temperature_param, beta_param, delta_param):
    normal_weights = args.normal_weight

    if args.StaticFF == 1 and args.DynamicFF == 1:
        savefilename = 'BothFF_Demo'
    elif args.StaticFF == 1:
        savefilename = 'StaticFF_Demo'
    elif args.DynamicFF == 1:
        savefilename = 'DynamicFF_Demo'
    else:
        savefilename = 'noFF_Demo'
    savefolder = os.path.join(os.path.dirname(args.normal_weight), 'images', savefilename)

    img_paths = dataset.Datapath(args.test_path, args.dataset)
    os.makedirs(savefolder, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.bn != 0 or args.do_rate > 0.0:
        load_weight = True
    else:
        load_weight = False
    CANnet = CANNet2s(load_weights=load_weight, activate=args.activate, bn=args.bn, do_rate=args.do_rate)
    CANnet.to(device)
    CANnet.load_state_dict(fix_model_state_dict(torch.load(normal_weights)['state_dict']))
    CANnet.eval()


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    img_dict_keys = [
        'input',
        'pred'
    ]
    img_dict = {
        img_dict_keys[0]: ('img', None),
        img_dict_keys[1]: ('img', None)
    }

    if args.DynamicFF == 1:
        img_dict_keys.append('Dynamic FF')
        img_dict_keys.append('Dynamic hist')
        img_dict['Dynamic FF'] = ('img', None)
        img_dict['Dynamic hist'] = ('hist', None)
    if args.StaticFF == 1:
        img_dict_keys.append('Static FF')
        img_dict_keys.append('Static hist')
        img_dict['Static FF'] = ('img', None)
        img_dict['Static hist'] = ('hist', None)

    past_output = None

    with open(os.path.join(os.path.dirname(args.normal_weight), "staticff_test.pickle"), "rb") as f:
        staticff = pickle.load(f)

    pedestrian_num = []
    pix_mae = []
    pix_rmse = []

    pred_scene = []
    gt = []
    for i in range(start, end):
        DemoImg = CompareOutput(img_dict_keys)

        prev_img, img, target = img_paths[i]
        target_num = np.array(target)
        # pedestrian_num.append(target_num.sum()/target_num.max())
        # print("pedestrian", target_num.sum()/target_num.max())
        pedestrian_num.append(target_num.sum())
        print("pedestrian", target_num.sum())

        prev_img = prev_img.resize((640,360))
        img = img.resize((640,360))
        torch_prev_img = torchvision.transforms.ToTensor()(prev_img)
        torch_img = torchvision.transforms.ToTensor()(img)

        of_prev_img = prev_img.resize((80, 45))
        of_prev_img = np.array(of_prev_img)
        of_img = img.resize((80, 45))
        of_img = np.array(of_img)

        prev_img = transform(prev_img).cuda()
        img = transform(img).cuda()

        prev_img = prev_img.cuda()
        prev_img = Variable(prev_img)

        img = img.cuda()
        img = Variable(img)

        img = img.unsqueeze(0)
        prev_img = prev_img.unsqueeze(0)

        input_num = prev_img[0, :, :, :].detach().cpu().numpy()
        input_num = input_num.transpose((1, 2, 0))
        input_num = input_num * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

        normal_dense = np.load(os.path.join(os.path.dirname(args.normal_weight),"output", "{}.npz".format(i)))["x"]
        height, width = normal_dense.shape

        if args.StaticFF == 1:
            normal_dense *= static_param*staticff

        normal_dense_gauss = gaussian_filter(normal_dense, 3)

        if args.DynamicFF == 1 and past_output is not None:
            d_t_prev = gaussian_filter(past_output, 3)
            past_output = beta_param * normal_dense_gauss + (1 - delta_param) * d_t_prev
            # normal_dense_gauss *= gaussian_filter(past_output, 3)
            past_output = gaussian_filter(past_output, 3)
            exp_sum = np.sum(np.exp(past_output/temperature_param)) + 1e-7
            dynamic_ff = height * width * np.exp(past_output/temperature_param) / exp_sum
            # normal_dense *= gaussian_filter(dynamic_param*past_output, 3)
            normal_dense *= dynamic_ff

        if past_output is None:
            past_output = beta_param * normal_dense_gauss
        """
        direct_num = output_direct[0, :, :, :].detach().cpu().numpy()
        direct_quiver = NormalizeQuiver(direct_num)
        direct_num = direct_num.transpose((1, 2, 0))
        """
        # direct_dense = tm_output_to_dense(direct_num)
        # print(np.any(np.isnan(input_num)))
        img_dict = {
            img_dict_keys[0]: ('img', input_num),
            img_dict_keys[1]: ('img', normal_dense),
        }

        if args.DynamicFF == 1:
            img_dict['Dynamic FF'] = ('img', past_output)
            img_dict['Dynamic hist'] = ('hist', past_output.ravel())
        if args.StaticFF == 1:
            img_dict['Static FF'] = ('img', staticff)
            img_dict['Static hist'] = ('hist', static_param*staticff.ravel())

        DemoImg.append_pred(img_dict)

        # del D_CANnet
        del img
        del prev_img

        plt.close()

        print("{} done\n".format((i+1)), end="")

        DemoImg.plot_img(suptitle=str(args.res))
        DemoImg.save_fig(name=os.path.join(savefolder, 'demo-{}.png'.format(int(i))))
        pix_mae.append(mean_absolute_error(np.squeeze(target), normal_dense))
        pix_rmse.append(np.sqrt(mean_squared_error(np.squeeze(target), normal_dense)))

        pred_scene.append(np.sum(normal_dense))
        gt.append(np.sum(target))

    mae = mean_absolute_error(pred_scene, gt)
    rmse = np.sqrt(mean_squared_error(pred_scene, gt))
    pix_mae_ = np.mean(np.array(pix_mae))
    pix_rmse_ = np.mean(np.array(pix_rmse))

    with open(os.path.join(os.path.dirname(args.normal_weight), '{}_result_test.csv'.format(savefilename)), mode='w') as f:
        writer = csv.writer(f)
        writer.writerow([mae, rmse, pix_mae_, pix_rmse_])

    print(len(DemoImg.losses_dict['input']))
    print("Scene average pedestrian num: {}".format(sum(pedestrian_num)/len(pedestrian_num)))

def pred_no_ff(args, savefolder, device, scene):
    # data set
    test_dir = "/groups1/gca50095/aca10350zi/CityStreet/GT_density_maps/camera_view/test/"
    test_dataset = dataset.CityStreetDataset(
        test_dir,
        data_type=args.data_mode,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        scene=scene
    )
    print(len(test_dataset))
    print(test_dataset.scene)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1
    )

    # load model
    if args.bn != 0 or args.do_rate > 0.0:
        load_weight = True
    else:
        load_weight = False
    model = CANNet2s(load_weights=load_weight, activate=args.activate, bn=args.bn, do_rate=args.do_rate)
    checkpoint = torch.load(args.load_model, torch.device(device))
    model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
    best_prec1 = checkpoint['val']

    # multi gpu
    if torch.cuda.device_count() > 1:
        print("You can use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    mae = 0
    rmse = 0
    _pix_mae = []
    _pix_rmse = []

    pred_scene = []
    gt = []

    past_output = None

    for i, (prev_img, img, post_img, target) in enumerate(test_loader):
        prev_img = prev_img.to(device)
        img = img.to(device)

        if os.path.isfile(os.path.join(savefolder, "{}.npz".format(i))):
            pred = np.load(os.path.join(savefolder, "{}.npz".format(i)))["x"]
        else:
            with torch.no_grad():
                prev_flow = model(prev_img, img)
                prev_flow_inverse = model(img, prev_img)

                mask_boundry = torch.zeros(prev_flow.shape[2:]).to(device)
                mask_boundry[0,:] = 1.0
                mask_boundry[-1,:] = 1.0
                mask_boundry[:,0] = 1.0
                mask_boundry[:,-1] = 1.0

                reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
                reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry

                overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).type(torch.FloatTensor)
                pred = overall.detach().numpy().copy()

        target = target.detach().numpy().copy()

        pred_sum = np.sum(pred)
        if not os.path.isfile(os.path.join(savefolder, "{}.npz".format(i))):
            print(savefolder, i, "saved")
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)
            np.savez_compressed(os.path.join(savefolder, "{}.npz".format(i)), x=pred)

        _pix_mae.append(np.nanmean(np.abs(target.squeeze()-pred)))
        _pix_rmse.append(np.sqrt(np.nanmean(np.square(target.squeeze()-pred))))
        pred_scene.append(pred_sum)
        gt.append(np.sum(target))

    abs_diff = np.abs(np.array(pred_scene)-np.array(gt))
    mae = np.nanmean(abs_diff)
    mae_std = np.nanstd(abs_diff)

    squared_diff = np.square(np.array(pred_scene)-np.array(gt))
    rmse = np.sqrt(np.array(np.nanmean(squared_diff)))
    rmse_std = np.sqrt(np.array(np.nanstd(squared_diff)))

    pix_mae = np.nanmean(np.array(_pix_mae))
    pix_mae_std = np.nanstd(np.array(_pix_mae))

    pix_rmse = np.nanmean(np.array(_pix_rmse))
    pix_rmse_std = np.nanstd(np.array(_pix_rmse))

    print(scene)
    print("MAE, RMSE, pix-MAE, pix-RMSE")
    print("{}, {}, {}, {}".format(mae, mae_std, rmse, rmse_std))
    print("{}, {}, {}, {}".format(pix_mae, pix_mae_std, pix_rmse_std, pix_rmse_std))

    return pred_scene, gt, _pix_mae, _pix_rmse


def pred_noff(args):
    view_list = ["view1", "view2", "view3"]
    pred = []
    gt = []
    _pix_mae = []
    _pix_rmse = []
    for scene in view_list:
        savefolder = os.path.join(os.path.dirname(args.load_model), scene)
        print(savefolder)
        tmp_pred, tmp_gt, tmp_pix_mae, tmp_pix_rmse = pred_no_ff(args, savefolder, device, scene)
        pred.extend(tmp_pred)
        gt.extend(tmp_gt)
        _pix_mae.extend(tmp_pix_mae)
        _pix_rmse.extend(tmp_pix_rmse)

    print("Whole scene")
    abs_diff = np.abs(np.array(pred)-np.array(gt))
    mae = np.nanmean(abs_diff)
    mae_std = np.nanstd(abs_diff)
    squared_diff = np.square(np.array(pred)-np.array(gt))
    rmse = np.sqrt(np.array(np.nanmean(squared_diff)))
    rmse_std = np.sqrt(np.array(np.nanstd(squared_diff)))
    pix_mae = np.nanmean(np.array(_pix_mae))
    pix_mae_std = np.nanstd(np.array(_pix_mae))
    pix_rmse = np.nanmean(np.array(_pix_rmse))
    pix_rmse_std = np.nanstd(np.array(_pix_rmse))
    print("MAE, RMSE, pix-MAE, pix-RMSE")
    print("{}, {}, {}, {}".format(mae, mae_std, rmse, rmse_std))
    print("{}, {}, {}, {}".format(pix_mae, pix_mae_std, pix_rmse, pix_rmse_std))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the csv file of the Datasets path.
                                                 In default, path is 'Data/TestData_Path.csv'
                                                 """)

    parser.add_argument('val_path', default='Val Data_Path.csv')  # val data path csv
    parser.add_argument('test_path', default='Test Data_Path.csv')  # test data path csv
    parser.add_argument('load_model', type=str)
    parser.add_argument('-num', '--img_num', default=10)
    parser.add_argument('--dataset', default="FDST")
    parser.add_argument('--data_mode', default='once')  # once or add
    parser.add_argument('--activate', default="leaky")
    parser.add_argument('--bn', default=0, type=int)
    parser.add_argument('--do_rate', default=0.0, type=float)
    parser.add_argument('--res', default=None)
    parser.add_argument('--imageCond', default=None)
    parser.add_argument('--penalty', default=0)
    parser.add_argument('--DynamicFF', default=0, type=int)
    parser.add_argument('--StaticFF', default=0, type=int)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pred_noff(args)
    # view_list = ["view1", "view2", "view3"]
    # pred = []
    # gt = []
    # _pix_mae = []
    # _pix_rmse = []
    # for scene in view_list:
    #     savefolder = os.path.join(os.path.dirname(args.load_model), scene)
    #     print(savefolder)
    #     tmp_pred, tmp_gt, tmp_pix_mae, tmp_pix_rmse = pred_no_ff(args, savefolder, device, scene)
    #     pred.extend(tmp_pred)
    #     gt.extend(tmp_gt)
    #     _pix_mae.extend(tmp_pix_mae)
    #     _pix_rmse.extend(tmp_pix_rmse)

    # print("Whole scene")
    # abs_diff = np.abs(np.array(pred)-np.array(gt))
    # mae = np.nanmean(abs_diff)
    # mae_std = np.nanstd(abs_diff)
    # squared_diff = np.square(np.array(pred)-np.array(gt))
    # rmse = np.sqrt(np.array(np.nanmean(squared_diff)))
    # rmse_std = np.sqrt(np.array(np.nanstd(squared_diff)))
    # pix_mae = np.nanmean(np.array(_pix_mae))
    # pix_mae_std = np.nanstd(np.array(_pix_mae))
    # pix_rmse = np.nanmean(np.array(_pix_rmse))
    # pix_rmse_std = np.nanstd(np.array(_pix_rmse))
    # print("MAE, RMSE, pix-MAE, pix-RMSE")
    # print("{}, {}, {}, {}".format(mae, mae_std, rmse, rmse_std))
    # print("{}, {}, {}, {}".format(pix_mae, pix_mae_std, pix_rmse, pix_rmse_std))

    scenes = ["view1", "view2", "view3"]
    scene_nums = 0
    for k in scenes:
        print("\n\n===== Scene index is", k, "=====")
        # static_param, temperature_param, beta_param, delta_param, savefolder = search(args, k, scene_nums)
        # print("best StaticFF param: {}, best temperature param: {}, best beta param: {}, best delta param: {}".format(static_param, temperature_param, beta_param, delta_param))
        mae, mae_std, rmse, rmse_std, pix_mae, pix_mae_std, pix_rmse, pix_rmse_std = val(args, k, scene_nums)
        print('best MAE {mae:.3f} ({mae_std:.3f}), pix MAE {pix_mae:.5f} ({pix_mae_std:.5f})'.format(mae=mae, pix_mae=pix_mae, mae_std=mae_std, pix_mae_std=pix_mae_std))
        print('best RMSE {rsme:.3f} ({rmse_std:.3f}), pix RMSE {pix_rmse:.5f} ({pix_rmse_std:.5f})'.format(rsme=rmse, pix_rmse=pix_rmse, rmse_std=rmse_std, pix_rmse_std=pix_rmse_std))
        # with open(os.path.join(savefolder, 'test.csv'), mode='w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([mae, mae_std, rmse, rmse_std, pix_mae, pix_mae_std, pix_rmse, pix_rmse_std])
        mae, mae_std, rmse, rmse_std, pix_mae, pix_mae_std, pix_rmse, pix_rmse_std = val(args, k, scene_nums, pix=True)
        print('best MAE {mae:.3f} ({mae_std:.3f}), pix MAE {pix_mae:.5f} ({pix_mae_std:.5f})'.format(mae=mae, pix_mae=pix_mae, mae_std=mae_std, pix_mae_std=pix_mae_std))
        print('best RMSE {rsme:.3f} ({rmse_std:.3f}), pix RMSE {pix_rmse:.5f} ({pix_rmse_std:.5f})'.format(rsme=rmse, pix_rmse=pix_rmse, rmse_std=rmse_std, pix_rmse_std=pix_rmse_std))
        # with open(os.path.join(savefolder, 'test_pix.csv'), mode='w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([mae, mae_std, rmse, rmse_std, pix_mae, pix_mae_std, pix_rmse, pix_rmse_std])

        # raise ValueError
    # main(args, 0, len(test_pathes), static_param, beta_param, delta_param, temperature_param)

