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
from lib import model
from lib import dataset
from torchvision import transforms
from torch.autograd import Variable
import scipy.io
from scipy.ndimage.filters import gaussian_filter
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


def search(args, scene_num):
    print("Scene Num:", scene_num)
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
    os.makedirs(savefolder, exist_ok=True)
    # if os.path.isfile(os.path.join(savefolder, 'ff_param.csv')):
    #     with open(os.path.join(savefolder, 'ff_param.csv')) as f:
    #         reader = csv.reader(f)
    #         static_param, dynamic_param = reader[0][0], reader[0][1]
    #         return static_param, dynamic_param

    # img_paths = dataset.Datapath(args.test_path, args.dataset, mode="add")
    img_paths = dataset.Datapath(args.test_path, args.dataset, mode="add")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.bn != 0 or args.do_rate > 0.0:
        load_weight = True
    else:
        load_weight = False
    # CANnet = model.CANNet2s(load_weights=load_weight, activate=args.activate, bn=args.bn, do_rate=args.do_rate)
    # CANnet.to(device)
    # CANnet.load_state_dict(fix_model_state_digct(torch.load(normal_weights, torch.device(device))['state_dict']))
    # CANnet.eval()

    past_output = None

    with open(os.path.join(os.path.dirname(args.normal_weight), "staticff_test.pickle"), "rb") as f:
        staticff = pickle.load(f)

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
    # temperature_params = [0.1, 0.5, 1., 5., 10., 50., 100.]
    temperature_params = [0.3, 0.5, 1., 5., 10., 50., 100.]

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
                        prev_img, img, target = img_paths[i]
                        target_num = np.array(target)

                        if len(target_nums) < scene_num:
                            target_nums.append(target_num.sum())

                        normal_dense = np.load(os.path.join(os.path.dirname(args.normal_weight), "output", "{}.npz".format(i)))["x"]
                        height, width = normal_dense.shape

                        if args.StaticFF == 1:
                            normal_dense *= s*staticff

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

    return static_param, temperature_param, beta_param, delta_param


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
    CANnet = model.CANNet2s(load_weights=load_weight, activate=args.activate, bn=args.bn, do_rate=args.do_rate)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the csv file of the Datasets path.
                                                 In default, path is 'Data/TestData_Path.csv'
                                                 """)

    parser.add_argument('val_path', default='Val Data_Path.csv')  # val data path csv
    parser.add_argument('test_path', default='Test Data_Path.csv')  # test data path csv
    parser.add_argument('-wd', '--width', type=int, default=640)  # image width that input to model
    parser.add_argument('-ht', '--height', type=int, default=360)  # image height thta input to model
    parser.add_argument('-nw', '--normal_weight')
    parser.add_argument('-num', '--img_num', default=10)
    parser.add_argument('--dataset', default="CrowdFlow")
    parser.add_argument('--activate', default="leaky")
    parser.add_argument('--bn', default=0, type=int)
    parser.add_argument('--do_rate', default=0.0, type=float)
    parser.add_argument('--cf', default=0)
    parser.add_argument('--cod', default="activate-relu")
    parser.add_argument('--cond2res', default="Cond2Res.json")
    parser.add_argument('--res', default=None)
    parser.add_argument('--imageCond', default=None)
    parser.add_argument('--DynamicFF', default=0, type=int)
    parser.add_argument('--StaticFF', default=0, type=int)

    args = parser.parse_args()

    if args.dataset == "CrowdFlow":
        with open(args.val_path) as f:
            reader = csv.reader(f)
            val_pathes = [row for row in reader]
        with open(args.test_path) as f:
            reader = csv.reader(f)
            test_pathes = [row for row in reader]
    elif args.dataset == "FDST":
        with open(args.val_path, "r") as f:
            val_pathes = json.load(f)
        with open(args.test_path, "r") as f:
            test_pathes = json.load(f)

    """
    with open(args.cond2res) as f:
        df = json.load(f)

    args.res = df[args.imageCond]["cf_{}_{}".format(args.cf, args.cod)]
    """
    static_param, temperature_param, beta_param, delta_param = search(args, 100)
    print("best StaticFF param: {}, best temperature param: {}, best beta param: {}, best delta param: {}".format(static_param, temperature_param, beta_param, delta_param))
    # main(args, 0, len(test_pathes), static_param, beta_param, delta_param, temperature_param)

