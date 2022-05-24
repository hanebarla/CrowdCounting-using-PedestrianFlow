import os
import json
import cv2
import PIL.Image as Image
import numpy as np
import torch
import torchvision
import argparse
import csv

from lib.utils import *
from lib import model
from torchvision import transforms
from torch.autograd import Variable
import scipy.io
from scipy.ndimage.filters import gaussian_filter


class Datapath():
    def __init__(self, json_path=None, datakind=None) -> None:
        if datakind == "CrowdFlow":
            with open(json_path) as f:
                reader = csv.reader(f)
                self.img_paths = [row for row in reader]
        else:
            with open(json_path, 'r') as outfile:
                self.img_paths = json.load(outfile)
        self.datakind = datakind

    def __getitem__(self, index):
        if self.datakind == "FDST":
            img_path = self.img_paths[i]

            img_folder = os.path.dirname(img_path)
            img_name = os.path.basename(img_path)
            index = int(img_name.split('.')[0])

            prev_index = int(max(1,index-5))
            prev_img_path = os.path.join(img_folder,'%03d.jpg'%(prev_index))
            print(prev_img_path)
            prev_img = Image.open(prev_img_path).convert('RGB')
            img = Image.open(img_path).convert('RGB')

            gt_path = img_path.replace('.jpg','_resize.h5')
            gt_file = h5py.File(gt_path)
            target = np.asarray(gt_file['density'])

            return prev_img, img, target

        elif self.datakind == "CrowdFlow":
            pathlist = self.img_paths[index]
            t_img_path = pathlist[0]
            t_person_path = pathlist[1]
            t_m_img_path = pathlist[2]
            t_m_person_path = pathlist[3]
            t_m_t_flow_path = pathlist[4]
            t_p_img_path = pathlist[5]
            t_p_person_path = pathlist[6]
            t_t_p_flow_path = pathlist[7]

            prev_img = Image.open(t_m_img_path).convert('RGB')
            img = Image.open(t_img_path).convert('RGB')

            target = Image.open(t_person_path).convert('L')

            return prev_img, img, target

        elif self.datakind == "venice":
            prev_path = self.img_paths[index]["prev"]
            now_path = self.img_paths[index]["now"]
            next_path = self.img_paths[index]["next"]
            target_path = self.img_paths[index]["target"]

            prev_img = Image.open(prev_path).convert('RGB')
            img = Image.open(now_path).convert('RGB')

            target_dict = scipy.io.loadmat(target_path)
            target = np.zeros((720, 1280))

            for p in range(target_dict['annotation'].shape[0]):
                target[int(target_dict['annotation'][p][1]), int(target_dict['annotation'][p][0])] = 1

            target = gaussian_filter(target, 3) * 64
            target = cv2.resize(target, (80, 45))

            return prev_img, img, target

        elif self.datakind == "animal":
            prev_path = self.img_paths[index]["prev"]
            now_path = self.img_paths[index]["now"]
            next_path = self.img_paths[index]["next"]
            target = None

            prev_img = Image.open(prev_path).convert('RGB')
            img = Image.open(now_path).convert('RGB')

            return prev_img, img, target


def demo(args, start, end):
    test_d_path = args.path
    normal_weights = args.normal_weight
    num = args.img_num

    if args.StaticFF == 1 and args.DynamicFF == 1:
        savefilename = 'BothFF_Demo'
    elif args.StaticFF == 1:
        savefilename = 'StaticFF_Demo'
    elif args.DynamicFF == 1:
        savefilename = 'DynamicFF_Demo'
    else:
        savefilename = 'noFF_Demo'
    savefolder = os.path.join(args.normal_weight, savefilename)

    # json file contains the test images
    test_json_path = './movie_data.json'
    # the floder to output density map and flow maps
    output_floder = './plot'

    img_paths = Datapath(args.path, args.dataset)
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
        'pred',
        'pred_quiver'
    ]

    img_dict = {
        img_dict_keys[0]: ('img', None),
        img_dict_keys[1]: ('img', None),
        img_dict_keys[2]: ('quiver', None),
    }

    DemoImg = CompareOutput(img_dict_keys)
    past_output = None

    for i in range(start, end):
        prev_img, img, target = img_paths[i]

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

        result_pp = result_pp.to('cpu').detach().numpy().copy()
        result_pp = np.transpose(result_pp, (1, 2, 0))
        result_pp = cv2.resize(result_pp, (80, 45))

        with torch.set_grad_enabled(False):
            output_normal = CANnet(prev_img, img)
            # output_normal = sigma(output_normal) - 0.5

            """
            output_direct = D_CANnet(prev_img, img)
            # output_direct = sigma(output_direct) - 0.5
            """

        input_num = prev_img[0, :, :, :].detach().cpu().numpy()
        input_num = input_num.transpose((1, 2, 0))
        input_num = input_num * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

        normal_num = output_normal[0, :, :, :].detach().cpu().numpy()
        normal_dense = tm_output_to_dense(normal_num.transpose((1, 2, 0)))

        if args.StaticFF == 1:
            normal_dense_gauss = gaussian_filter(normal_dense, 3)
            normal_dense_gauss_mean = np.mean(normal_dense_gauss)
            normal_dense_gauss[normal_dense_gauss<normal_dense_gauss_mean] = 0
            k = 1
            staticff = np.exp(k*normal_dense_gauss)
            print("Output: {}, {}".format(normal_dense.shape, np.max(normal_dense)))
            print("staticFF: {}, {}".format(staticff.shape, np.max(staticff)))
            normal_num *= staticff

        normal_quiver = NormalizeQuiver(normal_num)
        normal_num = normal_num.transpose((1, 2, 0))
        normal_dense = tm_output_to_dense(normal_num)

        if args.DynamicFF == 1 and past_output is not None:
            d = 0.1
            past_output = normal_dense
            normal_dense += d * past_output

        if past_output is None:
            past_output = normal_dense

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
            img_dict_keys[2]: ('quiver', normal_quiver)
        }

        DemoImg.append_pred(img_dict)

        del CANnet
        # del D_CANnet
        del img
        del prev_img
        del output_normal

        plt.close()

        print("{} done\n".format((i+1)), end="")

    print(len(DemoImg.losses_dict['input']))
    DemoImg.plot_img(suptitle=str(args.res))
    DemoImg.save_fig(name=os.path.join(savefolder, 'demo-{}.png'.format(int(start))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                                 Please specify the csv file of the Datasets path.
                                                 In default, path is 'Data/TestData_Path.csv'
                                                 """)

    parser.add_argument('path', default='TestData_Path.csv')  # Testdata path csv
    parser.add_argument('-wd', '--width', type=int, default=640)  # image width that input to model
    parser.add_argument('-ht', '--height', type=int, default=360)  # image height thta input to model
    parser.add_argument('-nw', '--normal_weight')
    parser.add_argument('-num', '--img_num', default=10)
    parser.add_argument('--dataset', default="FDST")
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
        with open(args.path) as f:
            reader = csv.reader(f)
            pathes = [row for row in reader]
    else:
        with open(args.path, "r") as f:
            pathes = json.load(f)

    """
    with open(args.cond2res) as f:
        df = json.load(f)

    args.res = df[args.imageCond]["cf_{}_{}".format(args.cf, args.cod)]
    """

    # len(pathes)
    print(len(pathes))
    for i in range(len(pathes)):
        # try:
        demo(args, i, i+1)
        # except Exception as e:
            # print("{} is Error".format(i))
            # print(e)
