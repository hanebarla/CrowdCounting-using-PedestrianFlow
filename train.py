import os
from random import shuffle
from lib.model import CANNet2s, SimpleCNN
from lib.utils import save_checkpoint, fix_model_state_dict

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

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CANNet2s')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')
parser.add_argument('--dataset', default="FDST")
parser.add_argument('--exp', default='.')
parser.add_argument('--myloss', default='0.01')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--trainmodel', default="CAN")
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--opt', default="adam")
parser.add_argument('--activate', default="leaky")
parser.add_argument('--bn', default=0, type=int)
parser.add_argument('--do_rate', default=0.0, type=float)
parser.add_argument('--pretrained', default=0, type=int)

dloss_on = False


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
                ]), train=False
            )
    elif arguments.dataset == "CrowdFlow":
        return dataset.CrowdDatasets(
            dlist,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
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
                )
            ])
        )
    else:
        raise ValueError


def main():
    global args, best_prec1, dloss_on

    best_prec1 = 200

    args = parser.parse_args()
    # args.lr = 1e-4
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5*1e-4
    # args.decay         = 1e-3
    # args.start_epoch   = 0
    args.epochs = 50
    args.workers = 8
    args.seed = int(time.time())
    dloss_on = not (float(args.myloss) == 0)

    if args.dataset == "FDST":
        args.print_freq = 400
        with open(args.train_json, 'r') as outfile:
            train_list = json.load(outfile)
        with open(args.val_json, 'r') as outfile:
            val_list = json.load(outfile)
    elif args.dataset == "CrowdFlow":
        args.print_freq = 200
        train_list = args.train_json
        val_list = args.val_json
    elif args.dataset == "venice":
        args.print_freq = 10
        train_list = args.train_json
        val_list = args.val_json
    else:
        raise ValueError

    if args.lr != 1e-4:
        args.savefolder = os.path.join(args.exp, args.dataset, args.myloss, 'lr-' + str(args.lr))
    elif args.opt != "adam":
        args.savefolder = os.path.join(args.exp, args.dataset, args.myloss, 'opt-' + args.opt)
    elif args.activate != "leaky":
        args.savefolder = os.path.join(args.exp, args.dataset, args.myloss, 'activate-' + args.activate)
    elif args.do_rate != 0.0:
        args.savefolder = os.path.join(args.exp, args.dataset, args.myloss, 'do_rate-' + str(args.do_rate))
    elif args.bn != 0:
        args.savefolder = os.path.join(args.exp, args.dataset, args.myloss, 'bn-' + str(args.bn))
    else:
        args.savefolder = os.path.join(args.exp, args.dataset, args.myloss, 'no_change')

    if not os.path.exists(args.savefolder):
        os.makedirs(args.savefolder)
    # logging.basicConfig(filename=os.path.join(args.savefolder, 'train.log'), level=logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.manual_seed(args.seed)

    if os.path.exists(os.path.join(args.savefolder, 'log.txt')) and args.start_epoch == 0:
        os.remove(os.path.join(args.savefolder, 'log.txt'))

    if args.bn != 0 or args.do_rate > 0.0:
        load_weight = True
    else:
        load_weight = False

    if args.trainmodel == "CAN":
        model = CANNet2s(load_weights=load_weight, activate=args.activate, bn=args.bn, do_rate=args.do_rate)
    elif args.trainmodel == "SimpleCNN":
        model = SimpleCNN()

    # pretrained
    if os.path.isfile(os.path.join(args.savefolder, 'checkpoint.pth.tar')):
        checkpoint = torch.load(os.path.join(args.savefolder, 'checkpoint.pth.tar'))
        model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
        args.start_epoch = checkpoint['epoch']
        print("Train resumed: {} epoch".format(args.start_epoch))
        try:
            best_prec1 = checkpoint['val']
            print("best val: {}".format(best_prec1))
        except KeyError:
            print("No Key: val")

    if torch.cuda.device_count() > 1:
        print("You can use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)
    best_prec1 = 100

    # criterion = nn.MSELoss(size_average=False)
    criterion = nn.MSELoss(reduction='sum')

    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)
    elif args.opt == "amsgrad":
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay, amsgrad=True)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr)

    torch.backends.cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):

        train(train_list, model, criterion, optimizer, epoch, device)
        prec1 = validate(val_list, model, criterion, device)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'state_dict': model.state_dict(),
            'val': prec1.item(),
            'epoch': epoch
        }, is_best,
            filename=os.path.join(args.savefolder, 'checkpoint.pth.tar'),
            bestname=os.path.join(args.savefolder, 'model_best.pth.tar'))


def train(train_list, model, criterion, optimizer, epoch, device):
    global args

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_dataset = dataset_factory(train_list, args, mode="train")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    mae = 0
    model.train()
    end = time.time()

    for i, (prev_img, img, post_img, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        prev_img = prev_img.to(device, dtype=torch.float)
        prev_img = Variable(prev_img)

        img = img.to(device, dtype=torch.float)
        img = Variable(img)

        post_img = post_img.to(device, dtype=torch.float)
        post_img = Variable(post_img)

        prev_flow = model(prev_img, img)
        post_flow = model(img, post_img)

        prev_flow_inverse = model(img, prev_img)
        post_flow_inverse = model(post_img, img)

        target = target.type(torch.FloatTensor)[0].cuda()
        target = Variable(target)

        # mask the boundary locations where people can move in/out between regions outside image plane
        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())

        reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
        reconstruction_from_post = torch.sum(post_flow[0,:9,:,:],dim=0)+post_flow[0,9,:,:]*mask_boundry
        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry
        reconstruction_from_post_inverse = F.pad(post_flow_inverse[0,0,1:,1:],(0,1,0,1))+F.pad(post_flow_inverse[0,1,1:,:],(0,0,0,1))+F.pad(post_flow_inverse[0,2,1:,:-1],(1,0,0,1))+F.pad(post_flow_inverse[0,3,:,1:],(0,1,0,0))+post_flow_inverse[0,4,:,:]+F.pad(post_flow_inverse[0,5,:,:-1],(1,0,0,0))+F.pad(post_flow_inverse[0,6,:-1,1:],(0,1,1,0))+F.pad(post_flow_inverse[0,7,:-1,:],(0,0,1,0))+F.pad(post_flow_inverse[0,8,:-1,:-1],(1,0,1,0))+post_flow_inverse[0,9,:,:]*mask_boundry


        # prev_density_reconstruction = torch.sum(prev_flow[0,:9,:,:],dim=0)+prev_flow[0,9,:,:]*mask_boundry
        # prev_density_reconstruction_inverse = F.pad(prev_flow_inverse[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow_inverse[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow_inverse[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow_inverse[0,3,:,1:],(0,1,0,0))+prev_flow_inverse[0,4,:,:]+F.pad(prev_flow_inverse[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow_inverse[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow_inverse[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow_inverse[0,8,:-1,:-1],(1,0,1,0))+prev_flow_inverse[0,9,:,:]*mask_boundry
        # post_density_reconstruction_inverse = torch.sum(post_flow_inverse[0,:9,:,:],dim=0)+post_flow_inverse[0,9,:,:]*mask_boundry
        # post_density_reconstruction = F.pad(post_flow[0,0,1:,1:],(0,1,0,1))+F.pad(post_flow[0,1,1:,:],(0,0,0,1))+F.pad(post_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(post_flow[0,3,:,1:],(0,1,0,0))+post_flow[0,4,:,:]+F.pad(post_flow[0,5,:,:-1],(1,0,0,0))+F.pad(post_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(post_flow[0,7,:-1,:],(0,0,1,0))+F.pad(post_flow[0,8,:-1,:-1],(1,0,1,0))+post_flow[0,9,:,:]*mask_boundry

        loss_prev_flow = criterion(reconstruction_from_prev, target)
        loss_post_flow = criterion(reconstruction_from_post, target)
        loss_prev_flow_inverse = criterion(reconstruction_from_prev_inverse, target)
        loss_post_flow_inverse = criterion(reconstruction_from_post_inverse, target)

        # cycle consistency
        loss_prev_consistency = criterion(prev_flow[0,0,1:,1:], prev_flow_inverse[0,8,:-1,:-1])+criterion(prev_flow[0,1,1:,:], prev_flow_inverse[0,7,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow_inverse[0,6,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow_inverse[0,5,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow_inverse[0,3,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow_inverse[0,2,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow_inverse[0,1,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow_inverse[0,0,1:,1:])
        loss_post_consistency = criterion(post_flow[0,0,1:,1:], post_flow_inverse[0,8,:-1,:-1])+criterion(post_flow[0,1,1:,:], post_flow_inverse[0,7,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow_inverse[0,6,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow_inverse[0,5,:,:-1])+criterion(post_flow[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow_inverse[0,3,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow_inverse[0,2,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow_inverse[0,1,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow_inverse[0,0,1:,1:])


        loss = loss_prev_flow+loss_post_flow+loss_prev_flow_inverse+loss_post_flow_inverse+loss_prev_consistency+loss_post_consistency

        # direct loss
        if dloss_on:
            loss_prev_direct = criterion(prev_flow[0,0,1:,1:], prev_flow[0,0,1:,1:])+criterion(prev_flow[0,1,1:,:], prev_flow[0,1,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow[0,2,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow[0,3,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow[0,5,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow[0,6,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow[0,7,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow[0,8,1:,1:])
            loss_post_direct = criterion(post_flow[0,0,1:,1:], post_flow[0,0,1:,1:])+criterion(post_flow[0,1,1:,:], post_flow[0,1,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow[0,2,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow[0,3,:,:-1])+criterion(post_flow[0,4,:,:], post_flow[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow[0,5,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow[0,6,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow[0,7,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow[0,8,1:,1:])
            loss_prev_inv_direct = criterion(prev_flow_inverse[0,0,1:,1:], prev_flow_inverse[0,0,1:,1:])+criterion(prev_flow_inverse[0,1,1:,:], prev_flow_inverse[0,1,:-1,:])+criterion(prev_flow_inverse[0,2,1:,:-1], prev_flow_inverse[0,2,:-1,1:])+criterion(prev_flow_inverse[0,3,:,1:], prev_flow_inverse[0,3,:,:-1])+criterion(prev_flow_inverse[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow_inverse[0,5,:,:-1], prev_flow_inverse[0,5,:,1:])+criterion(prev_flow_inverse[0,6,:-1,1:], prev_flow_inverse[0,6,1:,:-1])+criterion(prev_flow_inverse[0,7,:-1,:], prev_flow_inverse[0,7,1:,:])+criterion(prev_flow_inverse[0,8,:-1,:-1], prev_flow_inverse[0,8,1:,1:])
            loss_post_inv_direct = criterion(post_flow_inverse[0,0,1:,1:], post_flow_inverse[0,0,1:,1:])+criterion(post_flow_inverse[0,1,1:,:], post_flow_inverse[0,1,:-1,:])+criterion(post_flow_inverse[0,2,1:,:-1], post_flow_inverse[0,2,:-1,1:])+criterion(post_flow_inverse[0,3,:,1:], post_flow_inverse[0,3,:,:-1])+criterion(post_flow_inverse[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow_inverse[0,5,:,:-1], post_flow_inverse[0,5,:,1:])+criterion(post_flow_inverse[0,6,:-1,1:], post_flow_inverse[0,6,1:,:-1])+criterion(post_flow_inverse[0,7,:-1,:], post_flow_inverse[0,7,1:,:])+criterion(post_flow_inverse[0,8,:-1,:-1], post_flow_inverse[0,8,1:,1:])

            loss += float(args.myloss) *(loss_prev_direct + loss_post_direct + loss_prev_inv_direct + loss_post_inv_direct)

        # MAE
        overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).type(torch.FloatTensor)
        mae += abs(overall.data.sum()-target.sum())

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        del prev_img
        del img
        del post_img
        del target

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            with open(os.path.join(args.savefolder, 'log.txt'), mode='a') as f:
                f.write('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t\n'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    mae = mae/len(train_loader)
    print(' * Train MAE {mae:.3f} '
              .format(mae=mae))
    print(' * Train Loss {loss:.3f} '
              .format(loss=losses.avg))
    with open(os.path.join(args.savefolder, 'log.txt'), mode='a') as f:
        f.write('Train MAE:{mae:.3f} \nTrain Loss:{loss:.3f} \n\n'
              .format(mae=mae, loss=losses.avg))

def validate(val_list, model, criterion, device):
    global args
    print ('begin val')
    with open(os.path.join(args.savefolder, 'log.txt'), mode='a') as f:
        f.write('begin val\n')
    val_dataset = dataset_factory(val_list, args, mode="val")
    val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1)

    model.eval()

    losses = AverageMeter()
    mae = 0

    for i,(prev_img, img, post_img, target ) in enumerate(val_loader):
        # only use previous frame in inference time, as in real-time application scenario, future frame is not available
        with torch.no_grad():
            prev_img = prev_img.to(device, dtype=torch.float)
            prev_img = Variable(prev_img)

            img = img.to(device, dtype=torch.float)
            img = Variable(img)

            post_img = post_img.to(device, dtype=torch.float)
            post_img = Variable(post_img)

            prev_flow = model(prev_img,img)
            prev_flow_inverse = model(img,prev_img)

            post_flow = model(img, post_img)
            post_flow_inverse = model(post_img, img)

            target = target.type(torch.FloatTensor)[0].to(device, dtype=torch.float)
            target = Variable(target)

            mask_boundry = torch.zeros(prev_flow.shape[2:])
            mask_boundry[0,:] = 1.0
            mask_boundry[-1,:] = 1.0
            mask_boundry[:,0] = 1.0
            mask_boundry[:,-1] = 1.0

            mask_boundry = Variable(mask_boundry.cuda())


            reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
            reconstruction_from_post = torch.sum(post_flow[0,:9,:,:],dim=0)+post_flow[0,9,:,:]*mask_boundry
            reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry
            reconstruction_from_post_inverse = F.pad(post_flow_inverse[0,0,1:,1:],(0,1,0,1))+F.pad(post_flow_inverse[0,1,1:,:],(0,0,0,1))+F.pad(post_flow_inverse[0,2,1:,:-1],(1,0,0,1))+F.pad(post_flow_inverse[0,3,:,1:],(0,1,0,0))+post_flow_inverse[0,4,:,:]+F.pad(post_flow_inverse[0,5,:,:-1],(1,0,0,0))+F.pad(post_flow_inverse[0,6,:-1,1:],(0,1,1,0))+F.pad(post_flow_inverse[0,7,:-1,:],(0,0,1,0))+F.pad(post_flow_inverse[0,8,:-1,:-1],(1,0,1,0))+post_flow_inverse[0,9,:,:]*mask_boundry

            overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).type(torch.FloatTensor)

            loss_prev_flow = criterion(reconstruction_from_prev, target)
            loss_post_flow = criterion(reconstruction_from_post, target)
            loss_prev_flow_inverse = criterion(reconstruction_from_prev_inverse, target)
            loss_post_flow_inverse = criterion(reconstruction_from_post_inverse, target)

            # cycle consistency
            loss_prev_consistency = criterion(prev_flow[0,0,1:,1:], prev_flow_inverse[0,8,:-1,:-1])+criterion(prev_flow[0,1,1:,:], prev_flow_inverse[0,7,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow_inverse[0,6,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow_inverse[0,5,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow_inverse[0,3,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow_inverse[0,2,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow_inverse[0,1,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow_inverse[0,0,1:,1:])
            loss_post_consistency = criterion(post_flow[0,0,1:,1:], post_flow_inverse[0,8,:-1,:-1])+criterion(post_flow[0,1,1:,:], post_flow_inverse[0,7,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow_inverse[0,6,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow_inverse[0,5,:,:-1])+criterion(post_flow[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow_inverse[0,3,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow_inverse[0,2,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow_inverse[0,1,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow_inverse[0,0,1:,1:])


            loss = loss_prev_flow+loss_post_flow+loss_prev_flow_inverse+loss_post_flow_inverse+loss_prev_consistency+loss_post_consistency

            if dloss_on:
                loss_prev_direct = criterion(prev_flow[0,0,1:,1:], prev_flow[0,0,1:,1:])+criterion(prev_flow[0,1,1:,:], prev_flow[0,1,:-1,:])+criterion(prev_flow[0,2,1:,:-1], prev_flow[0,2,:-1,1:])+criterion(prev_flow[0,3,:,1:], prev_flow[0,3,:,:-1])+criterion(prev_flow[0,4,:,:], prev_flow[0,4,:,:])+criterion(prev_flow[0,5,:,:-1], prev_flow[0,5,:,1:])+criterion(prev_flow[0,6,:-1,1:], prev_flow[0,6,1:,:-1])+criterion(prev_flow[0,7,:-1,:], prev_flow[0,7,1:,:])+criterion(prev_flow[0,8,:-1,:-1], prev_flow[0,8,1:,1:])
                loss_post_direct = criterion(post_flow[0,0,1:,1:], post_flow[0,0,1:,1:])+criterion(post_flow[0,1,1:,:], post_flow[0,1,:-1,:])+criterion(post_flow[0,2,1:,:-1], post_flow[0,2,:-1,1:])+criterion(post_flow[0,3,:,1:], post_flow[0,3,:,:-1])+criterion(post_flow[0,4,:,:], post_flow[0,4,:,:])+criterion(post_flow[0,5,:,:-1], post_flow[0,5,:,1:])+criterion(post_flow[0,6,:-1,1:], post_flow[0,6,1:,:-1])+criterion(post_flow[0,7,:-1,:], post_flow[0,7,1:,:])+criterion(post_flow[0,8,:-1,:-1], post_flow[0,8,1:,1:])
                loss_prev_inv_direct = criterion(prev_flow_inverse[0,0,1:,1:], prev_flow_inverse[0,0,1:,1:])+criterion(prev_flow_inverse[0,1,1:,:], prev_flow_inverse[0,1,:-1,:])+criterion(prev_flow_inverse[0,2,1:,:-1], prev_flow_inverse[0,2,:-1,1:])+criterion(prev_flow_inverse[0,3,:,1:], prev_flow_inverse[0,3,:,:-1])+criterion(prev_flow_inverse[0,4,:,:], prev_flow_inverse[0,4,:,:])+criterion(prev_flow_inverse[0,5,:,:-1], prev_flow_inverse[0,5,:,1:])+criterion(prev_flow_inverse[0,6,:-1,1:], prev_flow_inverse[0,6,1:,:-1])+criterion(prev_flow_inverse[0,7,:-1,:], prev_flow_inverse[0,7,1:,:])+criterion(prev_flow_inverse[0,8,:-1,:-1], prev_flow_inverse[0,8,1:,1:])
                loss_post_inv_direct = criterion(post_flow_inverse[0,0,1:,1:], post_flow_inverse[0,0,1:,1:])+criterion(post_flow_inverse[0,1,1:,:], post_flow_inverse[0,1,:-1,:])+criterion(post_flow_inverse[0,2,1:,:-1], post_flow_inverse[0,2,:-1,1:])+criterion(post_flow_inverse[0,3,:,1:], post_flow_inverse[0,3,:,:-1])+criterion(post_flow_inverse[0,4,:,:], post_flow_inverse[0,4,:,:])+criterion(post_flow_inverse[0,5,:,:-1], post_flow_inverse[0,5,:,1:])+criterion(post_flow_inverse[0,6,:-1,1:], post_flow_inverse[0,6,1:,:-1])+criterion(post_flow_inverse[0,7,:-1,:], post_flow_inverse[0,7,1:,:])+criterion(post_flow_inverse[0,8,:-1,:-1], post_flow_inverse[0,8,1:,1:])

                loss += float(args.myloss) *(loss_prev_direct + loss_post_direct + loss_prev_inv_direct + loss_post_inv_direct)

            target = target.type(torch.FloatTensor)

            losses.update(loss.item(), img.size(0))
            mae += abs(overall.data.sum()-target.sum())

        del prev_img
        del img
        del target

    mae = mae/len(val_loader)
    print(' * Val MAE {mae:.3f} '
              .format(mae=mae))
    print(' * Val Loss {loss:.3f} '
              .format(loss=losses.avg))
    with open(os.path.join(args.savefolder, 'log.txt'), mode='a') as f:
        f.write('Val MAE:{mae:.3f} \nVal Loss:{loss:.3f} \n\n'
              .format(mae=mae, loss=losses.avg))

    return mae

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
