#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import pdb

import time

import numpy as np

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import drn
import data_transforms as transforms
from utils import *
from drn_seg import DRNSeg
from drn_depth_seg import DRNDepthSeg

from tensorboardX import SummaryWriter
import torchvision.utils as vutils

try:
    from modules import batchnormsync
except ImportError:
    pass



FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

writer = SummaryWriter('logs/34')

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

class SegDepthList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False, return_depth=True):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.depth_list = None
        self.bbox_list = None
        self.return_depth = return_depth
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        if self.depth_list is not None and self.return_depth:
            data.append(Image.open(join(self.data_dir, self.depth_list[index])))
        
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            if self.depth_list is None and self.return_depth:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        depth_path = join(self.list_dir, self.phase + '_depths.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)
        
        if exists(depth_path):
            self.depth_list = [line.strip() for line in open(depth_path, 'r')]
            assert len(self.image_list) == len(self.depth_list)

class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = data[0].size
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        # data = list(self.transforms(*data))
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(data[0].resize((int(w * s), int(h * s)),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


def validate(val_loader, model, criterion, eval_score=None, print_freq=10, epoch=-1, num_classes=19):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    hist = np.zeros((num_classes, num_classes))

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)

        _, pred = torch.max(output, 1)
        pred = pred.cpu().data.numpy()

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        if eval_score is not None:
            score.update(eval_score(output, target_var), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        label = target.cpu().numpy()
        hist += fast_hist(pred.flatten(), label.flatten(), num_classes)

        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))

    ious = per_class_iu(hist) * 100
    mAP = round(np.nanmean(ious), 2)
    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))
    logger.info(' '.join('{:.03f}'.format(i) for i in ious))
    logger.info('mAP: %f', mAP)
    
    writer.add_scalar('val/loss', loss.data[0], epoch)
    writer.add_scalar('val/accuracy', score.avg, epoch)
    writer.add_scalar('val/mAP', mAP, epoch)
    # writer.add_image('val/gt', target_var, epoch)
    # writer.add_image('val/pred', output, epoch)
    # writer.add_image('val/input', input, epoch)

    return score.avg


def train(train_loader, model, criterion_dict, optimizer, epoch,
          eval_score=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    d_cls_losses = AverageMeter()
    d_reg_losses = AverageMeter()
    seg_losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    d_cls_criterion = criterion_dict['d_cls']
    d_reg_criterion = criterion_dict['d_reg']
    seg_criterion = criterion_dict['seg']

    seg_weight = 0
    d_cls_weight = 1
    d_reg_weight = 1

    for i, (input, depth_target, seg_target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # if type(seg_criterion) in [torch.nn.modules.loss.L1Loss,
        #                        torch.nn.modules.loss.MSELoss]:
        #     seg_target = seg_target.float()
        
        depth_target_flt = depth_target.float()

        input = input.cuda()
        depth_target = depth_target.cuda(async=True)
        depth_target_flt = depth_target.cuda(async=True)
        seg_target = seg_target.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        depth_target_var = torch.autograd.Variable(depth_target)
        depth_target_flt_var = torch.autograd.Variable(depth_target_flt)
        seg_target_var = torch.autograd.Variable(seg_target)

        # compute output
        output = model(input_var)[0]
        seg_loss = seg_criterion(output, seg_target_var)
        d_cls_loss = d_cls_criterion(output, depth_target_var)
        d_reg_loss = d_reg_criterion(output, depth_target_flt_var)

        total_loss = seg_weight * seg_loss + d_cls_weight * d_cls_loss + d_reg_weight * d_reg_loss

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        seg_losses.update(seg_loss.data[0], input.size(0))
        d_cls_losses.update(d_cls_loss.data[0], input.size(0))
        d_reg_losses.update(d_reg_loss.data[0], input.size(0))
        if eval_score is not None:
            scores.update(eval_score(output, seg_target_var), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        iteration = epoch * len(train_loader) + i

        # gt_arr = vutils.make_grid(target_var.data, normalize=True, scale_each=True)
        # pred_arr = vutils.make_grid(output.data, normalize=True, scale_each=True)
        # input_arr = vutils.make_grid(input.data, normalize=True, scale_each=True)
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'seg_loss {loss1.val:.4f} ({loss.avg:.4f})\t'
                        'd_cls_loss {loss2.val:.4f} ({loss.avg:.4f})\t'
                        'd_reg_loss {loss3.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss1=seg_losses, loss2=d_cls_losses,
                loss3=d_reg_losses, top1=scores))
            
            writer.add_scalar('train/seg_loss', seg_loss.data[0], iteration)
            writer.add_scalar('train/d_cls_loss', d_cls_loss.data[0], iteration)
            writer.add_scalar('train/d_reg_loss', d_reg_loss.data[0], iteration)
            writer.add_scalar('train/total_loss', total_loss.data[0], iteration)
            writer.add_scalar('train/seg_accuracy', scores.val, iteration)
            # writer.add_image('train/gt', gt_arr, iteration)
            # writer.add_image('train/pred', pred_arr, iteration)
            # writer.add_image('train/input', input_arr, iteration)


def train_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, None,
                          pretrained=True)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()
    criterion = nn.NLLLoss2d(ignore_index=255)

    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []
    if args.random_rotate > 0:
        t.append(transforms.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScale(args.random_scale))
    t.extend([transforms.RandomCrop(crop_size),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              normalize])
    train_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'train', transforms.Compose(t)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=False, drop_last=True
    )
    # val_loader = torch.utils.data.DataLoader(
    #     SegList(data_dir, 'val', transforms.Compose([
    #         transforms.RandomCrop(crop_size),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=1, shuffle=False, num_workers=num_workers,
    #     pin_memory=True, drop_last=False
    # )

    val_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'val', transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=False, drop_last=False
    )

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.optim_parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=accuracy, epoch=-1)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              eval_score=accuracy)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, eval_score=accuracy, epoch=epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = 'checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % 10 == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)


def train_depth_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNDepthSeg(args.arch, seg_classes=args.classes, depth_classes=5, 
                    pretrained_model=None, pretrained=True, train_base=args.train_base,
                    train_seg=args.train_seg, train_depth=args.train.depth)

    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()
    
    if args.train_seg:
        seg_criterion = nn.NLLLoss2d(ignore_index=255)
        seg_criterion.cuda()

    if args.train_depth:
        d_cls_criterion = nn.NLLLoss2d(ignore_index=0)
        d_cls_criterion.cuda()
        d_reg_criterion = nn.MSELoss()
        d_reg_criterion.cuda()
    

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []
    if args.random_rotate > 0:
        t.append(transforms.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScale(args.random_scale))
    t.extend([transforms.RandomHorizontalFlip(),
              transforms.Rescale(0.25),
              transforms.ToTensor(),
              normalize])
    train_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'train', transforms.Compose(t)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=False, drop_last=True
    )
    # val_loader = torch.utils.data.DataLoader(
    #     SegList(data_dir, 'val', transforms.Compose([
    #         transforms.RandomCrop(crop_size),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=1, shuffle=False, num_workers=num_workers,
    #     pin_memory=True, drop_last=False
    # )

    val_loader = torch.utils.data.DataLoader(
        SegList(data_dir, 'val', transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=False, drop_last=False
    )

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.optim_parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, seg_criterion, eval_score=accuracy, epoch=-1)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch,
              eval_score=accuracy)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, eval_score=accuracy, epoch=epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = 'checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % 10 == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)

def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    for iter, (image, label, name) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)
        final = model(image_var)[0]
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    num_scales = len(scales)
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]
        h, w = input_data[0].size()[2:4]
        images = [input_data[0]]
        images.extend(input_data[-num_scales:])
        # pdb.set_trace()
        outputs = []
        for image in images:
            image_var = Variable(image, requires_grad=False, volatile=True)
            final = model(image_var)[0]
            outputs.append(final.data)
        final = sum([resize_4d_tensor(out, w, h) for out in outputs])
        # _, pred = torch.max(torch.from_numpy(final), 1)
        # pred = pred.cpu().numpy()
        pred = final.argmax(axis=1)
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)

def summarize_seg(args):
    input_size = (3, args.crop_size, args.crop_size)
    single_model = DRNDepthSeg(args.arch, seg_classes=args.classes, depth_classes=5, pretrained_model=None, pretrained=False)
    single_model = single_model.cuda()
    single_model.summary(input_size)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    if args.ms:
        dataset = SegListMS(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), scales)
    else:
        dataset = SegList(data_dir, phase, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        mAP = test(test_loader, model, args.classes, save_vis=False,
                   has_gt=phase != 'test' or args.with_gt, output_dir=out_dir)
    logger.info('mAP: %f', mAP)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test', 'summarize'])
    parser.add_argument('-d', '--data-dir', default=None)
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    args = parser.parse_args()

    assert args.data_dir is not None
    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args


def main():
    args = parse_args()
    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)
    elif args.cmd == 'summarize':
        summarize_seg(args)


if __name__ == '__main__':
    main()
