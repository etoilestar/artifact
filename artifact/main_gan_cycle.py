import os
import time
import numpy as np
import torch
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import MYdataloader
from model import Generator, Discriminator
from tqdm import tqdm
from eval import *
from tensorboardX import SummaryWriter#使用tensorboardX查看训练进程
from operate import *
from playmodel import Play_model
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision.models import * #如果想用官方预训练模型可以这样使用，包含预训练模型
# model = resnet50(pretrained=True)

class AverageMeter(object):
    """计算平均值"""
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


def train(structure, train_dataloader, epoch, optimizer_D, optimizer_G, writer, op, gpu=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Dlosses = AverageMeter()
    Glosses = AverageMeter()

    #
    # model_D.train()#指定train模式，不可漏掉
    # model_G.train()

    end = time.time()
    for step, (inputs, targets) in enumerate(tqdm(train_dataloader)):#分批次读取数据
        data_time.update(time.time() - end)

        #转到gpu
        inputs = inputs.float().cuda(gpu[0])
        targets = targets.float().cuda(gpu[0])

        # for _ in range(5):
        op.train_process(inputs, targets, step)

        Dloss, Gloss = op.get_result()

        Dlosses.update(Dloss)
        Glosses.update(Gloss)

        # if step == 200:
        #     op.generate_img(inputs, targets)
            # exit()

        batch_time.update(time.time() - end)
        end = time.time()
    writer.add_scalar('train_loss_epoch', Glosses.avg, epoch)
    return Dlosses.avg, Glosses.avg, data_time.avg, batch_time.avg


def validation(structure, val_dataloader, epoch, optimizer_D, optimizer_G, writer, op, test_only = False,gpu=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Dlosses = AverageMeter()
    Glosses = AverageMeter()
    # model_D.eval()#指定eval模式，不可漏掉，验证，测试都要写
    # model_G.eval()

    end = time.time()
    with torch.no_grad():#验证，测试模式下不累计梯度
        for step, (inputs, targets) in enumerate(tqdm(val_dataloader)):
            data_time.update(time.time() - end)
            inputs = inputs.float().cuda(gpu[0])
            targets = targets.float().cuda(gpu[0])
            if test_only:
                op.generate_img(inputs, targets, svimg_path=params['svimg_path'])
                continue
            op.eval(inputs, targets)

            Dloss, Gloss = op.get_result(mode='eval')

            Dlosses.update(Dloss)
            Glosses.update(Gloss)


            batch_time.update(time.time() - end)
            end = time.time()
    writer.add_scalar('val_loss_epoch', Glosses.avg, epoch)
    return Dlosses.avg, Glosses.avg, data_time.avg, batch_time.avg

def main():
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    gpu = params['gpu']
    batchsize = params['batch_size']
    pretrained_path = params['pretrained']
    structure = params['structure']
    lr_d = params['lr_d']
    lr_g = params['lr_g']
    weight_decay = params['weight_decay']
    #定义训练集和验证集的dataloader
    train_dataloader = \
        DataLoader(
            MYdataloader(path=params['train_path'],mode='train', debug=False),
            batch_size=batchsize, shuffle=True, num_workers=params['num_workers'])

    val_dataloader = \
        DataLoader(
            MYdataloader(path=params['train_path'], mode='valid', debug=False),
            batch_size=batchsize, shuffle=False, num_workers=params['num_workers'])

    Playmodels = Play_model(Discriminator, Generator, gpu=gpu, structure=structure)

    #加载预训练参数，可以是之前训练中停的参数
    if pretrained_path is not None:
        Playmodels.load(pretrained_path)
        print('load model successfully')

    model_D, model_G = Playmodels.return_model()

    optimizer_D, optimizer_G = Playmodels.return_optim(lr_d, lr_g, weight_decay)
    # scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min',factor=0.6, patience=6, verbose=True)
    # scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min',factor=0.6, patience=6, verbose=True)
    # scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, 10, 1)
    # scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, 10, 1)

    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=10000)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=10000)
    #定义优化器
    op = operate(structure=structure,optimizer_D=optimizer_D, optimizer_G=optimizer_G)
    op.loadon_model(model_D, model_G)

    model_save_dir = os.path.join(params['save_path'], cur_time)

    if params['test_only']:
        print('-----------------start testing----------------------------')
        validation(structure, train_dataloader, 0, optimizer_D, optimizer_G, writer, op, test_only = True, gpu=gpu)
        exit()
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('\n')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print('-----------------start training----------------------------')
    for epoch in range(params['epoch_num']):
        train_Dloss, train_Gloss, train_data_time, train_batch_time = train(structure, train_dataloader, epoch, optimizer_D, optimizer_G, writer, op, gpu=gpu)
        valid_Dloss, valid_Gloss, valid_data_time, valid_batch_time = validation(structure, val_dataloader, epoch, optimizer_D, optimizer_G, writer, op, gpu=gpu)
        #训练过程中通过不同的学习策略调节学习率
        scheduler_D.step()
        scheduler_G.step()
        #打印结果
        print('\n')
        print('-----------------------------------------')
        print('epoch:', str(epoch + 1) + "/" + str(params['epoch_num']))
        print('train Dloss:%0.4f'%train_Dloss, 'train Gloss:%0.4f'%train_Gloss,'lr_g:', optimizer_G.param_groups[0]['lr'], 'lr_d:', optimizer_D.param_groups[0]['lr'])
        print('train_data_time:%0.4f'%train_data_time, 'train_batch_time:%0.4f'%train_batch_time)
        print('valid Dloss:%0.4f'%valid_Dloss, 'valid Gloss:%0.4f'%valid_Gloss)
        print('valid_data_time:%0.4f'%valid_data_time, 'valid_batch_time:%0.4f'%valid_batch_time)

        Playmodels.save(model_save_dir)

    writer.close


if __name__ == '__main__':
    main()
