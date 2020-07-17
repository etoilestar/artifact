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


def train(model_D, model_G, train_dataloader, epoch, optimizer_D, optimizer_G, writer, op):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    D_costs = AverageMeter()
    Wasserstains = AverageMeter()
    G_costs = AverageMeter()


    model_D.train()#指定train模式，不可漏掉
    model_G.train()

    end = time.time()
    for step, (inputs, targets) in enumerate(tqdm(train_dataloader)):#分批次读取数据
        data_time.update(time.time() - end)

        #转到gpu
        inputs = inputs.float().cuda(params['gpu'][0])
        targets = targets.float().cuda(params['gpu'][0])

        # for _ in range(5):
        op.stepD(optimizer_D, inputs, targets)
        # if step % 5 == 0:
        for p in model_D.parameters():
            p.requires_grad_(False)
        op.stepG(optimizer_G, inputs)
        for p in model_D.parameters():
            p.requires_grad_(True)

        loss, D_cost, Wasserstain, G_cost = op.get_result()

        losses.update(loss)
        D_costs.update(D_cost)
        Wasserstains.update(Wasserstain)
        G_costs.update(G_cost)

        # if step == 200:
        #     op.generate_img(inputs, targets)
            # exit()

        batch_time.update(time.time() - end)
        end = time.time()
    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    return losses.avg, D_costs.avg, Wasserstains.avg, G_costs.avg, data_time.avg, batch_time.avg


def validation(model_D, model_G, val_dataloader, epoch, optimizer_D, optimizer_G, writer, op, test_only = False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Wasserstains = AverageMeter()
    G_costs = AverageMeter()
    model_D.eval()#指定eval模式，不可漏掉，验证，测试都要写
    model_G.eval()

    end = time.time()
    with torch.no_grad():#验证，测试模式下不累计梯度
        for step, (inputs, targets) in enumerate(tqdm(val_dataloader)):
            data_time.update(time.time() - end)
            inputs = inputs.float().cuda(params['gpu'][0])
            targets = targets.float().cuda(params['gpu'][0])
            if test_only:
                op.generate_img(inputs, targets)
                continue
            op.eval(inputs, targets)

            loss, Wasserstain, G_cost = op.get_result(mode='eval')

            losses.update(loss)
            Wasserstains.update(Wasserstain)
            G_costs.update(G_cost)

            batch_time.update(time.time() - end)
            end = time.time()
    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    return losses.avg, Wasserstains.avg, G_costs.avg, data_time.avg, batch_time.avg

def main():
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    #定义训练集和验证集的dataloader
    train_dataloader = \
        DataLoader(
            MYdataloader(path=params['train_path'],mode='train'),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    val_dataloader = \
        DataLoader(
            MYdataloader(path=params['train_path'], mode='valid'),
            batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

#    model = MYmodel()#也可以使用别人定义好的模型库，如：
#     model = model.cuda(params['gpu'][0])#将模型转到gpu
#     model = nn.DataParallel(model, device_ids=params['gpu'])  # 多gpu并行

    model_D = Discriminator()
    model_G = Generator()
    model_D = model_D.cuda(params['gpu'][0])#将模型转到gpu
    model_D = nn.DataParallel(model_D, device_ids=params['gpu'])  # 多gpu并行
    model_G = model_G.cuda(params['gpu'][0])#将模型转到gpu
    model_G = nn.DataParallel(model_G, device_ids=params['gpu'])  # 多gpu并行

    #加载预训练参数，可以是之前训练中停的参数
    if params['pretrained'] is not None:
        pretrained_dict_G = torch.load(os.path.join(params['pretrained'], 'checkpoint_G.pth'), map_location='cpu')
        pretrained_dict_D = torch.load(os.path.join(params['pretrained'], 'checkpoint_D.pth'), map_location='cpu')
        # try:
        #     model_dict_D = model_G.module.state_dict()
        #     model_dict_G = model_G.module.state_dict()
        # except AttributeError:
        model_dict_D = model_D.state_dict()
        model_dict_G = model_G.state_dict()
        print(model_dict_G.keys())
        print(pretrained_dict_G.keys())
        pretrained_dict_D = {k: v for k, v in pretrained_dict_D.items() if k in model_dict_D}
        pretrained_dict_G = {k: v for k, v in pretrained_dict_G.items() if k in model_dict_G}
        model_dict_G.update(pretrained_dict_G)
        model_G.load_state_dict(model_dict_G)
        model_dict_D.update(pretrained_dict_D)
        model_D.load_state_dict(model_dict_D)

    #如果是类似于图像分类简单的任务，只有单个参数，可以用简单的官方定义好的loss计算函数，如：
    #criterion = nn.CrossEntropyLoss()
    # criterion = MYevaluate()
    #使用动量梯度下降，或者使用ADAM
    #optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    optimizer_D = optim.Adam(model_D.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    optimizer_G = optim.Adam(model_G.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    #定义优化器
    op = operate(model_D, model_G)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min',factor=0.6, patience=6, verbose=True)
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min',factor=0.6, patience=6, verbose=True)
    model_save_dir = os.path.join(params['save_path'], cur_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('\n')
    print('-----------------start training----------------------------')
    if params['test_only']:
        validation(model_D, model_G, val_dataloader, 0, optimizer_D, optimizer_G, writer, op, test_only = True)
        exit()
    for epoch in range(params['epoch_num']):
        train_loss, trainDC, tWasserstain, trainGC, train_data_time, train_batch_time = train(model_D, model_G, train_dataloader, epoch, optimizer_D, optimizer_G, writer, op)
        valid_loss, vWasserstain, validGC, valid_data_time, valid_batch_time = validation(model_D, model_G, val_dataloader, epoch, optimizer_D, optimizer_G, writer, op)
        # train_loss, train_data_time, train_batch_time = train(model, train_dataloader, epoch, criterion, optimizer, writer)
        # valid_loss, valid_data_time, valid_batch_time = validation(model, val_dataloader, epoch, criterion, optimizer, writer)
        #训练过程中通过不同的学习策略调节学习率
        scheduler_D.step(valid_loss)
        scheduler_G.step(valid_loss)
        #打印结果
        print('\n')
        print('-----------------------------------------')
        print('epoch:', str(epoch + 1) + "/" + str(params['epoch_num']))
        print('train loss:%0.4f'%train_loss, 'lr:', optimizer_G.param_groups[0]['lr'])
        print('trainDC:%0.4f'%trainDC, 'train wasserstain:%0.4f'%tWasserstain, 'trainGC:%0.4f'%trainGC)
        print('train_data_time:%0.4f'%train_data_time, 'train_batch_time:%0.4f'%train_batch_time)
        print('valid loss:%0.4f'%valid_loss)
        print('valid wasserstain:%0.4f'%vWasserstain, 'validGC:%0.4f'%validGC)
        print('valid_data_time:%0.4f'%valid_data_time, 'valid_batch_time:%0.4f'%valid_batch_time)

        checkpoint_D = os.path.join(model_save_dir,"checkpoint_D.pth")
        torch.save(model_D.state_dict(), checkpoint_D)
        checkpoint_G = os.path.join(model_save_dir,"checkpoint_G.pth")
        torch.save(model_G.state_dict(), checkpoint_G)

    writer.close


if __name__ == '__main__':
    main()
