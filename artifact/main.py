import os
import time
import numpy as np
import torch
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import MYdataloader
from model import MYmodel
from tqdm import tqdm
from eval import *
from tensorboardX import SummaryWriter#使用tensorboardX查看训练进程
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

def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    model.train()#指定train模式，不可漏掉
    end = time.time()
    for step, (inputs, labels) in enumerate(tqdm(train_dataloader)):#分批次读取数据
        data_time.update(time.time() - end)

        #转到gpu
        inputs = inputs.float().cuda(params['gpu'][0])
        labels = labels.float().cuda(params['gpu'][0])
        outputs = model(inputs)

        #计算loss值
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        batch_time.update(time.time() - end)
        end = time.time()
    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    return losses.avg, data_time.avg, batch_time.avg

def validation(model, val_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.eval()#指定eval模式，不可漏掉，验证，测试都要写

    end = time.time()
    with torch.no_grad():#验证，测试模式下不累计梯度
        for step, (inputs, labels) in enumerate(tqdm(val_dataloader)):
            data_time.update(time.time() - end)
            inputs = inputs.float().cuda(params['gpu'][0])
            labels = labels.float().cuda(params['gpu'][0])
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss

            losses.update(loss.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    return losses.avg, data_time.avg, batch_time.avg

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

    model = MYmodel()#也可以使用别人定义好的模型库，如：
#    model = resnet.resnet50(num_classes=params['num_classes'])
#    model = EfficientNet.from_name(params['pretrained'], data=Data, override_params={'num_classes': params['num_classes']})

    model = model.cuda(params['gpu'][0])#将模型转到gpu
    model = nn.DataParallel(model, device_ids=params['gpu'])  # 多gpu并行

    #加载预训练参数，可以是之前训练中停的参数
    if params['pretrained'] is not None:
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    #如果是类似于图像分类简单的任务，只有单个参数，可以用简单的官方定义好的loss计算函数，如：
    #criterion = nn.CrossEntropyLoss()
    criterion = MYevaluate()
    #使用动量梯度下降，或者使用ADAM
    #optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    #定义优化器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.6, patience=6, verbose=True)
    model_save_dir = os.path.join(params['save_path'], cur_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('\n')
    print('-----------------start training----------------------------')
    for epoch in range(params['epoch_num']):
        train_loss, train_data_time, train_batch_time = train(model, train_dataloader, epoch, criterion, optimizer, writer)
        valid_loss, valid_data_time, valid_batch_time = validation(model, val_dataloader, epoch, criterion, optimizer, writer)
        #训练过程中通过不同的学习策略调节学习率
        scheduler.step(valid_loss)

        #打印结果
        print('\n')
        print('-----------------------------------------')
        print('epoch:', str(epoch + 1) + "/" + str(params['epoch_num']))
        print('train loss:%0.4f'%train_loss, 'lr:', optimizer.param_groups[0]['lr'])
        print('train_data_time:%0.4f'%train_data_time, 'train_batch_time:%0.4f'%train_batch_time)
        print('valid loss:%0.4f'%valid_loss)
        print('valid_data_time:%0.4f'%valid_data_time, 'valid_batch_time:%0.4f'%valid_batch_time)

        checkpoint = os.path.join(model_save_dir,"checkpoint.pth")
        torch.save(model.state_dict(), checkpoint)

    writer.close

if __name__ == '__main__':
    main()
