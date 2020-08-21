import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import torch.autograd.Variable as Variable
from eval import *
from itertools import chain



class operate:
    def __init__(self, gpu=0, batchsize=1, structure='wgan-gp', optimizer_D=None, optimizer_G=None, lambda_idt=0.5, lambda_A=10.0, lambda_B=10.0):
        self.gpu = gpu
        self.NUM = 0
        self.lambda_idt = lambda_idt
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        self.LAMBDA = 10
        self.batchsize = batchsize
        self.MYLoss = MYevaluate(self.gpu)
        self.structure = structure
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G

    def loadon_model(self, model_D, model_G):
        if self.structure == 'cyclegan':
            assert len(model_G) == 2, 'please give two model_G'
            self.model_D_A = model_D[0]
            self.model_D_B = model_D[1]
            self.model_G_A = model_G[0]
            self.model_G_B = model_G[1]
        else:
            self.model_G = model_G
            self.model_D = model_D

    def train_process(self, inputs=None, targets=None, step=0):
        if self.structure == 'cyclegan':
            Dloss_A = self.stepD(self.model_D_A, self.model_G_A, inputs, targets, self.optimizer_D)
            Dloss_B = self.stepD(self.model_D_B, self.model_G_B, inputs, targets, self.optimizer_D)
            self.Dloss = (Dloss_A+Dloss_B)
            parametersD = chain(self.model_D_A.parameters(), self.model_D_B.parameters())
        else:
            self.Dloss = self.stepD(self.model_D, self.model_G, inputs, targets,self.optimizer_D)
            parametersD = self.model_D.parameters()
        self.optimizer_D.step()
        if step % 1 == 0:
            for p in parametersD:
                p.requires_grad_(False)
            self.Gloss = self.stepG(inputs, targets, self.optimizer_G)
            self.optimizer_G.step()
            for p in parametersD:
                p.requires_grad_(True)

    def calc_gradient_penalty(self, real, fake, model_D):
        alpha = torch.rand((real.size(0), 1)).unsqueeze_(-1).unsqueeze_(-1)
        # print(alpha.size(), real.size())
        alpha = alpha.expand(real.size())
        alpha = alpha.cuda(self.gpu)
        interpolates = alpha * real.detach() + ((1-alpha)*fake.detach())
        interpolates.requires_grad_(True)
        interpolates = interpolates.cuda(self.gpu)
        disc = model_D(interpolates)
        grad = torch.autograd.grad(outputs=disc, inputs=interpolates,
                                   grad_outputs=torch.ones(disc.size()).cuda(self.gpu),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad = grad.view(grad.size(0), -1)
        gradient = ((grad.norm(2,dim=1)-1)**2).mean()*self.LAMBDA
        return gradient


    def stepD(self, model_D, model_G, inputs, targets, optimizer_D):
        optimizer_D.zero_grad()
        # noise = torch.randn((self.batchsize, 128))
        # noise = noise.cuda(self.gpu)
        # noisev = Variable(noise)

        real_data_v = targets
        fake = model_G(inputs).detach()
        inputv = fake

        #train with real
        Dreal = model_D(real_data_v)

        Dreal = Dreal.mean()
        # Dreal.backward(self.mone)


        #train with fake
        Dfake = model_D(inputv)
        Dfake = Dfake.mean()
        # Dfake.backward(self.one)

        gradient = self.calc_gradient_penalty(real_data_v.data, fake.data, model_D)
        # gradient.backward()

        D_loss = Dfake - Dreal + gradient
        D_loss.backward()
        return D_loss
        # print(Dreal, Dfake, gradient)
        # self.Wasserstain = Dreal - Dfake

    def stepG_Base(self, model_D, model_G, inputs, targets, optimizer_G):
        # noise = torch.randn((self.batchsize, 128))
        # noise = noise.cuda(self.gpu)
        # noisev = Variable(noise)
        fake = model_G(inputs)
        G = model_D(fake)
        G = G.mean()
        calloss = self.MYLoss(fake*2048.0+2048.0, targets*2048.0+2048.0)
        calloss = calloss - G
        return calloss, fake

    def stepG(self, inputs, targets, optimizer_G):
        optimizer_G.zero_grad()
        if self.structure != 'cyclegan':
            G_loss, _ = self.stepG_Base(self.model_D, self.model_G, inputs, targets, optimizer_G)
        else:
            lossA_B, fakeA = self.stepG_Base(self.model_D_A, self.model_G_A, inputs, targets, optimizer_G)
            lossB_A, fakeB = self.stepG_Base(self.model_D_B, self.model_G_B, inputs, targets, optimizer_G)
            recB = self.model_G_B(fakeA)
            recA = self.model_G_A(fakeB)
            idtA = self.model_G_A(inputs)
            idtB = self.model_G_B(targets)
            G_loss = self.MYLoss.forward_cycle(inputs, targets, recA, recB, idtA, idtB, lossA_B, lossB_A, self.lambda_idt, self.lambda_A, self.lambda_B)
        G_loss.backward()
        return G_loss

    def eval(self, inputs, targets):
        real_data_v = targets

        if self.structure == 'cyclegan':
            fake = self.model_G_A(inputs).data
            Dreal = self.model_D_A(real_data_v)
            Dfake = self.model_D_A(fake)
            G = self.model_D_A(fake)
        else:
            fake = self.model_G(inputs).data
            Dreal = self.model_D(real_data_v)
            Dfake = self.model_D(fake)
            G = self.model_D(fake)
        Dreal = Dreal.mean()

        Dfake = Dfake.mean()
        self.Dloss= Dreal - Dfake

        G = G.mean()
        self.Gloss = self.MYLoss(fake*2048.0+2048.0, targets*2048.0+2048.0)-G

    def get_result(self,mode='train'):
        if mode == 'train':
            return self.Dloss.item(), self.Gloss.item()
        else:
            return self.Dloss.item(), self.Gloss.item()

    def generate_img(self, input, target,svimg_path=None):
        if self.structure == 'cyclegan':
            output = self.model_G_A(input)
        else:
            output = self.model_G(input)
        for i in range(output.size(0)):
            input_np = input[i].squeeze().detach().cpu().numpy()

            generate_np = output[i].squeeze().detach().cpu().numpy()

            target_np = target[i].squeeze().detach().cpu().numpy()
            psnr_ot, ssim_ot, psnr_it, ssim_it = get_evaluate(input_np, generate_np, target_np)
            if svimg_path is None:
                plt.figure()
                self.show_img(input_np*2048.0+2048.0, 'input', subnum=1)
                self.show_img(generate_np*2048.0+2048.0, 'generate', subnum=2)
                self.show_img(target_np*2048.0+2048.0, 'target', subnum=3)
                plt.show()
            else:
                path = os.path.join(svimg_path, str(self.NUM))
                if not os.path.exists(path):
                    os.mkdir(path)
                self.save_img(input_np*2048.0+2048.0, os.path.join(path, 'input.jpg'))
                self.save_img(generate_np*2048.0+2048.0, os.path.join(path, 'generate.jpg'))
                self.save_img(target_np*2048.0+2048.0, os.path.join(path, 'target.jpg'))
                self.save_eval(psnr_ot, ssim_ot, psnr_it, ssim_it, os.path.join(path, 'result.txt'))
                self.NUM += 1

    def show_img(self, image, title='title', subnum=None):
        plt.subplot(1, 3, subnum)
        plt.imshow(image,vmin=0, vmax=4096, cmap='gray')
        plt.title(title)


    def save_img(self,image,image_path='image.jpg'):
        plt.imsave(image_path, image,vmin=0, vmax=4096, cmap="gray")

    def save_eval(self, psnr_it, ssim_it, psnr_ot, ssim_ot, file):
        with open(file, 'w') as f:
            f.write('psnr input & target:   '+str(psnr_it)+'\n')
            f.write('ssim input & target:   '+str(ssim_it)+'\n')
            f.write('psnr output & target:   '+str(psnr_ot)+'\n')
            f.write('ssim output & target:   '+str(ssim_ot)+'\n')
