import torch
from config import params
import numpy as np
import matplotlib.pyplot as plt
# import torch.autograd.Variable as Variable


class operate:
    def __init__(self, model_D, model_G):
        self.gpu = params['gpu'][0]
        one = torch.tensor(1, dtype=torch.float).cuda(self.gpu)
        self.LAMBDA = 10
        self.one = one
        self.mone = -1*one
        self.batchsize = params['batch_size']
        self.modelD = model_D
        self.modelG = model_G

    def calc_gradient_penalty(self, real, fake):
        alpha = torch.rand((real.size(0), 1)).unsqueeze_(-1).unsqueeze_(-1)
        # print(alpha.size(), real.size())
        alpha = alpha.expand(real.size())
        alpha = alpha.cuda(self.gpu)
        interpolates = alpha * real.detach() + ((1-alpha)*fake.detach())
        interpolates.requires_grad_(True)
        interpolates = interpolates.cuda(self.gpu)
        disc = self.modelD(interpolates)
        grad = torch.autograd.grad(outputs=disc, inputs=interpolates,
                                   grad_outputs=torch.ones(disc.size()).cuda(self.gpu),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad = grad.view(grad.size(0), -1)
        gradient = ((grad.norm(2,dim=1)-1)**2).mean()*self.LAMBDA
        return gradient


    def stepD(self, optimizer_D, inputs, targets):
        optimizer_D.zero_grad()
        # noise = torch.randn((self.batchsize, 128))
        # noise = noise.cuda(self.gpu)
        # noisev = Variable(noise)

        noisev = inputs
        real_data_v = targets
        fake = self.modelG(noisev).detach()
        inputv = fake

        #train with real
        Dreal = self.modelD(real_data_v)

        Dreal = Dreal.mean()
        self.loss = -Dreal
        # Dreal.backward(self.mone)


        #train with fake
        Dfake = self.modelD(inputv)
        Dfake = Dfake.mean()
        # Dfake.backward(self.one)

        gradient = self.calc_gradient_penalty(real_data_v.data, fake.data)
        # gradient.backward()

        self.D_cost = Dfake - Dreal + gradient
        # print(Dreal, Dfake, gradient)
        self.Wasserstain = Dreal - Dfake
        self.D_cost.backward()
        optimizer_D.step()

    def stepG(self, optimizer_G, inputs):
        # noise = torch.randn((self.batchsize, 128))
        # noise = noise.cuda(self.gpu)
        # noisev = Variable(noise)
        optimizer_G.zero_grad()
        noisev = inputs
        fake = self.modelG(noisev)
        G = self.modelD(fake)
        G = G.mean()
        G.backward(self.mone)
        self.G_cost = -G
        optimizer_G.step()

    def eval(self, inputs, targets):
        noisev = inputs
        real_data_v = targets
        fake = self.modelG(noisev).data
        inputv = fake
        Dreal = self.modelD(real_data_v)
        Dreal = Dreal.mean()
        self.loss = -Dreal
        Dfake = self.modelD(inputv)
        Dfake = Dfake.mean()
        self.Wasserstain = Dreal - Dfake
        G = self.modelD(fake)
        G = G.mean()
        self.G_cost = -G

    def get_result(self,mode='train'):
        if mode == 'train':
            return self.loss.item(), self.D_cost.item(), self.Wasserstain.item(), self.G_cost.item()
        else:
            return self.loss.item(), self.Wasserstain.item(), self.G_cost.item()

    def generate_img(self, input, target=None):
        output = self.modelG(input)
        for i in range(output.size(0)):
            input_np = input[i].squeeze().detach().cpu().numpy()
            plt.imshow(input_np * 255.0, cmap='gray')
            plt.title('input')
            plt.show()
            generate_np = output[i].squeeze().detach().cpu().numpy()
            plt.imshow(generate_np*255.0,cmap='gray')#
            plt.title('generate')
            plt.show()

            if target is not None:
                target_np = target[i].squeeze().detach().cpu().numpy()
                plt.imshow(target_np*255.0,cmap='gray')
                plt.title('target')
                plt.show()




