import os
import torch
from torch import nn, optim
from itertools import chain

class Play_model:
    def __init__(self, Discriminator, Generator, gpu=[0], structure='wgan-gp'):
        self.structure = structure
        self.gpu = gpu
        if structure == 'cyclegan':
            self.model_D_A = self.define_model(Discriminator)
            self.model_D_B = self.define_model(Discriminator)
            self.model_G_A = self.define_model(Generator)
            self.model_G_B = self.define_model(Generator)
        else:
            self.model_D = self.define_model(Discriminator)
            self.model_G = self.define_model(Generator)

    def define_model(self, net):
        model = net()
        model = model.cuda(self.gpu[0])
        model = nn.DataParallel(model, device_ids=self.gpu)
        return model

    def load_dict(self, model, ckp_path):
        pretrained_dict = torch.load(ckp_path, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def load(self, pretrained_path=None):
        assert pretrained_path is not None

        if self.structure == 'cyclegan':
            self.model_D_A = self.load_dict(self.model_D_A, os.path.join(pretrained_path, 'checkpoint_D_A.pth'))
            self.model_D_B = self.load_dict(self.model_D_B, os.path.join(pretrained_path, 'checkpoint_D_B.pth'))
            self.model_G_A = self.load_dict(self.model_G_A, os.path.join(pretrained_path, 'checkpoint_G_A.pth'))
            self.model_G_B = self.load_dict(self.model_G_B, os.path.join(pretrained_path, 'checkpoint_G_B.pth'))

        else:
            self.model_D = self.load_dict(self.model_D, os.path.join(pretrained_path, 'checkpoint_D.pth'))
            self.model_G = self.load_dict(self.model_G, os.path.join(pretrained_path, 'checkpoint_G.pth'))


    def return_model(self):
        if self.structure == 'cyclegan':
            return [self.model_D_A,self.model_D_B], [self.model_G_A, self.model_G_B]
        else:
            return self.model_D, self.model_G


    def return_optim(self,lr_d, lr_g, weight_decay=1e-5):
        if self.structure == 'cyclegan':
            optimizer_D = optim.Adam(chain(self.model_D_A.parameters(),self.model_D_B.parameters()), lr=lr_d, betas=(0.5, 0.999),
                                     weight_decay=weight_decay)
            optimizer_G = optim.Adam(chain(self.model_G_A.parameters(),self.model_G_B.parameters()), lr=lr_g, betas=(0.5, 0.999),
                                     weight_decay=weight_decay)
        else:
            optimizer_D = optim.Adam(self.model_D.parameters(), lr=lr_d, betas=(0.5, 0.999),
                                     weight_decay=weight_decay)
            optimizer_G = optim.Adam(self.model_G.parameters(), lr=lr_g, betas=(0.5, 0.999),
                                     weight_decay=weight_decay)
        return optimizer_D, optimizer_G

    def save(self,model_save_dir):
        if self.structure == 'cyclegan':
            checkpoint_D_A = os.path.join(model_save_dir, "checkpoint_D_A.pth")
            torch.save(self.model_D_A.state_dict(), checkpoint_D_A)
            checkpoint_D_B = os.path.join(model_save_dir, "checkpoint_D_B.pth")
            torch.save(self.model_D_B.state_dict(), checkpoint_D_B)
            checkpoint_G_A = os.path.join(model_save_dir, "checkpoint_G_A.pth")
            torch.save(self.model_G_A.state_dict(), checkpoint_G_A)
            checkpoint_G_B = os.path.join(model_save_dir, "checkpoint_G_B.pth")
            torch.save(self.model_G_B.state_dict(), checkpoint_G_B)
        else:
            checkpoint_D = os.path.join(model_save_dir, "checkpoint_D.pth")
            torch.save(self.model_D.state_dict(), checkpoint_D)
            checkpoint_G = os.path.join(model_save_dir, "checkpoint_G.pth")
            torch.save(self.model_G.state_dict(), checkpoint_G)

