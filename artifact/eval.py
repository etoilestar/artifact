import skimage
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class MYevaluate(nn.Module):
    def __init__(self, gpu=0, max_pixel=4095.0, F_PSNR=-2.0, F_ssim=-0.2):#定义loss计算方式
        super(MYevaluate, self).__init__()
        self.MSELOSS = nn.MSELoss(reduction='mean')
        self.l1loss = nn.L1Loss(reduction='mean')
        self.max_pixel = max_pixel
        self.F_PSNR = F_PSNR
        self.F_ssim = F_ssim
        self.gpu = gpu

    def forward_cycle(self, realA, realB, recA, recB, idtA, idtB, lossA_B, lossB_A, lambda_idt, lambda_A, lambda_B):
        idt_loss_A = self.l1loss(realA, idtA)
        idt_loss_B = self.l1loss(realB, idtB)
        cycle_loss_A = self.l1loss(realA, recA)
        cycle_loss_B = self.l1loss(realB, recB)
        return lossA_B+lossB_A+lambda_A*cycle_loss_A+lambda_A*lambda_idt*idt_loss_A+lambda_B*cycle_loss_B+lambda_B*lambda_idt*idt_loss_B

    def forward(self, input, target):
        return self.F_PSNR*self.PSNR(target, input)+self.F_ssim*self.MSSIM(target, input)

    def PSNR(self, img1, img2):
        return 10*torch.log10(self.max_pixel**2/self.MSELOSS(img1, img2))

    def Adgauss(self, size=11, sigma=1.5):
        x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        x = torch.tensor(x_data).requires_grad_(False).unsqueeze_(0).unsqueeze_(0).float().cuda(self.gpu)
        y = torch.tensor(y_data).requires_grad_(False).unsqueeze_(0).unsqueeze_(0).float().cuda(self.gpu)
        g = torch.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    def MSSIM(self, img1, img2, k1=0.01, k2=0.02, window_size=11):
        window = self.Adgauss(window_size)
        mu1 = F.conv2d(img1, window, stride=1, padding=1)
        mu2 = F.conv2d(img2, window, stride=1, padding=1)
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, stride=1, padding=1)-mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, stride=1, padding=1)-mu2_sq
        sigma1_2 = F.conv2d(img1*img2, window, stride=1, padding=1)-mu1_mu2

        c1 = (k1*self.max_pixel)**2
        c2 = (k2*self.max_pixel)**2

        ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()

def get_evaluate(inp_np, out_np, tar_np):
    """Computes psnr and ssim"""
    # psnr = 0
    # ssim = 0
    # for i, (inp, out, tar) in enumerate(zip(input, output, target)):
    # inp_np = input.squeeze().detach().cpu().numpy()
    # out_np = output.squeeze().detach().cpu().numpy()
    # tar_np = target.squeeze().detach().cpu().numpy()
    psnr_ot = skimage.measure.compare_psnr(out_np, tar_np, 255)
    ssim_ot = skimage.measure.compare_ssim(out_np, tar_np, 255)
    psnr_it = skimage.measure.compare_psnr(inp_np, tar_np, 255)
    ssim_it = skimage.measure.compare_ssim(inp_np, tar_np, 255)
    return psnr_ot, ssim_ot, psnr_it, ssim_it


