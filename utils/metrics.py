# CMD

import torch
import torch.nn.functional as F
import cv2


def calculate_psnr(img1, img2):
    """
    data range [0, 1]
    """
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    mse = torch.mean((img1 - img2) ** 2, [1, 2, 3])
    # if mse == 0:
    #     return 100
    PIXEL_MAX = 1
    return 20 * torch.mean(torch.log10(PIXEL_MAX / torch.sqrt(mse)))


def calculate_ssim(img1, img2):
    # implemented with pytorch
    assert isinstance(img1, torch.Tensor)
    assert isinstance(img1, torch.Tensor)

    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2

    # img1 = img1.to(torch.float32)
    # img2 = img2.to(torch.float32)
    kernel = gaussian(11, 1.5).to(img1).unsqueeze(1)
    window = kernel.mm(kernel.t()).float().expand(3, 1, 11, 11)

    mu1 = F.conv2d(img1, window, groups = 3)  # valid
    mu2 = F.conv1d(img2, window, groups = 3)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1**2, window, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=3) - mu1_mu2

    # mu1 = F.conv2d(img1, window, padding = 11//2, groups = 3)  # same
    # mu2 = F.conv1d(img2, window, padding = 11//2, groups = 3)
    # mu1_sq = mu1**2
    # mu2_sq = mu2**2
    # mu1_mu2 = mu1 * mu2
    # sigma1_sq = F.conv2d(img1**2, window, padding=11//2, groups=3) - mu1_sq
    # sigma2_sq = F.conv2d(img2**2, window, padding=11//2, groups=3) - mu2_sq
    # sigma12 = F.conv2d(img1 * img2, window, padding=11//2, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def gaussian(window_size, sigma):
    gauss = torch.exp(torch.Tensor([-(x - window_size//2)**2/float(2*sigma**2) for x in range(window_size)]).float())
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = (_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim2(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


if __name__ == "__main__":
    img1 = torch.ones(1, 3, 256, 256)*0.95
    img2 = torch.ones(1, 3, 256, 256) 
    print(ssim2(img1, img2))
    print(ssim(img1, img2))
    print(psnr(img1, img2))