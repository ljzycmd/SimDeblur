import torch
import torch.nn.functional as F
import numpy as np
import math


def get_pixel_value(img, x, y, h, w):
    """
    Args: 
        img: shape(B, H, W, 3)
        x: shape(B, H, W)
        y: shape(B, H, W)
    """
    img = img.permute(0, 3, 1, 2)

    x_ = torch.unsqueeze(x, 3)
    y_ = torch.unsqueeze(y, 3)

    f = torch.cat([x_, y_], 3)
    # f = torch.cat([ f[:, 0:1, :, :] / ((w - 1.0) / 2.0), f[:, 1:2, :, :] / ((h - 1.0) / 2.0) ], 1)
    f = torch.cat([2.0 * f[:, :, :, 0:1] / (w - 1.0) - 1,
                  2.0 * f[:, :, :, 1:2] / (h - 1.0) - 1], 3)

    # indices = torch.stack([b, y, x], 3)

    return torch.nn.functional.grid_sample(input=img, grid=f, mode='nearest', padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)


Grid = {}


def get_pixel_volume(img, flow, pad, H, W, ksize=5):
    """
    Args:
        img: shape(B, 3, H, W)
        flow: shape(B, 2, H, W)
        pad: SHAPE(B, 3, H, W)
    """
    img = img.permute(0, 2, 3, 1)
    flow = flow.permute(0, 2, 3, 1)
    pad = pad.permute(0, 2, 3, 1)

    batch_size = img.size()[0]
    hksize = int(np.floor(ksize/2))

    flow = flow.permute(0, 3, 1, 2)

    # transfer the relative position into absolute position index
    if str(flow.size()) not in Grid:
        x, y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))
        x = torch.unsqueeze(torch.unsqueeze(x.permute(1, 0), 0), 0).type(
            torch.cuda.FloatTensor)
        y = torch.unsqueeze(torch.unsqueeze(y.permute(1, 0), 0), 0).type(
            torch.cuda.FloatTensor)
        grid = torch.cat([x, y], axis=1)
        Grid[str(flow.size())] = grid
        flows = grid + flow
    else:
        flows = Grid[str(flow.size())] + flow

    x = flows[:, 0, :, :]
    y = flows[:, 1, :, :]

    max_y = H - 1
    max_x = W - 1

    img_gray = torch.unsqueeze(
        0.2989*img[:, :, :, 0]+0.5870*img[:, :, :, 1]+0.1140*img[:, :, :, 2], axis=3)
    pad_gray = torch.unsqueeze(
        0.2989*pad[:, :, :, 0]+0.5870*pad[:, :, :, 1]+0.1140*pad[:, :, :, 2], axis=3)
    out = []
    for i in range(-hksize, hksize+1):
        for j in range(-hksize, hksize+1):
            # clip to range [0, H/W] to not violate img boundaries
            x0_ = x.type(torch.cuda.IntTensor) + i  # (b, h, w)
            y0_ = y.type(torch.cuda.IntTensor) + j  # (b, h, w)
            x0 = torch.clamp(x0_, min=0, max=max_x)
            y0 = torch.clamp(y0_, min=0, max=max_y)

            # get pixel value at corner coords
            Ia = get_pixel_value(img_gray, x0, y0, H, W)

            mask_x = torch.lt(x0_, 1.0).type(
                torch.cuda.FloatTensor) + torch.gt(x0_, max_x-1).type(torch.cuda.FloatTensor)
            mask_y = torch.lt(y0_, 1.0).type(
                torch.cuda.FloatTensor) + torch.gt(y0_, max_y-1).type(torch.cuda.FloatTensor)
            mask = torch.gt(mask_x+mask_y, 0).type(torch.cuda.FloatTensor)

            mask = F.pad(mask, (hksize, hksize, hksize, hksize, 0, 0))
            mask = mask[:, hksize-j:hksize-j+H, hksize-i:hksize-i+W]

            # Ia: shape(B, H, W, C)
            Ia = F.pad(Ia, (0, 0, hksize, hksize, hksize, hksize, 0, 0))
            Ia = Ia[:, hksize-j:hksize-j+H, hksize-i:hksize-i+W, :]

            Ia = torch.mul(Ia, 1 - torch.unsqueeze(mask, axis=3)) + \
                torch.mul(pad_gray, torch.unsqueeze(mask, axis=3))
            out.append(Ia)

    out = torch.cat(out, 3)
    # out = tf.reshape(out,[batch_size,H,W,ksize*ksize])
    out = out.permute(0, 3, 1, 2)
    return out
