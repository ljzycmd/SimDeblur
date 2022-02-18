import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math
from . import utils
from .import correlation


def make_model(args):
    pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    return Flow_PWC(load_pretrain=True, pretrain_fn=pretrain_fn)


class Flow_PWC(nn.Module):
    def __init__(self, load_pretrain=False, pretrain_fn=''):
        super(Flow_PWC, self).__init__()
        self.moduleNetwork = Network()
        print("Creating Flow PWC")

        if load_pretrain:
            # self.moduleNetwork.load_state_dict(torch.load(pretrain_fn))
            # { strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch', file_name='pwc-' + arguments_strModel).items() }
            self.moduleNetwork.load_state_dict({ strKey: tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch').items()})
            print('Loading Flow PWC pretrain model from {}'.format(pretrain_fn))

    def estimate_flow(self, tensorFirst, tensorSecond):
        b, c, intHeight, intWidth = tensorFirst.size()

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorFirst,
                                                                  size=(intPreprocessedHeight, intPreprocessedWidth),
                                                                  mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorSecond,
                                                                   size=(intPreprocessedHeight, intPreprocessedWidth),
                                                                   mode='bilinear', align_corners=False)

        outputFlow = self.moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond)

        tensorFlow = 20.0 * torch.nn.functional.interpolate(input=outputFlow, size=(intHeight, intWidth),
                                                            mode='bilinear', align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tensorFlow

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(x)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border', align_corners=True)  # align_corners=True when torch 1.3 later
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

        return output, mask

    def forward(self, frame_1, frame_2):
        # flow
        flow = self.estimate_flow(frame_1, frame_2)

        # the gradient of frame_2
        _, _, frame_2_grad = utils.calc_grad_sobel(frame_2.detach())

        # warp
        frame_2_grad_warp, _ = self.warp(frame_2_grad, flow.detach())
        frame_2_warp, mask = self.warp(frame_2, flow)

        return frame_2_warp, flow, frame_2_grad_warp, mask


def Backward(tensorInput, tensorFlow):
    Backward_tensorGrid = {}
    Backward_tensorPartial = {}

    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).to(tensorInput)

    if str(tensorFlow.size()) not in Backward_tensorPartial:
        Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones(
            [tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)])

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)
    tensorInput = torch.cat([tensorInput, Backward_tensorPartial[str(tensorFlow.size())]], 1)

    tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(
            Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear',
                                                   padding_mode='zeros', align_corners=True)

    tensorMask = tensorOutput[:, -1:, :, :];
    tensorMask[tensorMask > 0.999] = 1.0;
    tensorMask[tensorMask < 1.0] = 0.0

    return tensorOutput[:, :-1, :, :] * tensorMask


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()

                intPrevious = \
                    [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                        intLevel + 1]
                intCurrent = \
                    [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                        intLevel + 0]

                if intLevel < 6:
                    self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2,
                                                                 kernel_size=4, stride=2, padding=1)
                if intLevel < 6:
                    self.moduleUpfeat = torch.nn.ConvTranspose2d(
                        in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2,
                        padding=1)
                if intLevel < 6:
                    self.dblBackward = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3,
                                    stride=1, padding=1)
                )

            def forward(self, tensorFirst, tensorSecond, objectPrevious):
                tensorFlow = None
                tensorFeat = None

                if objectPrevious is None:
                    tensorFlow = None
                    tensorFeat = None

                    tensorVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, tensorSecond=tensorSecond),
                        negative_slope=0.1, inplace=False)

                    tensorFeat = torch.cat([tensorVolume], 1)

                elif objectPrevious is not None:
                    tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
                    tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

                    tensorVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tensorFirst=tensorFirst,
                                                              tensorSecond=Backward(tensorInput=tensorSecond,
                                                                                    tensorFlow=tensorFlow * self.dblBackward)),
                        negative_slope=0.1, inplace=False)

                    tensorFeat = torch.cat([tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)

                tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
                tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)

                tensorFlow = self.moduleSix(tensorFeat)

                return {
                    'tensorFlow': tensorFlow,
                    'tensorFeat': tensorFeat
                }

        class Refiner(torch.nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )

            def forward(self, tensorInput):
                return self.moduleMain(tensorInput)

        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

        # self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
        objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
        objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
        objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

        return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])
