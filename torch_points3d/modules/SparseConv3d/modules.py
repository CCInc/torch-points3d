import torch

import sys
import logging

from torch_points3d.core.common_modules import Seq, Identity
import torch_points3d.modules.SparseConv3d.nn as snn

log = logging.getLogger(__name__)

try:
    from torch_points3d.modules.SPVCNN.utils import initial_voxelize, point_to_voxel, voxel_to_point
    from torchsparse import PointTensor
except:
    log.error("Can't load torchsparse, SPVCNN modules will be unavailable.")

class ResBlock(torch.nn.Module):
    """
    Basic ResNet type block

    Parameters
    ----------
    input_nc:
        Number of input channels
    output_nc:
        number of output channels
    convolution
        Either MinkowskConvolution or MinkowskiConvolutionTranspose
    dimension:
        Dimension of the spatial grid
    """

    def __init__(self, input_nc, output_nc, convolution):
        super().__init__()
        self.block = (
            Seq()
            .append(convolution(input_nc, output_nc, kernel_size=3, stride=1))
            .append(snn.BatchNorm(output_nc))
            .append(snn.ReLU())
            .append(convolution(output_nc, output_nc, kernel_size=3, stride=1))
            .append(snn.BatchNorm(output_nc))
            .append(snn.ReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(snn.Conv3d(input_nc, output_nc, kernel_size=1, stride=1)).append(snn.BatchNorm(output_nc))
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


class BottleneckBlock(torch.nn.Module):
    """
    Bottleneck block with residual
    """

    def __init__(self, input_nc, output_nc, convolution, reduction=4):
        super().__init__()

        self.block = (
            Seq()
            .append(snn.Conv3d(input_nc, output_nc // reduction, kernel_size=1, stride=1))
            .append(snn.BatchNorm(output_nc // reduction))
            .append(snn.ReLU())
            .append(convolution(output_nc // reduction, output_nc // reduction, kernel_size=3, stride=1,))
            .append(snn.BatchNorm(output_nc // reduction))
            .append(snn.ReLU())
            .append(snn.Conv3d(output_nc // reduction, output_nc, kernel_size=1,))
            .append(snn.BatchNorm(output_nc))
            .append(snn.ReLU())
        )

        if input_nc != output_nc:
            self.downsample = (
                Seq().append(convolution(input_nc, output_nc, kernel_size=1, stride=1)).append(snn.BatchNorm(output_nc))
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        return out


_res_blocks = sys.modules[__name__]


class ResNetDown(torch.nn.Module):
    """
    Resnet block that looks like

    in --- strided conv ---- Block ---- sum --[... N times]
                         |              |
                         |-- 1x1 - BN --|
    """

    CONVOLUTION = "Conv3d"

    def __init__(
        self, down_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1, block="ResBlock", skip_feat=[], **kwargs,
    ):
        block = getattr(_res_blocks, block)
        super().__init__()
        conv1_output = down_conv_nn[1]

        conv = getattr(snn, self.CONVOLUTION)
        self.conv_in = (
            Seq()
            .append(
                conv(
                    in_channels=down_conv_nn[0],
                    out_channels=conv1_output,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                )
            )
            .append(snn.BatchNorm(conv1_output))
            .append(snn.ReLU())
        )

        if N > 0:
            self.blocks = Seq()
            for i, _ in enumerate(range(N)):
                # add skip connections to 1st res block
                if i == 0 and skip_feat:
                    conv1_output += skip_feat
                self.blocks.append(block(conv1_output, down_conv_nn[1], conv))
                conv1_output = down_conv_nn[1]
        else:
            self.blocks = None

    def forward(self, x):
        out = self.conv_in(x)
        if self.blocks:
            out = self.blocks(out)
        return out


class ResNetUp(ResNetDown):
    """
    Same as Down conv but for the Decoder
    """

    CONVOLUTION = "Conv3dTranspose"

    def __init__(self, up_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1, skip_feat=[], dropout=0., **kwargs):        
        super().__init__(
            down_conv_nn=up_conv_nn, kernel_size=kernel_size, dilation=dilation, stride=stride, N=N, skip_feat=skip_feat, **kwargs,
        )
        self.dropout = torch.nn.Dropout(dropout, True)

    def forward(self, x, skip):
        x.F = self.dropout(x.F)

        out = self.conv_in(x)

        if skip is not None:
            out = snn.cat(out, skip)  

        if self.blocks:
            out = self.blocks(out)

        return out

class ResNetDownPV(ResNetDown):
    def __init__(self, down_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1, point_nn=None, res=1.0, skip_feat=[], **kwargs):
        super().__init__(
            down_conv_nn=down_conv_nn, kernel_size=kernel_size, dilation=dilation, stride=stride, skip_feat=skip_feat, N=N, **kwargs,
        )

        self.res = res
        if point_nn is not None:
            self.point_layer = (
                Seq()
                    .append(torch.nn.Linear(point_nn[0], point_nn[1]))
                    .append(torch.nn.BatchNorm1d(point_nn[1]))
                    .append(torch.nn.ReLU(True))
            )
        else:
            self.point_layer = None

    def forward(self, x):
        if not isinstance(x, tuple):
            pointFeats = PointTensor(x.F, x.C.float())
            voxelFeats = initial_voxelize(pointFeats, self.res, self.res)

            voxelFeats = self.conv_in(voxelFeats)
            pointFeats = voxel_to_point(voxelFeats, pointFeats, nearest=False)
            voxelFeats = point_to_voxel(voxelFeats, pointFeats)
        else:
            voxelFeats = x[0]
            pointFeats = x[1]

            voxelFeats = self.conv_in(voxelFeats)

        if self.blocks:
            voxelFeats = self.blocks(voxelFeats)

        if self.point_layer:
            point_out = self.point_layer(pointFeats.F)
            pointFeats = voxel_to_point(voxelFeats, pointFeats)
            pointFeats.F = pointFeats.F + point_out
            voxelFeats = point_to_voxel(voxelFeats, pointFeats)

        return (voxelFeats, pointFeats)


class ResNetUpPV(ResNetDownPV):
    CONVOLUTION = "Conv3dTranspose"

    def __init__(self, up_conv_nn=[], kernel_size=2, dilation=1, stride=2, N=1, point_nn=None, res=1.0, dropout=0.3, skip_feat=[],  **kwargs):
        # print("up")
        super().__init__(
            down_conv_nn=up_conv_nn, kernel_size=kernel_size, dilation=dilation, stride=stride, N=N, point_nn=point_nn, res=res, skip_feat=skip_feat, **kwargs,
        )
        self.dropout = torch.nn.Dropout(dropout, True)

    def forward(self, x, skip):
        voxelFeats = x[0]
        pointFeats = x[1]

        voxelFeats.F = self.dropout(voxelFeats.F)
        voxelFeats = self.conv_in(voxelFeats)
        voxelFeats = snn.cat(voxelFeats, skip[0])
        voxelFeats = self.blocks(voxelFeats)

        if self.point_layer:
            point_out = self.point_layer(pointFeats.F)
            pointFeats = voxel_to_point(voxelFeats, pointFeats)
            pointFeats.F = pointFeats.F + point_out
            voxelFeats = point_to_voxel(voxelFeats, pointFeats)
        
        return (voxelFeats, pointFeats)
