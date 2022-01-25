# All code taken from se3cnn
# github: https://github.com/mariogeiger/se3cnn
# MIT License
# Copyright (c) 2019 Mario Geiger

from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from se3cnn.image.gated_block import GatedBlock
from se3cnn.image.norm_block import NormBlock

from se3cnn.non_linearities import NormSoftplus
from se3cnn.image.gated_activation import GatedActivation

from se3cnn.image.kernel import gaussian_window_wrapper

class ResNet(nn.Module):
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[block for block in blocks if block is not None])

    def forward(self, x):
        return self.blocks(x)

class CustomResNet(ResNet):
    def __init__(self,
                 n_input,
                 n_output,
                 args,
                 module_filters=(8,8,8,8),
                 filter_factor=2,
                 final_size=None,
                 size_resblocks = [3,4,6,2]
                 ):

        if final_size is None: # Automatically set the final size to the penultimate conv layers sizes * their multiplicity
            final_size = int(sum([math.ceil((filter_factor**3) * mf*(2*idx+1)) for idx, mf in enumerate(module_filters)]))
        assert type(module_filters) == tuple, 'needs to be a tuple of length at least 1'
        assert type(final_size) == int, 'final output needs to be an integer'
        assert len(size_resblocks) == 4, 'need the size for 4 resblocks'
        

        features = [[[module_filters]],
                    [[module_filters] * 2] * size_resblocks[0],
                    [[tuple([math.ceil(filter_factor * mf) for mf in module_filters])] * 2] * size_resblocks[1],
                    [[tuple([math.ceil((filter_factor ** 2) * mf) for mf in module_filters])] * 2] * size_resblocks[2],
                    [[tuple([math.ceil((filter_factor ** 3) * mf) for mf in module_filters])] * 2] * size_resblocks[3] 
                            + [[tuple([math.ceil((filter_factor ** 3) * mf) for mf in module_filters]), (final_size,0,0,0)]]]
        common_params = {
            'radial_window': partial(gaussian_window_wrapper,
                                     mode=args.bandlimit_mode, border_dist=0, sigma=0.6),
            'batch_norm_momentum': 0.01,
            # TODO: probability needs to be adapted to capsule order
            'capsule_dropout_p': args.p_drop_conv,  # drop probability of whole capsules
            'downsample_by_pooling': args.downsample_by_pooling,
            'normalization': args.normalization
        }
        if args.SE3_nonlinearity == 'gated':
            res_block = SE3GatedResBlock
        else:
            res_block = SE3NormResBlock
        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=partial(res_block, **common_params))
        super().__init__(
            OuterBlock((n_input,),          features[0], size=7),
            OuterBlock(features[0][-1][-1], features[1], size=args.kernel_size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=args.kernel_size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=args.kernel_size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=args.kernel_size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=args.p_drop_fully, inplace=True) if args.p_drop_fully is not None else None,
            nn.Linear(features[4][-1][-1][0], n_output))
class LargeNetwork(ResNet):
    def __init__(self,
                 n_input,
                 n_output,
                 args,
                 ):

        features = [[[(8,  8,  8,  8)]],          # 128 channels
                    [[(8,  8,  8,  8)] * 2] * 3,  # 128 channels
                    [[(16, 16, 16, 16)] * 2] * 4,  # 256 channels
                    [[(32, 32, 32, 32)] * 2] * 6,  # 512 channels
                    [[(64, 64, 64, 64)] * 2] * 2 + [[(64, 64, 64, 64), (1024, 0, 0, 0)]]]  # 1024 channels
        common_params = {
            'radial_window': partial(gaussian_window_wrapper,
                                     mode=args.bandlimit_mode, border_dist=0, sigma=0.6),
            'batch_norm_momentum': 0.01,
            # TODO: probability needs to be adapted to capsule order
            'capsule_dropout_p': args.p_drop_conv,  # drop probability of whole capsules
            'downsample_by_pooling': args.downsample_by_pooling,
            'normalization': args.normalization
        }
        if args.SE3_nonlinearity == 'gated':
            res_block = SE3GatedResBlock
        else:
            res_block = SE3NormResBlock
        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=partial(res_block, **common_params))
        super().__init__(
            OuterBlock((n_input,),          features[0], size=7),
            OuterBlock(features[0][-1][-1], features[1], size=args.kernel_size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=args.kernel_size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=args.kernel_size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=args.kernel_size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=args.p_drop_fully, inplace=True) if args.p_drop_fully is not None else None,
            nn.Linear(features[4][-1][-1][0], n_output))


class SmallNetwork(ResNet):
    def __init__(self,
                 n_input,
                 n_output,
                 args):

        features = [[[(2,  2,  2,  2)]],          #  32 channels
                    [[(2,  2,  2,  2)] * 2] * 3,  #  32 channels
                    [[(4,  4,  4,  4)] * 2] * 4,  #  64 channels
                    [[(8,  8,  8,  8)] * 2] * 6,  # 128 channels
                    # [[(8, 8, 8, 8)] * 2] * 2 + [[(8, 8, 8, 8), (128, 0, 0, 0)]]]  # 256 channels
                    [[(16, 16, 16, 16)] * 2] * 2 + [[(16, 16, 16, 16), (256, 0, 0, 0)]]]  # 256 channels
        common_params = {
            'radial_window': partial(gaussian_window_wrapper,
                                     mode=args.bandlimit_mode, border_dist=0, sigma=0.6),
            'batch_norm_momentum': 0.01,
            # TODO: probability needs to be adapted to capsule order
            'capsule_dropout_p': args.p_drop_conv,  # drop probability of whole capsules
            'normalization': args.normalization,
            'downsample_by_pooling': args.downsample_by_pooling,
        }
        if args.SE3_nonlinearity == 'gated':
            res_block = SE3GatedResBlock
        else:
            res_block = SE3NormResBlock
        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=partial(res_block, **common_params))
        super().__init__(
            OuterBlock((n_input,),          features[0], size=7),
            OuterBlock(features[0][-1][-1], features[1], size=args.kernel_size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=args.kernel_size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=args.kernel_size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=args.kernel_size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=args.p_drop_fully,
                       inplace=True) if args.p_drop_fully is not None else None,
            nn.Linear(features[4][-1][-1][0], n_output)
        )


class Merge(nn.Module):
    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class AvgSpacial(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)


class SE3GatedResBlock(nn.Module):
    def __init__(self, in_repr, out_reprs,
                 size=3,
                 stride=1,
                 radial_window=None,
                 batch_norm_momentum=0.01,
                 normalization="batch",
                 capsule_dropout_p=0.1,
                 scalar_gate_activation=(F.relu, F.sigmoid),
                 downsample_by_pooling=False):
        super().__init__()

        reprs = [in_repr] + out_reprs

        self.layers = []
        single_layer = len(out_reprs) == 1
        conv_stride = 1 if downsample_by_pooling else stride
        for i in range(len(reprs) - 1):
            # No activation in last block
            activation = scalar_gate_activation
            if i == (len(reprs) - 2) and not single_layer:
                activation = None
            self.layers.append(
                GatedBlock(reprs[i], reprs[i + 1],
                           size=size, padding=size//2,
                           stride=conv_stride if i == 0 else 1,
                           activation=activation,
                           radial_window=radial_window,
                           batch_norm_momentum=batch_norm_momentum,
                           normalization=normalization,
                           smooth_stride=False,
                           capsule_dropout_p=capsule_dropout_p))
            if downsample_by_pooling and i == 0 and stride > 1:
                self.layers.append(nn.AvgPool3d(kernel_size=size,
                                                padding=size//2,
                                                stride=stride))
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = None
        self.activation = None
        # Add shortcut if number of layers is larger than 1
        if not single_layer:
            # Use identity is input and output reprs are identical
            if in_repr == out_reprs[-1] and stride == 1:
                self.shortcut = lambda x: x
            else:
                self.shortcut = []
                self.shortcut.append(
                    GatedBlock(reprs[0], reprs[-1],
                               size=size, padding=size//2,
                               stride=conv_stride,
                               activation=None,
                               radial_window=radial_window,
                               batch_norm_momentum=batch_norm_momentum,
                               normalization=normalization,
                               smooth_stride=False,
                               capsule_dropout_p=capsule_dropout_p))
                if downsample_by_pooling and stride > 1:
                    self.shortcut.append(nn.AvgPool3d(kernel_size=size,
                                                      padding=size//2,
                                                      stride=stride))
                self.shortcut = nn.Sequential(*self.shortcut)

            self.activation = GatedActivation(
                repr_in=reprs[-1],
                size=size,
                radial_window=radial_window,
                batch_norm_momentum=batch_norm_momentum,
                normalization=normalization)

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
            out = self.activation(out)
        return out


class SE3NormResBlock(nn.Module):
    def __init__(self, in_repr, out_reprs,
                 size=3,
                 stride=1,
                 radial_window=None,
                 batch_norm_momentum=0.01,
                 normalization="batch",
                 capsule_dropout_p=0.1,
                 scalar_activation=F.relu,
                 activation_bias_min=0.5,
                 activation_bias_max=2,
                 downsample_by_pooling=False):
        super().__init__()

        reprs = [in_repr] + out_reprs

        self.layers = []
        single_layer = len(out_reprs) == 1
        conv_stride = 1 if downsample_by_pooling else stride
        for i in range(len(reprs) - 1):
            # No activation in last block
            activation = scalar_activation
            if i == (len(reprs) - 2) and not single_layer:
                activation = None
            self.layers.append(
                NormBlock(reprs[i], reprs[i + 1],
                          size=size, padding=size//2,
                          stride=conv_stride if i == 0 else 1,
                          activation=activation,
                          radial_window=radial_window,
                          normalization=normalization,
                          batch_norm_momentum=batch_norm_momentum,
                          capsule_dropout_p=capsule_dropout_p))
            if downsample_by_pooling and i == 0 and stride > 1:
                self.layers.append(nn.AvgPool3d(kernel_size=size,
                                                padding=size//2,
                                                stride=stride))
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = None
        self.activation = None
        # Add shortcut if number of layers is larger than 1
        if not single_layer:
            # Use identity is input and output reprs are identical
            if in_repr == out_reprs[-1] and stride == 1:
                self.shortcut = lambda x: x
            else:
                self.shortcut = []
                self.shortcut.append(
                    NormBlock(reprs[0], reprs[-1],
                              size=size, padding=size//2,
                              stride=conv_stride,
                              activation=None,
                              activation_bias_min=activation_bias_min,
                              activation_bias_max=activation_bias_max,
                              radial_window=radial_window,
                              normalization=normalization,
                              batch_norm_momentum=batch_norm_momentum,
                              capsule_dropout_p=capsule_dropout_p))
                if downsample_by_pooling and stride > 1:
                    self.shortcut.append(nn.AvgPool3d(kernel_size=size,
                                                      padding=size//2,
                                                      stride=stride))
                self.shortcut = nn.Sequential(*self.shortcut)

            capsule_dims = [2 * n + 1 for n, mul in enumerate(out_reprs[-1]) for i in
                            range(mul)]  # list of capsule dimensionalities
            self.activation = NormSoftplus(capsule_dims,
                                           scalar_act=scalar_activation,
                                           bias_min=activation_bias_min,
                                           bias_max=activation_bias_max)

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut is not None:
            out += self.shortcut(x)
            out = self.activation(out)
        return out


class OuterBlock(nn.Module):
    def __init__(self, in_repr, out_reprs, res_block, size=3, stride=1, **kwargs):
        super().__init__()

        reprs = [[in_repr]] + out_reprs

        self.layers = []
        for i in range(len(reprs) - 1):
            self.layers.append(
                res_block(reprs[i][-1], reprs[i+1],
                          size=size,
                          stride=stride if i == 0 else 1,
                          **kwargs)
            )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        return out




class NonlinearityBlock(nn.Module):
    ''' wrapper around GatedBlock and NormBlock, selects based on string SE3Nonlniearity '''
    def __init__(self, features_in, features_out, SE3_nonlinearity, **kwargs):
        super().__init__()
        if SE3_nonlinearity == 'gated':
            conv_block = GatedBlock
        elif SE3_nonlinearity == 'norm':
            conv_block = NormBlock
        else:
            raise NotImplementedError('unknown SE3_nonlinearity')
        self.conv_block = conv_block(features_in, features_out, **kwargs)
    def forward(self, x):
        return self.conv_block(x)
