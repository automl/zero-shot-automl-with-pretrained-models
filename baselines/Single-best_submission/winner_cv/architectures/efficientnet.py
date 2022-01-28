import logging
import sys
from collections import OrderedDict
from copy import copy
import numpy as np

import skeleton
import torch
from torch import nn
from torch.nn import functional as F

from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)

formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s %(filename)s] %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, in_channels=3, num_classes=10, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        if in_channels == 3:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
            )
        elif in_channels == 1:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                skeleton.nn.CopyChannels(3),
            )
        else:
            self.stem = torch.nn.Sequential(
                skeleton.nn.Normalize(0.5, 0.25, inplace=False),
                torch.nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(3),
            )

        in_channels = 3

        # Stem
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, num_classes,
                             bias=True) #set bias=False later
        self._swish = MemoryEfficientSwish()

        self._half = False
        self._class_normalize = True
        self._is_video = False


    def set_video(self, is_video=True, times=False):
        self.is_video = is_video
        if is_video:
            self.conv1d_prev = torch.nn.Sequential(
                skeleton.nn.SplitTime(times),
                skeleton.nn.Permute(0, 2, 1, 3, 4),
            )

            self.conv1d_post = torch.nn.Sequential()

    def is_video(self):
        return self._is_video

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        inputs = self.stem(inputs)
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward_origin(self, x):
        bs = x.size(0)
        if x.size(2) < 32:
            pad_length = 32 - x.size(2)
            x = nn.ZeroPad2d((0, 0, int(np.ceil(pad_length/2)),
                                    int(np.floor(pad_length/2))))(x)
        if x.size(3) < 32:
            pad_length = 32 - x.size(3)
            x = nn.ZeroPad2d((int(np.ceil(pad_length/2)),
                              int(np.floor(pad_length/2)), 0, 0))(x)

        # Convolution layers
        x = self.extract_features(x)

        # Pooling and final linear layer
        x = self._avg_pooling(x)

        #if self.is_video():
        #    x = self.conv1d_prev(x)
        #    x = x.view(bs, x.size(1), -1)
        #    x = self.conv1d(x)
        #    x = self.conv1d_post(x)

        x = x.view(bs, -1)
        x_f = x.view(x.size(0), -1)
        x = self._dropout(x_f)
        x = self._fc(x)
        return x, x_f


    def forward(self, inputs, targets=None, tau=8.0, reduction='avg'):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        dims = len(inputs.shape)

        #if self.is_video() and dims == 5:
        #    batch, times, channels, height, width = inpus.shape
        #    inputs = inputs.view(batch * times, channels, height, width)

        #inputs = self.stem(inputs)
        logits, features = self.forward_origin(inputs)
        logits /= tau

        if targets is None:
            return logits, features
        if targets.device != logits.device:
            targets = targets.to(device=logits.device)

        loss = self.loss_fn(input=logits, target=targets)

        if self._class_normalize and isinstance(
            self.loss_fn, (torch.nn.BCEWithLogitsLoss, skeleton.nn.BinaryCrossEntropyLabelSmooth)
        ):
            pos = (targets == 1).to(logits.dtype)
            neg = (targets < 1).to(logits.dtype)
            npos = pos.sum()
            nneg = neg.sum()

            positive_ratio = max(0.1, min(0.9, (npos) / (npos + nneg)))
            negative_ratio = max(0.1, min(0.9, (nneg) / (npos + nneg)))
            LOGGER.debug(
                '[BCEWithLogitsLoss] positive_ratio:%f, negative_ratio:%f', positive_ratio,
                negative_ratio
            )

            normalized_loss = (loss * pos) / positive_ratio
            normalized_loss += (loss * neg) / negative_ratio

            loss = normalized_loss

        if reduction == 'avg':
            loss = loss.mean()
        elif reduction == 'max':
            loss = loss.max()
        elif reduction == 'min':
            loss = loss.min()

        return logits, loss, features


    def half(self):
        for module in self.modules():
            if len([c for c in module.children()]) > 0:
                continue
            if not isinstance(module, (torch.nn.BatchNorm1d,
                                       torch.nn.BatchNorm2d)):
                module.half()
            else:
                module.float()
        self._half = True
        return self

    def init(self, model_dir, model_name,
             advprop=True, gain=1.):
        load_pretrained_weights(self, model_name, model_dir=model_dir,
                                advprop=advprop)

        torch.nn.init.xavier_uniform_(self._fc.weight, gain=gain)
        LOGGER.debug('initialize classifier weight')

    @classmethod
    def from_name(cls, in_channels, num_classes, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(in_channels, num_classes, blocks_args, global_params)

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnetb'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


def efficientnetb0(in_channels, num_classes):
    return EfficientNet.from_name(in_channels, num_classes, 'efficientnetb0',
                                  override_params={'num_classes': num_classes})

def efficientnetb1(in_channels, num_classes):
    return EfficientNet.from_name(in_channels, num_classes, 'efficientnetb1',
                                  override_params={'num_classes': num_classes})

def efficientnetb2(in_channels, num_classes):
    return EfficientNet.from_name(in_channels, num_classes, 'efficientnetb2',
                                  override_params={'num_classes': num_classes})

def efficientnetb3(in_channels, num_classes):
    return EfficientNet.from_name(in_channels, num_classes, 'efficientnetb3',
                                  override_params={'num_classes': num_classes})

def efficientnetb4(in_channels, num_classes):
    return EfficientNet.from_name(in_channels, num_classes, 'efficientnetb4',
                                  override_params={'num_classes': num_classes})

def efficientnetb5(in_channels, num_classes):
    return EfficientNet.from_name(in_channels, num_classes, 'efficientnetb5',
                                  override_params={'num_classes': num_classes})

def efficientnetb6(in_channels, num_classes):
    return EfficientNet.from_name(in_channels, num_classes, 'efficientnetb6',
                                  override_params={'num_classes': num_classes})

def efficientnetb7(in_channels, num_classes):
    return EfficientNet.from_name(in_channels, num_classes, 'efficientnetb7',
                                  override_params={'num_classes': num_classes})

