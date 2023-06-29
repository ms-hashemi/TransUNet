# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from distutils.command.config import config
import logging
import math

from os.path import join, sep

from pandas import concat
def pjoin(*args, **kwargs):
    return join(*args, **kwargs).replace(sep, '/')
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, Conv3d, Conv1d
from torch.nn.modules.utils import _pair, _triple
from torch.distributions.normal import Normal
from scipy import ndimage
from . import configs
from .vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish, "sigmoid": torch.nn.Sigmoid()}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size) if len(img_size) == 1 else img_size

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Embeddings3D(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    # Note that for the specific application of degradation, time frame is considered as an input 
    # feature in addition to the image data
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings3D, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _triple(img_size) if len(img_size) == 1 else img_size

        if config.patches.get("grid") is not None:   # CNN Encoder
            # Image size at the end of the CNN encoder transformations:
            img_size_CNN_output = [img_size[0] // (2**(len(config.encoder_channels) - 1)),
                img_size[1] // (2**(len(config.encoder_channels) - 1)),
                img_size[2] // (2**(len(config.encoder_channels) - 1))
            ]
            grid_size = config.patches["grid"]  # For finding the number of image patches which will constitute the input sequence of the transformer
            patch_size = (img_size_CNN_output[0] // grid_size[0], img_size_CNN_output[1] // grid_size[1], img_size_CNN_output[2] // grid_size[2])
            patch_size_real = patch_size
            n_patches = (img_size_CNN_output[0] // patch_size_real[0]) * (img_size_CNN_output[1] // patch_size_real[1]) * (img_size_CNN_output[2] // patch_size_real[2])
            self.hybrid = True
        else:
            patch_size = _triple(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = CNNFeatures3D(config)
            in_channels = self.hybrid_model.out_channels  # The number of image channels at the end of CNN encoder transformations
        self.patch_embeddings = Conv3d(in_channels=in_channels,  # Standard encoding of image patches similar to Vision Transformers or ViTs
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.time_embeddings = Conv1d(in_channels=1, out_channels=config.hidden_size, kernel_size=1)
        # Positions are embedded into the same space of embedded image patches, but the additional embedding
        # shown in "n_patches+1" belongs to the time variable
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x, time):
        x = x.float()
        if -1 not in time:
            time = time.float()
            time = torch.unsqueeze(time,1)
            time = torch.unsqueeze(time,2)
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # output shape: (B/Batch size, hidden/Transformer's hidden features, n_patches^(1/3)/config.patches.grid[0], n_patches^(1/3), n_patches^(1/3))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # Embed the time variable
        if -1 not in time:
            time = self.time_embeddings(time)
            time = time.flatten(2)
            time = time.transpose(-1, -2)  # (B, n_patches, hidden)
            # x = torch.cat([x, time], dim=1) # Concatanate the embedded time to the image embeddings
            embeddings = x + time + self.position_embeddings
        else:
            embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        # with torch.no_grad():
        #     query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        #     key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        #     value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        #     out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

        #     query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
        #     key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
        #     value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
        #     out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

        #     self.attn.query.weight.copy_(query_weight)
        #     self.attn.key.weight.copy_(key_weight)
        #     self.attn.value.weight.copy_(value_weight)
        #     self.attn.out.weight.copy_(out_weight)
        #     self.attn.query.bias.copy_(query_bias)
        #     self.attn.key.bias.copy_(key_bias)
        #     self.attn.value.bias.copy_(value_bias)
        #     self.attn.out.bias.copy_(out_bias)

        #     mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
        #     mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
        #     mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
        #     mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

        #     self.ffn.fc1.weight.copy_(mlp_weight_0)
        #     self.ffn.fc2.weight.copy_(mlp_weight_1)
        #     self.ffn.fc1.bias.copy_(mlp_bias_0)
        #     self.ffn.fc2.bias.copy_(mlp_bias_1)

        #     self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
        #     self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
        #     self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
        #     self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
        # Above code is not right if previous network layers are needed to be trained. So changed it into the below piece of code
        query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

        query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
        key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
        value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
        out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

        self.attn.query.weight.copy_(query_weight)
        self.attn.key.weight.copy_(key_weight)
        self.attn.value.weight.copy_(value_weight)
        self.attn.out.weight.copy_(out_weight)
        self.attn.query.bias.copy_(query_bias)
        self.attn.key.bias.copy_(key_bias)
        self.attn.value.bias.copy_(value_bias)
        self.attn.out.bias.copy_(out_bias)

        mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
        mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
        mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
        mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

        self.ffn.fc1.weight.copy_(mlp_weight_0)
        self.ffn.fc2.weight.copy_(mlp_weight_1)
        self.ffn.fc1.bias.copy_(mlp_bias_0)
        self.ffn.fc2.bias.copy_(mlp_bias_1)

        self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
        self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
        self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
        self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        if len(config.patches.size) == 3:
            self.embeddings = Embeddings3D(config, img_size=img_size)
        else:
            self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, time):
        embedding_output, features = self.embeddings(input_ids, time)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DownSample(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            stride=2,
            use_batchnorm=False,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True# bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()

        super(DownSample, self).__init__(conv, bn, relu)


class CNNFeatures3D(nn.Module):
    def __init__(
            self,
            config,
            in_channels=3,
            use_batchnorm=True,
    ):
        super().__init__()
        self.config = config
        if self.config.get("use_batchnorm") is not None: # For the generative encoder
            use_batchnorm = self.config.use_batchnorm
        in_channels = list([in_channels]) + list(config.encoder_channels[:-1])
        out_channels = config.encoder_channels
        self.out_channels = config.encoder_channels[-1]
        self.root = nn.Sequential(OrderedDict(
            [('conv1', Conv3dReLU(in_channels[0], out_channels[0], kernel_size=3, padding=1, use_batchnorm=use_batchnorm)),
            ('conv2', Conv3dReLU(out_channels[0], out_channels[0], kernel_size=3, padding=1, use_batchnorm=use_batchnorm))]
        ))

        self.body = nn.Sequential(OrderedDict([
            (f'block{i:d}', nn.Sequential(OrderedDict([
                ('conv1', Conv3dReLU(in_channels[i], out_channels[i], kernel_size=3, padding=1, use_batchnorm=use_batchnorm)),
                ('conv2', Conv3dReLU(out_channels[i], out_channels[i], kernel_size=3, padding=1, use_batchnorm=use_batchnorm)),
                ('down', DownSample(out_channels[i], out_channels[i], kernel_size=1, padding=0, stride=2, use_batchnorm=use_batchnorm))
                ]))) for i in range(1, len(config.encoder_channels))
        ]))
    
    def forward(self, x):
        features = []
        b, c, in_size, _, _ = x.size()
        x = self.root(x)
        features.append(x)
        # x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)):
            x = self.body[i](x)
            right_size = int(in_size / (2**(i+1)))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        # x = self.body[-1](x)
        for i in range(len(self.config.skip_channels)-len(self.config.encoder_channels)):
            if self.config.skip_channels[len(self.config.skip_channels)-len(self.config.encoder_channels)-1-i] != 0:
                feat = nn.MaxPool3d(2)(feat)
                features.append(feat)
            else:
                features.append(None)
        return x, features[::-1]


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderBlock3D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class Morph3D(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(scale_factor=upsampling, mode='trilinear', align_corners=False) if upsampling > 1 else nn.Identity()
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d, upsampling)


class DecoderForGenerativeOutput(nn.Sequential):
    """
    The class for the decoder output layer of the generative TransVNet 
    """
    def __init__(self, config, in_channels, out_channels, kernel_size=3, upsampling=1):
        self.config = config
        if len(self.config.patches.size) == 3: # 3D
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            # conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
            # conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
            upsampling = nn.Upsample(scale_factor=upsampling, mode='trilinear', align_corners=False) if upsampling > 1 else nn.Identity()
        else: # 2D
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        if self.config.get("output_nonlinearity") is None:
            out = nn.Identity()
        elif not self.config.output_nonlinearity:
            out = nn.Identity()
        else:
            out = ACT2FN[self.config.output_nonlinearity]
        super().__init__(conv, upsampling, out)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.get("use_batchnorm") is not None: # For the generative encoder
            use_batchnorm = self.config.use_batchnorm
        else:
            use_batchnorm = True
        head_channels = config.decoder_channels[0]
        if config.classifier == 'gen':
            first_in_channels = config.label_size
            # first_in_channels = 1
        else:
            first_in_channels = config.hidden_size
        if len(config.patches.size) == 3:
            self.conv_more = Conv3dReLU(
                first_in_channels,
                head_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.conv_more = Conv2dReLU(
                first_in_channels,
                head_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
        # in_channels = config.decoder_channels[:-1]
        in_channels = [first_in_channels] + list(config.decoder_channels[:-1])
        # out_channels = config.decoder_channels[1:]
        out_channels = list(config.decoder_channels)
        # skip_channels = config.skip_channels[1:]
        skip_channels = list(config.skip_channels)

        if len(config.patches.size) == 3:
            blocks = [
                DecoderBlock3D(in_ch, out_ch, sk_ch, use_batchnorm) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
            ]
        else:
            blocks = [
                DecoderBlock(in_ch, out_ch, sk_ch, use_batchnorm) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
            ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w, d = int(round(n_patch**(1.0/3.0))), int(round(n_patch**(1.0/3.0))), int(round(n_patch**(1.0/3.0)))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w, d)
        # x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                # skip = features[i+1] if (self.config.skip_channels[i+1] > 0) else None
                skip = features[i] if (self.config.skip_channels[i] > 0) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Source: https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class EncoderForGenerativeModels(nn.Module):
    """
    A class encapsulating the whole encoder of TransVNet for generative purposes (not segmentation)
    """
    def __init__(self, config, img_size, vis):
        super().__init__()
        config.use_batchnorm = True
        self.config = config
        self.transformer = Transformer(config, img_size, vis)
        # Distribution parameters
        n_patch = 1
        for i in range(len(config.patches.grid)):
            n_patch = n_patch * config.patches.grid[i]
        self.conv = nn.Sequential(
            Conv1d(in_channels=config.hidden_size, out_channels=1, kernel_size=1),
            # nn.Tanh()
        )
        # self.fc_mean = nn.Sequential(
        #     nn.Linear(config.hidden_size*n_patch, n_patch),
        #     # nn.Tanh()
        # )
        # self.fc_log_variance = nn.Sequential(
        #     nn.Linear(config.hidden_size*n_patch, n_patch),
        #     # nn.Tanh()
        # )
        # self.fc_label = nn.Sequential(
        #     nn.Linear(config.hidden_size*n_patch, config.label_size),
            # nn.Linear(n_patch, config.label_size)
        # )
        self.conv_mean = nn.Sequential(
            Conv1d(in_channels=n_patch, out_channels=n_patch, groups=n_patch, kernel_size=1)
        )
        self.conv_log_variance = nn.Sequential(
            Conv1d(in_channels=n_patch, out_channels=n_patch, groups=n_patch, kernel_size=1)
        )
        self.fc_label = nn.Sequential(
            nn.Linear(n_patch, config.label_size)
        )

    def forward(self, x, time):
        # Encode (x, time) to get the mu and variance parameters as well as lebels
        if x.size()[1] == 1 and len(self.config.patches.size) == 3:
            x = x.repeat(1,3,1,1,1)
        elif x.size()[1] == 1 and len(self.config.patches.size) != 3:
            x = x.repeat(1,3,1,1)
        x_encoded, attn_weights, features = self.transformer(x, time)
        B, n_patch, hidden = x_encoded.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # x_encoded = x_encoded.contiguous().view(B, hidden*n_patch)
        # mu = self.fc_mean(x_encoded)
        # log_variance = self.fc_log_variance(x_encoded)
        # predicted_labels = self.fc_label(x_encoded)
        x_encoded = x_encoded.permute(0, 2, 1)
        x_encoded = torch.squeeze(self.conv(x_encoded), 1)
        x_encoded_variational = torch.unsqueeze(x_encoded, -1)
        mu = torch.squeeze(self.conv_mean(x_encoded_variational), -1)
        log_variance = torch.squeeze(self.conv_log_variance(x_encoded_variational), -1)
        predicted_labels = self.fc_label(x_encoded)
        return (mu, log_variance, predicted_labels, features)
        # return (predicted_labels, features)


class DecoderForGenerativeModels(nn.Module):
    """
    A class encapsulating the whole decoder of TransVNet for generative purposes (not segmentation)
    """
    def __init__(self, config, img_size):
        super().__init__()
        config.use_batchnorm = True
        self.config = config
        # Distribution parameters
        n_patch = 1
        for i in range(len(config.patches.grid)):
            n_patch = n_patch * config.patches.grid[i]
        # # Linear transformation on the mixed information (B, n_patch+label_size) of the latent space (B, n_patch) + labels (B, label_size) 
        # self.fc =  nn.Sequential(
        #     nn.Linear(config.label_size, n_patch), # n_patch + config.label_size
        #     # nn.Tanh()
        # )
        if self.config.get("n_decoder_CUPs") is not None: # For the generative decoder
            self.decoder = DecoderCup(config)
            if len(config.patches.size) == 3:
                self.morph_head = Morph3D(
                    in_channels=config['decoder_channels'][-1],
                    out_channels=config['n_classes'],
                    kernel_size=3,
                )
            else:
                self.segmentation_head = SegmentationHead(
                    in_channels=config['decoder_channels'][-1],
                    out_channels=config['n_classes'],
                    kernel_size=3,
                )
        elif self.config.n_decoder_CUPs <= 1:
            self.decoder = DecoderCup(config)
            if len(config.patches.size) == 3:
                self.morph_head = Morph3D(
                    in_channels=config['decoder_channels'][-1],
                    out_channels=config['n_classes'],
                    kernel_size=3,
                )
            else:
                self.segmentation_head = SegmentationHead(
                    in_channels=config['decoder_channels'][-1],
                    out_channels=config['n_classes'],
                    kernel_size=3,
                )
        else:
            self.decoder = [DecoderCup(config) for i in range(self.config.n_decoder_CUPs)]
            self.decoder_output = [DecoderForGenerativeOutput(config, in_channels=config['decoder_channels'][-1], out_channels=config['n_classes'], kernel_size=3) for i in range(self.config.n_decoder_CUPs)]
        
    def forward(self, x, features=None, time=0):
        # Decode the encoded input x to get a reconstructed image
        if self.config.get("n_decoder_CUPs") is not None: # For the generative decoder
            # x = torch.unsqueeze(self.fc(x), -1)
            x = self.decoder(x, features)
            if len(self.config.patches.size) == 3:
                x = self.morph_head(x)
                # x = self.spatial_transformer(src, x)
            else:
                x = self.segmentation_head(x)
            return x
        elif self.config.n_decoder_CUPs <= 1:
            # x = torch.unsqueeze(self.fc(x), -1)
            x = self.decoder(x, features)
            if len(self.config.patches.size) == 3:
                x = self.morph_head(x)
                # x = self.spatial_transformer(src, x)
            else:
                x = self.segmentation_head(x)
            return x
        else:
            batch_size = x.shape[0]
            idx = []
            xx = []
            counter = []
            for i in range(self.config.n_decoder_CUPs):
                idx.append(((0.2 + i * (0.7/self.config.n_decoder_CUPs) <= time) == (time < 0.2 + (i+1) * (0.7/self.config.n_decoder_CUPs))).nonzero())
                xx.append(self.decoder[i](torch.index_select(x, 0, idx[i][:, 0]), features))
                xx[i] = self.decoder_output[i](xx[i])
                counter.append(0)
            x = []
            for i in range(batch_size):
                for j in range(len(idx)):
                    if i in idx[j][:, 0]:
                        x.append(x[j][counter[j]])
                        counter[j] = counter[j] + 1
            x = torch.stack(x, 0)
            return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        if self.classifier == 'gen':
            self.encoder = EncoderForGenerativeModels(config, img_size, vis)
            # For the Gaussian likelihood used in reconstruction loss calculation
            self.log_scale = nn.Parameter(torch.zeros(img_size))
            self.decoder = DecoderForGenerativeModels(config, img_size)
        else:
            self.transformer = Transformer(config, img_size, vis)
            self.decoder = DecoderCup(config)
            if len(config.patches.size) == 3:
                self.morph_head = Morph3D(
                    in_channels=config['decoder_channels'][-1],
                    out_channels=config['n_classes'],
                    kernel_size=3,
                )
                self.spatial_transformer = SpatialTransformer(_triple(img_size) if len(img_size) == 1 else img_size)
            else:
                self.segmentation_head = SegmentationHead(
                    in_channels=config['decoder_channels'][-1],
                    out_channels=config['n_classes'],
                    kernel_size=3,
                )
        self.config = config

    def forward(self, x, time):
        # src = x[:,0:1,:,:,:].float()
        if x.size()[1] == 1 and len(self.config.patches.size) == 3:
            x = x.repeat(1,3,1,1,1)
        elif x.size()[1] == 1 and len(self.config.patches.size) != 3:
            x = x.repeat(1,3,1,1)
        if self.classifier == 'gen':
            # KL divergence estimation in the current batch using the VAE's encoder part (model.encoder)
            mu, log_variance, predicted_labels, features = self.encoder(x, torch.tensor([-1]).cuda()) # -1 indicates that no additional embeddings such as that of time in bone degradation is needed here.
            # predicted_labels, features = self.encoder(x, time)

            # Sample z from q(z|x)
            std = torch.exp(log_variance / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            # # Mixing the sampled tensor z(batch_size, number_of_patches) with predicted_labels(batch_size, label_size) to form the input tensor of the decoder for generative purposes
            # l = []
            # for i in range(predicted_labels.shape[1]):
            #     l.append(torch.mul(z, torch.sigmoid(torch.unsqueeze(predicted_labels[:, i], -1))))
            # decoder_input = torch.stack(l, 2)

            # # Concatenate the sampled tensor z(batch_size, number_of_patches) with the predicted_labels(batch_size, label_size) to form the input tensor of the decoder for generative purposes
            # decoder_input = torch.cat((z, predicted_labels), 1)

            # Chennel-wise addition of each predicted label to the respective sampled channel
            l = []
            for i in range(predicted_labels.shape[0]):
                l_i = z[i, :] + torch.unsqueeze(predicted_labels[i, :], -1)
                l.append(l_i.permute(1, 0))
            decoder_input = torch.stack(l, 0)

            # # Latent distribution is the isotropic normalized properties distribution
            # decoder_input = predicted_labels

            # # For testing!
            # decoder_input = mu

            # Monte carlo KL divergence
            # 1. define the first two probabilities (in this case Normal for both)
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std)) # Target latent distribution
            q = torch.distributions.Normal(mu, std) # Network-calculated latent distribution
            # 2. get the probabilities from the equation
            log_qzx = q.log_prob(z)
            log_pz = p.log_prob(z)
            # kl
            kl = None
            kl = torch.abs(log_qzx - log_pz)
            kl = kl.sum(-1)

            # Decoder output given a random sample of the encoder distribution and its predicted labels
            decoder_output = self.decoder(decoder_input, None, time)

            log_pxz = None
            # Gaussian likelihood for the reconstruction loss
            scale = torch.exp(self.log_scale)
            dist = torch.distributions.Normal(decoder_output, scale)
            # Measure prob of seeing image under p(x|y,z)
            log_pxz = dist.log_prob(x) # Reconstruction loss in VAEs
            if len(self.config.patches.size) == 3:
                log_pxz = log_pxz.sum(dim=(1, 2, 3, 4))
            else:
                log_pxz = log_pxz.sum(dim=(1, 2, 3))

            return (predicted_labels, decoder_output, kl, log_pxz)
        else:
            x, _, features = self.transformer(x, time)
            x = self.decoder(x, features)
            if len(self.config.patches.size) == 3:
                x = self.morph_head(x)
                # x = self.spatial_transformer(src, x)
            else:
                x = self.segmentation_head(x)
            return x

    def load_from(self, weights):        
        if weights:
            self.transformer.encoder.encoder_norm.weight.requires_grad_(False)
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.requires_grad_(False)
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.requires_grad_(False)
                    unit.load_from(weights, n_block=uname)
        # if weights: # Not suitable for 3D tasks due to the following line comment
        #     with torch.no_grad(): # Disables gradient calculation even for the previous layers! For more info, see https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc

        #         # res_weight = weights
        #         # self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
        #         # self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

        #         self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
        #         self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

        #         # posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

        #         # posemb_new = self.transformer.embeddings.position_embeddings
        #         # print("size of pre_trained = {}\n".format(posemb.shape))
        #         # print("size of new = {}\n".format(posemb_new.shape))
        #         # if posemb.size() == posemb_new.size():
        #         #     self.transformer.embeddings.position_embeddings.copy_(posemb)
        #         # elif posemb.size()[1]-1 == posemb_new.size()[1]:
        #         #     posemb = posemb[:, 1:]
        #         #     self.transformer.embeddings.position_embeddings.copy_(posemb)
        #         # else:
        #         #     logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
        #         #     ntok_new = posemb_new.size(1)
        #         #     if self.classifier == "seg":
        #         #         _, posemb_grid = posemb[:, :1], posemb[0, 1:]
        #         #     gs_old = int(np.sqrt(len(posemb_grid)))
        #         #     gs_new = int(np.sqrt(ntok_new))
        #         #     print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
        #         #     posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
        #         #     zoom = (gs_new / gs_old, gs_new / gs_old, 1)
        #         #     posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
        #         #     posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
        #         #     posemb = posemb_grid
        #         #     self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

        #         # Encoder whole
        #         for bname, block in self.transformer.encoder.named_children():
        #             for uname, unit in block.named_children():
        #                 unit.load_from(weights, n_block=uname)

        #         # if self.transformer.embeddings.hybrid:
        #         #     self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
        #         #     gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
        #         #     gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
        #         #     self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
        #         #     self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

        #         #     for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
        #         #         for uname, unit in block.named_children():
        #         #             unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

CONFIGS3D = {
    'ViT-B_16': configs.get_b16_3D_config(), # No initial CNN in the encoder
    'Conv-ViT-B_16': configs.get_conv_b16_3D_config(), # Degradation config - basic
    'Conv-ViT-Gen-B_16': configs.get_conv_b16_3D_gen_config(), # Design/VAE config - basic
    'Conv-ViT-Gen2-B_16': configs.get_conv_b16_3D_gen2_config() # Design/VAE config2
}

