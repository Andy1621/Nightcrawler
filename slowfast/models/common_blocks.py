from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import torch
import torch.nn as nn


class NLBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, nl_cfg, group=False):
        super(NLBlock, self).__init__()

        self.nl_cfg = nl_cfg
        self.group = group
        self.group_size = 4

        init_std = 0.01
        bias = True
        pool_stride = 2

        self.scale_value = dim_inner ** (-0.5)
        self.dim_inner = dim_inner

        self.theta = nn.Conv3d(dim_in, dim_inner, 1, bias=bias)
        nn.init.normal_(self.theta.weight, std=init_std)
        if bias:
            nn.init.constant_(self.theta.bias, 0)

        if True:
            self.maxpool = nn.MaxPool3d((1, pool_stride, pool_stride),
                                        stride=(1, pool_stride, pool_stride))

        self.phi = nn.Conv3d(dim_in, dim_inner, 1, bias=bias)
        nn.init.normal_(self.phi.weight, std=init_std)
        if bias:
            nn.init.constant_(self.phi.bias, 0)

        self.g = nn.Conv3d(dim_in, dim_inner, 1, bias=bias)
        nn.init.normal_(self.g.weight, std=init_std)
        if bias:
            nn.init.constant_(self.g.bias, 0)

        if True:
            self.softmax = nn.Softmax(dim=2)

        self.out = nn.Conv3d(dim_inner, dim_out, 1, bias=bias)
        if False:
            nn.init.constant_(self.out.weight, 0)
        else:
            nn.init.normal_(self.out.weight, std=init_std)
        if bias:
            nn.init.constant_(self.out.bias, 0)

        if True:
            if nl_cfg.FROZEN_BN:
                self.bn = FrozenBatchNorm3d(dim_out, eps= 1e-05)
            else:
                self.bn = nn.BatchNorm3d(dim_out, eps= 1e-05, momentum= 0.1)
            nn.init.constant_(self.bn.weight, 0.0)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError('expected 4D or 5D input (got {}D input)'
                             .format(x.dim()))

        if self.group:
            x = x.transpose(1, 2)
            sz_before_group = list(x.shape)
            sz_after_group = sz_before_group.copy()
            sz_after_group[0] = -1
            sz_after_group[1] = self.group_size
            x = x.contiguous().view(*sz_after_group)
            x = x.transpose(1, 2)

        batch_size = x.shape[0]

        theta = self.theta(x)

        if True:
            max_pool = self.maxpool(x)
        else:
            max_pool = x

        phi = self.phi(max_pool)

        g = self.g(max_pool)

        org_size = theta.size()
        mat_size = [batch_size, self.dim_inner, -1]
        theta = theta.view(*mat_size)
        phi = phi.view(*mat_size)
        g = g.view(*mat_size)

        theta_phi = torch.bmm(theta.transpose(1, 2), phi)

        if True:
            if True:
                theta_phi_sc = theta_phi * self.scale_value
            else:
                theta_phi_sc = theta_phi
            p = self.softmax(theta_phi_sc)
        else:
            p = theta_phi / theta_phi.shape[-1]

        t = torch.bmm(g, p.transpose(1, 2))

        t = t.view(org_size)

        out = self.out(t)

        if True:
            out = self.bn(out)
        out = out + x

        if self.group:
            out = out.transpose(1, 2)
            out = out.contiguous().view(*sz_before_group)
            out = out.transpose(1, 2)

        return out

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict():
                if isinstance(m_child, (nn.BatchNorm3d, FrozenBatchNorm3d)):
                    weight_map[name + '.weight'] = '{}_s'.format(name)
                    weight_map[name + '.running_mean'] = '{}_rm'.format(name)
                    weight_map[name + '.running_var'] = '{}_riv'.format(name)
                elif isinstance(m_child, nn.GroupNorm):
                    weight_map[name + '.weight'] = '{}_s'.format(name)
                else:
                    weight_map[name + '.weight'] = '{}_w'.format(name)
                weight_map[name + '.bias'] = '{}_b'.format(name)
        return weight_map


class _FrozenBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, track_running_stats=True):
        super(_FrozenBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.register_buffer("weight", torch.Tensor(num_features))
            self.register_buffer("bias", torch.Tensor(num_features))
        else:
            self.register_buffer("weight", None)
            self.register_buffer("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)
        view_shape = (1, self.num_features) + (1,) * (input.dim() - 2)

        if self.track_running_stats:
            scale = self.weight / (self.running_var + self.eps).sqrt()
            bias = self.bias - self.running_mean * scale
        else:
            scale = self.weight
            bias = self.bias

        return scale.view(*view_shape) * input + bias.view(*view_shape)

    def extra_repr(self):
        return '{num_features}, eps={eps}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(_FrozenBatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class FrozenBatchNorm1d(_FrozenBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class FrozenBatchNorm2d(_FrozenBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class FrozenBatchNorm3d(_FrozenBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

class Conv3dBN(nn.Module):
    def __init__(self, cfg, dim_in, dim_out, kernels, stride, padding, dilation=1, init_weight=None):
        super(Conv3dBN, self).__init__()
        self.conv = nn.Conv3d(dim_in, dim_out, kernels, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)
        if cfg.SF.FROZEN_BN:
            self.bn = FrozenBatchNorm3d(dim_out, eps=1e-05)
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 0.0)
        else:
            self.bn = nn.BatchNorm3d(dim_out, eps=1e-05, momentum= 0.1)
            if init_weight is not None:
                nn.init.constant_(self.bn.weight, init_weight)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

    def c2_weight_mapping(self):
        return {
            'conv.weight': 'w',
            'bn.weight': 'bn_s',
            'bn.bias': 'bn_b',
            'bn.running_mean': 'bn_rm',
            'bn.running_var': 'bn_riv'
        }


class Bottleneck(nn.Module):
    def __init__(self, cfg, dim_in, dim_out, dim_inner, stride, dilation=1, use_temp_conv=1, temp_stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv3dBN(cfg, dim_in, dim_inner, (1 + use_temp_conv * 2, 1, 1),
                                 stride=(temp_stride, 1, 1), padding=(use_temp_conv, 0, 0))
        self.conv2 = Conv3dBN(cfg, dim_inner, dim_inner, (1, 3, 3), stride=(1, stride, stride),
                                 dilation=(1, dilation, dilation),
                                 padding=(0, dilation, dilation))
        self.conv3 = Conv3dBN(cfg, dim_inner, dim_out, (1, 1, 1), stride=(1, 1, 1),
                                 padding=0, init_weight=0.0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out

    def c2_weight_mapping(self):
        weight_map = {}
        for i in range(1, 4):
            name = 'conv{}'.format(i)
            child_map = getattr(self, name).c2_weight_mapping()
            for key, val in child_map.items():
                new_key = name + '.' + key
                prefix = 'branch2{}_'.format(chr(ord('a') + i - 1))
                weight_map[new_key] = prefix + val
        return weight_map


class ResBlock(nn.Module):
    def __init__(self, cfg, dim_in, dim_out, dim_inner, stride, dilation=1, use_temp_conv=0, temp_stride=1, need_shortcut=False):
        super(ResBlock, self).__init__()

        self.btnk = Bottleneck(cfg, dim_in, dim_out, dim_inner=dim_inner, stride=stride, dilation=dilation,
                               use_temp_conv=use_temp_conv, temp_stride=temp_stride)
        if not need_shortcut:
            self.shortcut = None
        else:
            self.shortcut = Conv3dBN(cfg, dim_in, dim_out, (1, 1, 1),
                                     stride=(temp_stride, stride, stride), padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        tr = self.btnk(x)
        if self.shortcut is None:
            sc = x
        else:
            sc = self.shortcut(x)
        return self.relu(tr + sc)

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict():
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    if isinstance(m_child, Conv3dBN):
                        prefix = 'branch1_'
                    else:
                        prefix = ''
                    weight_map[new_key] = prefix + val
        return weight_map


class ResNLBlock(nn.Module):
    def __init__(self, cfg, dim_in, dim_out, stride, num_blocks, dim_inner, use_temp_convs, temp_strides,
                 dilation=1, nonlocal_mod=1000, group_nonlocal=False, lateral=False):
        super(ResNLBlock, self).__init__()
        self.blocks = []
        for idx in range(num_blocks):
            block_name = "res_{}".format(idx)
            block_stride = stride if idx == 0 else 1
            block_dilation = dilation
            dim_in0 = dim_in + int(dim_in * 0.125 * 2) if lateral and (idx == 0) and (
                    "tconv" != 'ttoc_sum') else dim_in
            # To transfer weight from classification model, we change res5_0 from stride 2 to stride 1,
            # and all res5_x layers from dilation 1 to dilation 2. In pretrain, res5_0 with stride 2 need a shortcut conv.
            # idx==0 and dilation!=1 means that it need a short cut in pretrain stage,
            # so we should keep it since we load weight from a pretrained model.
            # if idx!=0, block_stride will not be larger than 1 in pretrain stage.
            need_shortcut = not (dim_in0==dim_out and temp_strides[idx]==1 and block_stride==1) or \
                             (idx==0 and dilation!=1)
            res_module = ResBlock(cfg, dim_in0, dim_out, dim_inner=dim_inner,
                                  stride=block_stride,
                                  dilation=block_dilation,
                                  use_temp_conv=use_temp_convs[idx],
                                  temp_stride=temp_strides[idx],
                                  need_shortcut=need_shortcut)
            self.add_module(block_name, res_module)
            self.blocks.append(block_name)
            dim_in = dim_out
            if idx % nonlocal_mod == nonlocal_mod - 1:
                nl_block_name = "nonlocal_{}".format(idx)
                nl_module = NLBlock(dim_in, dim_in, int(dim_in / 2),
                                    cfg.SF.NONLOCAL, group=group_nonlocal)
                self.add_module(nl_block_name, nl_module)
                self.blocks.append(nl_block_name)

    def forward(self, x):
        for layer_name in self.blocks:
            x = getattr(self, layer_name)(x)
        return x

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            idx = name.split('_')[-1]
            if m_child.state_dict():
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    if isinstance(m_child, NLBlock):
                        prefix = 'nonlocal_conv{}_' + '{}_'.format(idx)
                    else:
                        prefix = 'res{}_' + '{}_'.format(idx)
                    weight_map[new_key] = prefix + val
        return weight_map