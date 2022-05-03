from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import torch
import torch.nn as nn
from slowfast.models.common_blocks import ResNLBlock,FrozenBatchNorm3d
import torch.utils.checkpoint as checkpoint
from .build import MODEL_REGISTRY
import os

model_path = '/mnt/lustre/share_data/likunchang.vendor/model'
model_path = {
    'SlowFast-ResNet101-8x8': os.path.join(model_path, 'SlowFast-ResNet101-8x8.pth'),
}

def get_slow_model_cfg(cfg):
    backbone_strs = ["Resnet101"]
    error_msg = 'Model backbone {} is not supported.'.format("Slowfast-Resnet101")

    use_temp_convs_1 = [0]
    temp_strides_1 = [1]
    max_pool_stride_1 = 1

    use_temp_convs_2 = [0, 0, 0]
    temp_strides_2 = [1, 1, 1]

    use_temp_convs_3 = [0, 0, 0, 0]
    temp_strides_3 = [1, 1, 1, 1]

    use_temp_convs_5 = [1, 1, 1]
    temp_strides_5 = [1, 1, 1]

    slow_stride = 8
    avg_pool_stride = int(64 / slow_stride)
    if backbone_strs[0] == 'Resnet50':
        block_config = (3, 4, 6, 3)

        use_temp_convs_4 = [1, 1, 1, 1, 1, 1]
        temp_strides_4 = [1, 1, 1, 1, 1, 1]
    elif backbone_strs[0] == 'Resnet101':
        block_config = (3, 4, 23, 3)

        use_temp_convs_4 = [1, ] * 23
        temp_strides_4 = [1, ] * 23
    else:
        raise KeyError(error_msg)

    if len(backbone_strs) > 1:
        raise KeyError(error_msg)

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2, use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2, temp_strides_3, temp_strides_4, temp_strides_5]
    pool_strides_set = [max_pool_stride_1, avg_pool_stride]
    return block_config, use_temp_convs_set, temp_strides_set, pool_strides_set


def get_fast_model_cfg(cfg):
    backbone_strs = ["Resnet101"]
    error_msg = 'Model backbone {} is not supported.'.format("Slowfast-Resnet101")

    use_temp_convs_1 = [2]
    temp_strides_1 = [1]
    max_pool_stride_1 = 1

    use_temp_convs_2 = [1, 1, 1]
    temp_strides_2 = [1, 1, 1]

    use_temp_convs_3 = [1, 1, 1, 1]
    temp_strides_3 = [1, 1, 1, 1]

    use_temp_convs_5 = [1, 1, 1]
    temp_strides_5 = [1, 1, 1]

    fast_stride = 8 // 4
    avg_pool_stride = int(64 / fast_stride)

    if backbone_strs[0] == 'Resnet50':
        block_config = (3, 4, 6, 3)

        use_temp_convs_4 = [1, 1, 1, 1, 1, 1]
        temp_strides_4 = [1, 1, 1, 1, 1, 1]
    elif backbone_strs[0] == 'Resnet101':
        block_config = (3, 4, 23, 3)

        use_temp_convs_4 = [1, ] * 23
        temp_strides_4 = [1, ] * 23
    else:
        raise KeyError(error_msg)

    if len(backbone_strs) > 1:
        raise KeyError(error_msg)

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2, use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2, temp_strides_3, temp_strides_4, temp_strides_5]
    pool_strides_set = [max_pool_stride_1, avg_pool_stride]
    return block_config, use_temp_convs_set, temp_strides_set, pool_strides_set

class LateralBlock(nn.Module):
    def __init__(self, conv_dim, alpha):
        super(LateralBlock, self).__init__()
        self.conv = nn.Conv3d(conv_dim, conv_dim * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                              padding=(2, 0, 0), bias=True)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        out = self.conv(x)
        return out


class FastPath(nn.Module):
    def __init__(self, cfg):
        super(FastPath, self).__init__()

        self.cfg = cfg

        block_config, use_temp_convs_set, temp_strides_set, pool_strides_set = get_fast_model_cfg(cfg)
        conv3_nonlocal = False
        conv4_nonlocal = False
        dim_inner = 8
        conv_dims = [8, 32, 64, 128, 256]
        self.dim_out = conv_dims[-1]
        n1, n2, n3, n4 = block_config
        layer_mod = 2
        conv3_nl_mod = layer_mod
        conv4_nl_mod = layer_mod
        if not conv3_nonlocal:
            conv3_nl_mod = 1000
        if not conv4_nonlocal:
            conv4_nl_mod = 1000
        self.c2_mapping = None

        self.conv1 = nn.Conv3d(3, conv_dims[0], (1 + use_temp_convs_set[0][0] * 2, 7, 7),
                               stride=(temp_strides_set[0][0], 2, 2),
                               padding=(use_temp_convs_set[0][0], 3, 3), bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)

        if cfg.SF.FROZEN_BN:
            self.bn1 = FrozenBatchNorm3d(conv_dims[0], eps=1e-05)
            nn.init.constant_(self.bn1.weight, 1.0)
            nn.init.constant_(self.bn1.bias, 0.0)
        else:
            self.bn1 = nn.BatchNorm3d(conv_dims[0], eps=1e-05,
                                      momentum=0.1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d((pool_strides_set[0], 3, 3), stride=(pool_strides_set[0], 2, 2))

        self.res_nl1 = ResNLBlock(cfg, conv_dims[0], conv_dims[1], stride=1, num_blocks=n1, dim_inner=dim_inner,
                                  use_temp_convs=use_temp_convs_set[1], temp_strides=temp_strides_set[1])

        self.res_nl2 = ResNLBlock(cfg, conv_dims[1], conv_dims[2], stride=2, num_blocks=n2,
                                  dim_inner=dim_inner * 2, use_temp_convs=use_temp_convs_set[2],
                                  temp_strides=temp_strides_set[2], nonlocal_mod=conv3_nl_mod,
                                  group_nonlocal=False)

        self.res_nl3 = ResNLBlock(cfg, conv_dims[2], conv_dims[3], stride=2, num_blocks=n3,
                                  dim_inner=dim_inner * 4, use_temp_convs=use_temp_convs_set[3],
                                  temp_strides=temp_strides_set[3], nonlocal_mod=conv4_nl_mod)

        self.res_nl4 = ResNLBlock(cfg, conv_dims[3], conv_dims[4], stride=1, num_blocks=n4,
                                  dim_inner=dim_inner * 8, use_temp_convs=use_temp_convs_set[4],
                                  temp_strides=temp_strides_set[4],
                                  dilation=2)

        # if cfg.MODEL.BACKBONE.SLOWFAST.LATERAL == 'tconv':
        self._tconv(conv_dims)

    def _tconv(self, conv_dims):
        alpha = 4
        self.Tconv1 = LateralBlock(conv_dims[0], alpha)
        self.Tconv2 = LateralBlock(conv_dims[1], alpha)
        self.Tconv3 = LateralBlock(conv_dims[2], alpha)
        self.Tconv4 = LateralBlock(conv_dims[3], alpha)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        tconv1 = self.Tconv1(out)

        out = self.res_nl1(out)
        tconv2 = self.Tconv2(out)

        out = self.res_nl2(out)
        tconv3 = self.Tconv3(out)

        out = self.res_nl3(out)
        tconv4 = self.Tconv4(out)

        out = self.res_nl4(out)
        return  [tconv1, tconv2, tconv3, tconv4]


class SlowPath(nn.Module):
    def __init__(self, cfg):
        super(SlowPath, self).__init__()

        self.cfg = cfg

        block_config, use_temp_convs_set, temp_strides_set, pool_strides_set = get_slow_model_cfg(cfg)
        conv3_nonlocal = False
        conv4_nonlocal = False

        dim_inner = 64
        conv_dims = [64, 256, 512, 1024, 2048]
        self.dim_out = conv_dims[-1]
        n1, n2, n3, n4 = block_config
        layer_mod = 2
        conv3_nl_mod = layer_mod
        conv4_nl_mod = layer_mod
        if not conv3_nonlocal:
            conv3_nl_mod = 1000
        if not conv4_nonlocal:
            conv4_nl_mod = 1000
        self.c2_mapping = None

        self.conv1 = nn.Conv3d(3, conv_dims[0], (1 + use_temp_convs_set[0][0] * 2, 7, 7),
                               stride=(temp_strides_set[0][0], 2, 2),
                               padding=(use_temp_convs_set[0][0], 3, 3), bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)

        if cfg.SF.FROZEN_BN:
            self.bn1 = FrozenBatchNorm3d(conv_dims[0], eps=1e-05)
            nn.init.constant_(self.bn1.weight, 1.0)
            nn.init.constant_(self.bn1.bias, 0.0)
        else:
            self.bn1 = nn.BatchNorm3d(conv_dims[0], eps=1e-05,
                                      momentum=0.1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d((pool_strides_set[0], 3, 3), stride=(pool_strides_set[0], 2, 2))

        self.res_nl1 = ResNLBlock(cfg, conv_dims[0], conv_dims[1], stride=1, num_blocks=n1, dim_inner=dim_inner,
                                  use_temp_convs=use_temp_convs_set[1], temp_strides=temp_strides_set[1],
                                  lateral=True)

        self.res_nl2 = ResNLBlock(cfg, conv_dims[1], conv_dims[2], stride=2, num_blocks=n2,
                                  dim_inner=dim_inner * 2, use_temp_convs=use_temp_convs_set[2],
                                  temp_strides=temp_strides_set[2], nonlocal_mod=conv3_nl_mod,
                                  group_nonlocal=False,
                                  lateral=True)

        self.res_nl3 = ResNLBlock(cfg, conv_dims[2], conv_dims[3], stride=2, num_blocks=n3,
                                  dim_inner=dim_inner * 4, use_temp_convs=use_temp_convs_set[3],
                                  temp_strides=temp_strides_set[3], nonlocal_mod=conv4_nl_mod,
                                  lateral=True)

        self.res_nl4 = ResNLBlock(cfg, conv_dims[3], conv_dims[4], stride=1, num_blocks=n4,
                                  dim_inner=dim_inner * 8, use_temp_convs=use_temp_convs_set[4],
                                  temp_strides=temp_strides_set[4], lateral=True,
                                  dilation=2)

    def forward(self, x, lateral_connection=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        if lateral_connection:
            out = torch.cat([out, lateral_connection[0]], dim=1)

        out = self.res_nl1(out)
        if lateral_connection:
            out = torch.cat([out, lateral_connection[1]], dim=1)

        out = self.res_nl2(out)
        if lateral_connection:
            out = torch.cat([out, lateral_connection[2]], dim=1)

        out = self.res_nl3(out)
        if lateral_connection:
            out = torch.cat([out, lateral_connection[3]], dim=1)

        out = self.res_nl4(out)
        return out

@MODEL_REGISTRY.register()
class SF(nn.Module):
    def __init__(self, cfg):
        super(SF, self).__init__()
        self.cfg = cfg
        self.slow = SlowPath(cfg)
        self.fast = FastPath(cfg)
        self.dim_out = self.slow.dim_out + self.fast.dim_out
        self.dropout = nn.Dropout(cfg.SF.DROPOUT_RATE)
        self.projection = nn.Linear(2048, cfg.MODEL.NUM_CLASSES, bias=True)

    def forward(self,inputs):
        out = self.forward_features(inputs)
        out = out.flatten(2).mean(-1)
        out = self.dropout(out)
        out = self.projection(out)
        return out
    
    def get_pretrained_model(self, cfg):
        if cfg.SF.PRETRAIN_NAME:
            checkpoint = torch.load(model_path[cfg.SF.PRETRAIN_NAME], map_location='cpu')
            checkpoint = checkpoint["model"]
            newcheckpoint=dict()
            for k,v in checkpoint.items():
                newcheckpoint[k.replace("module.","").replace("backbone.","")]=v
            del newcheckpoint["cls_head.pred.weight"]
            del newcheckpoint["cls_head.pred.bias"]
            return newcheckpoint
        else:
            return None

    
    def forward_features(self,inputs):
        inputs = inputs[0]
        fast_x = inputs
        T = fast_x.shape[2]
        slow_x = inputs[:,:,[ i*4 for i in range(T//4)],:,:].detach()

        tconv = self.fast(fast_x)
        out =self.slow(slow_x, tconv)
        return out


# cfg = edict({"MODEL":{"NUM_CLASSES":11},"SF":{"PRETRAIN_NAME":"SlowFast-ResNet101-8x8",DROPOUT_RATE":0.2,"FROZEN_BN":True,"NONLOCAL":{"FROZEN_BN":True}}})