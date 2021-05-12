import torch
import torch.nn as nn
import os
from lib.fcn.config import cfg
from lib.utils.print_and_log import print_and_log

__all__ = [
    'dcm_ot_v0'
]

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, inplace=True, relu='leaky'):
    if batchNorm:
        if relu=='leaky':
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1,inplace=inplace)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=inplace)
            )
    else:
        if relu=='leaky':
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.LeakyReLU(0.1,inplace=inplace)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.ReLU(0.1,inplace=inplace)
            )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)

def predict_mask(in_planes, num_classes=1):
    valid_num_classes = num_classes if cfg.NETWORK.REF.CLASS_AWARE else 1
    if cfg.LOSS.REF.MASK_LOSS == 'sigmoid':
        return nn.Conv2d(in_planes,1*valid_num_classes,kernel_size=3,stride=1,padding=1,bias=True)
    elif cfg.LOSS.REF.MASK_LOSS == 'softmax':
        return nn.Conv2d(in_planes,2*valid_num_classes,kernel_size=3,stride=1,padding=1,bias=True)
    else:
        raise Exception("No Mask Loss Type: {}".format(cfg.TRAIN.MASK_LOSS))

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def fc(in_planes, out_planes, relu=True, bias=True):
    if relu:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes, bias=bias),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Linear(in_planes, out_planes, bias=bias)

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

def all2quat(axangle):
    if cfg.NETWORK.REF.ROT_TYPE in ['axisangle', 'a3a3', 'a3xa3', 'ag1ax3']:
        if cfg.NETWORK.REF.ROT_TYPE == 'axisangle':
            theta = torch.norm(axangle, dim=1, keepdim=True)
            vector = torch.nn.functional.normalize(axangle)
        elif cfg.NETWORK.REF.ROT_TYPE == 'a3a3':
            theta = torch.norm(axangle[:, :3], dim=1, keepdim=True)
            vector = torch.nn.functional.normalize(axangle[:, 3:])
        elif cfg.NETWORK.REF.ROT_TYPE == 'a3xa3':
            theta = torch.norm(torch.mul(axangle[:, :3], axangle[:, 3:]), dim=1, keepdim=True)
            vector = torch.nn.functional.normalize(axangle[:, 3:])
        elif cfg.NETWORK.REF.ROT_TYPE == 'ag1ax3':
            theta = axangle[:, [0]]
            vector = torch.nn.functional.normalize(axangle[:, 1:])
        t2 = theta / 2.0
        st2 = torch.sin(t2)
        quaternion = torch.cat([torch.cos(t2), vector * st2], dim=1)
        return quaternion
    if cfg.NETWORK.REF.ROT_TYPE == 'ortho6d':
        raise Exception("NOT IMPLEMENTED")

def t_transform(T_src, T_delta, zoom_factor, num_classes, intrinsic_matrics):
    '''
    :param T_src: (x1, y1, z1)
    :param T_delta: (dx, dy, dz)
    :return: T_tgt: (x2, y2, z2)
    '''
    factor_x = torch.unsqueeze(zoom_factor[:, 0], 1)
    factor_y = torch.unsqueeze(zoom_factor[:, 1], 1)
    focal_x = torch.unsqueeze(intrinsic_matrics[:, 0, 0], 1)/cfg.NETWORK.REF.TRANS_SCALE_FACTOR
    focal_y = torch.unsqueeze(intrinsic_matrics[:, 1, 1], 1)/cfg.NETWORK.REF.TRANS_SCALE_FACTOR

    vx_0 = torch.mul(T_delta[:, 0::3], factor_x)
    vy_0 = torch.mul(T_delta[:, 1::3], factor_y)
    vx_0 = torch.div(vx_0, focal_x)
    vy_0 = torch.div(vy_0, focal_y)

    vz = torch.div(T_src[:, 2::3], torch.exp(T_delta[:, 2::3]))
    vx = torch.mul(vz, torch.addcdiv(vx_0, T_src[:, 0::3], T_src[:, 2::3]))
    vy = torch.mul(vz, torch.addcdiv(vy_0, T_src[:, 1::3], T_src[:, 2::3]))

    T_tgt = torch.zeros_like(T_src)
    T_tgt[:, 0::3] = vx
    T_tgt[:, 1::3] = vy
    T_tgt[:, 2::3] = vz

    return T_tgt

class FlowNetS(nn.Module):
    def __init__(self, batchNorm=False, has_deconv=False, input_channel_size=0):
        super(FlowNetS, self).__init__()
        self.batchNorm = batchNorm
        self.has_deconv = has_deconv
        self.conv1 = conv(self.batchNorm, input_channel_size, 64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        if has_deconv:
            self.deconv5 = deconv(1024, 512)
            self.deconv4 = deconv(1026, 256)
            self.deconv3 = deconv(770, 128)
            self.deconv2 = deconv(386, 64)

            self.predict_flow6 = predict_flow(1024)
            self.predict_flow5 = predict_flow(1026)
            self.predict_flow4 = predict_flow(770)
            self.predict_flow3 = predict_flow(386)
            self.predict_flow2 = predict_flow(194)

            self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
            self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
            self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
            self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        if self.has_deconv:
            flow6       = self.predict_flow6(out_conv6)
            flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
            out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

            concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
            flow5       = self.predict_flow5(concat5)
            flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
            out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

            concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
            flow4       = self.predict_flow4(concat4)
            flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
            out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

            concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
            flow3       = self.predict_flow3(concat3)
            flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
            out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

            concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
            return out_conv6, concat2
        else:
            return out_conv6, None

class DCMOT(nn.Module):
    expansion = 1

    def __init__(self, num_classes):
        super(DCMOT, self).__init__()

        self.num_classes = num_classes

        # build convs
        self.network_cfg = cfg.NETWORK.REF
        input_channel_size = 6
        depth_channel_size = 1 if self.network_cfg.DEPTH_TYPE == 'z' else 3
        if self.network_cfg.INPUT_OBS_DEPTH:
            input_channel_size += depth_channel_size
            if self.network_cfg.INPUT_OBS_DEPTH_MASK:
                input_channel_size += 1
        if self.network_cfg.INPUT_REND_DEPTH:
            input_channel_size += depth_channel_size
            if self.network_cfg.INPUT_REND_DEPTH_MASK:
                input_channel_size += 1
        if self.network_cfg.INPUT_RGB_DIFF:
            input_channel_size += 3
        if self.network_cfg.INPUT_DEPTH_DIFF:
            input_channel_size += 1
        if self.network_cfg.BASENET_TYPE == 'flownets':
            self.has_deconv = cfg.MODE=='TRAIN' and cfg.LOSS.REF.LW_MASK>0
            self._fcn = FlowNetS(has_deconv=self.has_deconv, input_channel_size=input_channel_size)
            out_channels = 1024
            feat_h = int(cfg.DATA.INPUT_HEIGHT / 64 + 0.5)
            feat_w = int(cfg.DATA.INPUT_WIDTH / 64 + 0.5)
        else:
            raise NotImplementedError

        if self.network_cfg.BASENET_TYPE == 'flownets':
            # build fc
            feat_size = feat_h*feat_w*out_channels
            fc_size = min(self.network_cfg.FC_SIZE, feat_size)
            self.fc6 = fc(feat_size, fc_size, relu=True)
            self.fc7 = fc(fc_size, fc_size, relu=True)
            final_feat_size = fc_size

        # build predictor
        if self.network_cfg.ROT_TYPE == 'quaternion' or self.network_cfg.ROT_TYPE == 'ag1ax3':
            dim_fcr = 4
        elif self.network_cfg.ROT_TYPE == 'axisangle':
            dim_fcr = 3
        elif self.network_cfg.ROT_TYPE == 'a3a3' or self.network_cfg.ROT_TYPE == 'a3xa3':
            dim_fcr = 6
        elif self.network_cfg.ROT_TYPE == 'ortho6d':
            dim_fcr = 6
        dim_fct = 3
        if self.network_cfg.CLASS_AWARE:
            # raise Exception('pick corresponding fcr not implemented')
            dim_fcr *= num_classes
            dim_fct *= num_classes
        self.fcr = fc(final_feat_size, dim_fcr, relu=False)
        self.fct = fc(final_feat_size, dim_fct, relu=False)

        if self.has_deconv:
            self.predict_mask = predict_mask(194, num_classes=num_classes)
            self.mask_upsample = nn.Upsample(scale_factor=4, mode='bilinear')

        # initialize xconvs and fc
        for name, module in self.named_modules():
            if name.startswith('fcn_obs'):
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # identity init predictor
        if self.network_cfg.IDENTITY_INIT:
            self.fcr.weight.data.uniform_()
            self.fcr.weight.data *= 0.01
            if self.network_cfg.ROT_TYPE == 'quaternion':
                self.fcr.weight.data[::4] += 0.03
            self.fcr.bias.data.zero_()

            # self.fct.weight.data.uniform_()
            self.fct.weight.data *= 0.01
            self.fct.bias.data.zero_()

        if self.network_cfg.BASENET_TYPE == 'flownets':
            official_model = 'data/checkpoints/flownets_EPE1.951.pth.tar'
            if os.path.exists(official_model):
                pretrained_dict = torch.load('data/checkpoints/flownets_EPE1.951.pth.tar', map_location=torch.device('cpu'))
                if 'state_dict' in pretrained_dict:
                    pretrained_dict = pretrained_dict['state_dict']

                model_dict = self._fcn.state_dict()
                inherit_data = {}
                for k, v in model_dict.items():
                    k_pretrain = k
                    if k_pretrain in pretrained_dict and v.size() == pretrained_dict[k_pretrain].size():
                        inherit_data[k] = pretrained_dict[k_pretrain]
                if 'conv1.0.weight' not in inherit_data:
                    print("extend the original conv1.0 to support additional input")
                    old_conv1_weight = pretrained_dict['conv1.0.weight']
                    ext_conv1_weight = torch.zeros_like(model_dict['conv1.0.weight'])
                    ext_conv1_weight[:,:6,:,:] = old_conv1_weight
                    inherit_data['conv1.0.weight'] = ext_conv1_weight
                print_and_log("load from offical pretrained weights:"+', '.join(inherit_data.keys()))
                model_dict.update(inherit_data)
                self._fcn.load_state_dict(model_dict)
            else:
                print('no file found at '+official_model)

    def select_channel(self, feature_all, class_idx_list):
        if not self.network_cfg.CLASS_AWARE:
            return feature_all
        selected_shape = list(feature_all.shape)
        valid_num_class = self.num_classes
        channel_each_class = feature_all.shape[1] // valid_num_class
        assert feature_all.shape[1]%valid_num_class == 0, "{}, {}".format(feature_all, valid_num_class)
        selected_shape[1] = channel_each_class
        feature_selected = torch.zeros(selected_shape, device=feature_all.device)
        for i, class_idx in enumerate(class_idx_list):
            class_idx = int(class_idx)
            feature_selected[i, :] \
                = feature_all[i, class_idx*channel_each_class:(class_idx+1)*channel_each_class]
        return feature_selected

    def forward(self, **kwargs):
        """
        :param data_ref: list of dict for each instance
                image: tensor HxWx3, cuda, float32, BGR
                K: tensor 3x3, cpu, float32
                class_idx: int
                pose_rend: tensor 1x7, cuda, float32
                image_rend: tensor HxWx3, cuda, float32
                # depth_rend:
                zoom_image_obs: tensor 1x3xHxW, cuda, float32, BGR
                zoom_image_rend: tensor 1x3xHxW, cuda, float32, BGR
                zoom_image_obs: tensor 1x3xHxW, cuda, float32, BGR
                zoom_image_rend: tensor 1x3xHxW, cuda, float32, BGR
                zoom_factor: tensor 1x4
        :return:
        """
        rgb_obs = kwargs['zoom_image_obs'].detach()
        rgb_rend = kwargs['zoom_image_rend'].detach()
        depth_obs = kwargs['zoom_depth_obs'].detach()
        depth_rend = kwargs['zoom_depth_rend'].detach()
        if self.network_cfg.INPUT_OBS_DEPTH_MASK:
            depth_mask_obs = (depth_obs > 0.05).float().detach()
        if self.network_cfg.INPUT_REND_DEPTH_MASK:
            depth_mask_rend = (depth_rend > 0.05).float().detach()
        class_idx_list = kwargs['class_idx'].cpu().numpy()
        pose_rend = kwargs['pose_rend']
        zoom_factor = kwargs['zoom_factor']
        intrinsic_matrix = kwargs['K']

        inputs = [rgb_obs, rgb_rend]
        if self.network_cfg.INPUT_OBS_DEPTH:
            if self.network_cfg.DEPTH_TYPE == 'z':
                inputs.append(depth_obs)
            else:
                raise KeyError
            if self.network_cfg.INPUT_OBS_DEPTH_MASK:
                inputs.append(depth_mask_obs)
        if self.network_cfg.INPUT_REND_DEPTH:
            if self.network_cfg.DEPTH_TYPE == 'z':
                inputs.append(depth_rend)
            else:
                raise KeyError
            if self.network_cfg.INPUT_REND_DEPTH_MASK:
                inputs.append(depth_mask_rend)
        if self.network_cfg.INPUT_RGB_DIFF:
            rgb_diff = (rgb_rend-rgb_obs)*(depth_rend>0.01)
            inputs.append(rgb_diff)
        if self.network_cfg.INPUT_DEPTH_DIFF:
            depth_diff = (depth_rend-depth_obs)*(depth_rend>0.01)
            inputs.append(depth_diff)
        inputs = torch.cat(inputs, dim=1).detach()

        if self.network_cfg.BASENET_TYPE == 'flownets':
            x, x_deconv = self._fcn(inputs)
            x = self.fc6(x.view(x.shape[0], -1))
            x = self.fc7(x)
        else:
            x = self._fcn(inputs)
            x = x.flatten(start_dim=1)

        # get rot pred
        out_fcr = self.select_channel(self.fcr(x), class_idx_list)
        quaternion = None
        mat = None
        if self.network_cfg.ROT_TYPE == 'quaternion':
            quaternion = torch.nn.functional.normalize(out_fcr)
        else:
            raise NotImplementedError

        # get trans pred
        translation_delta = self.select_channel(self.fct(x), class_idx_list)
        translation = t_transform(pose_rend[:, 4:],
                                  translation_delta,
                                  zoom_factor,
                                  self.num_classes,
                                  intrinsic_matrix)
        #
        # network_output = {'quat': quaternion.split(instance_per_image, dim=0),
        #                   'trans': translation.split(instance_per_image, dim=0)}
        network_output = {'quat': quaternion,
                          'trans': translation,
                          'mat': mat}

        if self.has_deconv:
            masks_all = self.mask_upsample(self.predict_mask(x_deconv))
            mask = self.select_channel(masks_all, class_idx_list)
            network_output['mask'] = mask

        return network_output

    def weight_parameters(self):
        weight_group = []
        for name, param in self.named_parameters():
            if 'weight' in name and not (self.network_cfg.FIX_RESNET_BN and '.bn' in name):
                weight_group.append(param)
        return weight_group

    def bias_parameters(self):
        bias_group = []
        for name, param in self.named_parameters():
            if 'bias' in name and not (self.network_cfg.FIX_RESNET_BN and '.bn' in name):
                bias_group.append(param)
        return bias_group


def dcm_ot_v0(num_classes, pretrained_dict=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = DCMOT(num_classes)

    if pretrained_dict is not None:
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        model_dict = model.state_dict()
        inherit_data = {}
        not_inherit_data = []
        for k, v in model_dict.items():
            if k.startswith('fcn_obs') and k not in pretrained_dict:
                k_pretrain = k[5:]
            else:
                k_pretrain = k
            if k_pretrain in pretrained_dict and v.size() == pretrained_dict[k_pretrain].size():
                inherit_data[k] = pretrained_dict[k_pretrain]
            else:
                not_inherit_data.append(k)

        print_and_log("load from my pretrained weights:"+', '.join(inherit_data.keys()))
        print_and_log("not loaded from my pretrained weights:"+", ".join(not_inherit_data))
        model_dict.update(inherit_data)
        model.load_state_dict(model_dict)
    else:
        print('No pretrained weights!')

    return model
