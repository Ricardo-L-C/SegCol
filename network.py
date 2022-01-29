from numpy import full_like
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from model.se_resnet import BottleneckX, SEResNeXt
from model.options import DEFAULT_NET_OPT

class MultiPrmSequential(nn.Sequential):
    def __init__(self, *args):
        super(MultiPrmSequential, self).__init__(*args)

    def forward(self, input, cat_feature):
        for module in self._modules.values():
            input = module(input, cat_feature)
        return input

def make_secat_layer(block, in_channels, out_channels, cat_channels, block_count, no_bn=False):
    inner_channels = out_channels // 4
    downsample = None
    if in_channels != out_channels:
        if no_bn:
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False))
        else:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    layers = []
    layers.append(block(in_channels, inner_channels, cat_channels, 16, 1, downsample, no_bn=no_bn))
    for _ in range(1, block_count):
        layers.append(block(out_channels, inner_channels, cat_channels, 16, no_bn=no_bn))

    return MultiPrmSequential(*layers)

class SeCatLayer(nn.Module):
    def __init__(self, channel, cat_channels, reduction=16):
        super(SeCatLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(cat_channels + channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, cat_feature):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.cat([y, cat_feature], 1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SECatBottleneckX(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, cat_channels, cardinality=16, stride=1, downsample=None, no_bn=False):
        super(SECatBottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=cardinality, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)

        self.bn1 = self.bn2 = self.bn3 = None
        if not no_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.se_layer = SeCatLayer(out_channels * self.expansion, cat_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, cat_feature):
        residual = x
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.bn3 is not None:
            out = self.bn3(out)

        out = self.se_layer(out, cat_feature)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FeatureConv(nn.Module):
    def __init__(self, input_dim=512, output_dim=256, input_size=32, output_size=16, net_opt=DEFAULT_NET_OPT):
        super(FeatureConv, self).__init__()

        no_bn = not net_opt['bn']

        if input_size == output_size * 4:
            stride1, stride2 = 2, 2
        elif input_size == output_size * 2:
            stride1, stride2 = 2, 1
        else:
            stride1, stride2 = 1, 1

        seq = []
        seq.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride1, padding=1, bias=False))
        if not no_bn:
            seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride2, padding=1, bias=False))
        if not no_bn:
            seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        seq.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*seq)

    def forward(self, x):
        return self.network(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, color_fc_out, block_num, no_bn):
        super(DecoderBlock, self).__init__()
        self.secat_layer = make_secat_layer(SECatBottleneckX, in_channels, out_channels, color_fc_out, block_num, no_bn=no_bn)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x, cat_feature):
        out = self.secat_layer(x, cat_feature)
        return self.ps(out)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformerEncoder(nn.Module):
    def __init__(self, dim_input, dim_output, num_inds=32, dim_hidden=128, num_heads=8, ln=False):
        super(SetTransformerEncoder, self).__init__()
        # self.positional = PositionalEncoding(19)
        # self.dropout = nn.Dropout(0.1)

        self.enc = nn.Sequential(
            SAB(dim_input, dim_output, num_heads, ln=ln),
            SAB(dim_output, dim_output, num_heads, ln=ln),
        )

        self.linear = nn.Linear(23 * dim_output, dim_output)

        # self.dec = nn.Sequential(
        #         PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         nn.Linear(dim_hidden, dim_output))

    def forward(self, x):
        # return self.dec(self.enc(X))
        x = x.view(-1, 23, 19)
        # x = self.dropout(self.positional(x))
        x = self.enc(x)
        x = x.view(-1, 23*64)
        x = self.linear(x)
        return x

class SimpleDecoderBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(SimpleDecoderBlock, self).__init__()
        self.decoder = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.decoder(x)
        return self.ps(x)

class Generator(nn.Module):
    def __init__(self, input_size, cv_class_num, iv_class_num, input_dim=1, output_dim=3,
                 layers=[12, 8, 5, 5], net_opt=DEFAULT_NET_OPT):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cv_class_num = cv_class_num
        self.iv_class_num = iv_class_num

        self.link_num = 23 * 19

        self.input_size = input_size
        self.layers = layers

        self.cardinality = 16

        self.bottom_h = self.input_size // 16
        # self.Linear = nn.Linear(cv_class_num + self.link_num, self.bottom_h*self.bottom_h*32)
        # self.Linear = nn.Linear(self.link_num, self.bottom_h*self.bottom_h*32)

        self.color_fc_out = 64
        self.net_opt = net_opt

        no_bn = not net_opt['bn']

        # if net_opt['relu']:
        #     self.colorFC = nn.Sequential(
        #         # nn.Linear(cv_class_num + self.link_num, self.color_fc_out), nn.ReLU(inplace=True),
        #         nn.Linear(self.link_num, self.color_fc_out), nn.ReLU(inplace=True),
        #         nn.Linear(self.color_fc_out, self.color_fc_out), nn.ReLU(inplace=True),
        #         nn.Linear(self.color_fc_out, self.color_fc_out), nn.ReLU(inplace=True),
        #         nn.Linear(self.color_fc_out, self.color_fc_out)
        #     )
        # else:
        #     self.colorFC = nn.Sequential(
        #         # nn.Linear(cv_class_num + self.link_num, self.color_fc_out),
        #         nn.Linear(self.link_num, self.color_fc_out),
        #         nn.Linear(self.color_fc_out, self.color_fc_out),
        #         nn.Linear(self.color_fc_out, self.color_fc_out),
        #         nn.Linear(self.color_fc_out, self.color_fc_out)
        #     )

        self.set_transformer = SetTransformerEncoder(19, 64)

        self.conv1 = self.make_encoder_block(self.input_dim, 16, True)
        self.conv2 = self.make_encoder_block(16, 32)
        self.conv3 = self.make_encoder_block(32, 64)
        self.conv4 = self.make_encoder_block(64, 128)
        self.conv5 = self.make_encoder_block(128, 256)

        bottom_layer_len = 256 #+ 64
        if net_opt["cit"]:
            bottom_layer_len += 256

        self.deconv1 = DecoderBlock(bottom_layer_len, 4*256, self.color_fc_out, self.layers[0], no_bn=no_bn) # (output_size / 8, output_size / 8)
        self.deconv2 = DecoderBlock(256 + 128 + 256, 4*128, self.color_fc_out, self.layers[1], no_bn=no_bn) # (output_size / 4, output_size / 4)
        self.deconv3 = DecoderBlock(128 + 64 + 128, 4*64, self.color_fc_out, self.layers[2], no_bn=no_bn) # (output_size / 2, output_size / 2)
        self.deconv4 = DecoderBlock(64 + 32 + 64, 4*32, self.color_fc_out, self.layers[3], no_bn=no_bn) # (output_size, output_size)
        self.deconv5 = nn.Sequential(
            nn.Conv2d(32 + 16 + 32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

        self.ske_deconv1 = SimpleDecoderBlock(bottom_layer_len, 4*256)
        self.ske_deconv2 = SimpleDecoderBlock(256 + 128, 4*128)
        self.ske_deconv3 = SimpleDecoderBlock(128 + 64, 4*64)
        self.ske_deconv4 = SimpleDecoderBlock(64 + 32, 4*32)
        self.ske_deconv5 = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh(),
        )

        if net_opt['cit']:
            self.featureConv = FeatureConv(net_opt=net_opt)

        # self.colorConv = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.Tanh(),
        # )

        if net_opt['guide']:
            self.deconv_for_decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # output is 64 * 64
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # output is 128 * 128
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # output is 256 * 256
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1, output_padding=0), # output is 256 * 256
                nn.Tanh(),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def unrequire_seg_branch(self):
        for i in self.ske_deconv1.parameters():
            i.requires_grad = False
        for i in self.ske_deconv2.parameters():
            i.requires_grad = False
        for i in self.ske_deconv3.parameters():
            i.requires_grad = False
        for i in self.ske_deconv4.parameters():
            i.requires_grad = False
        for i in self.ske_deconv5.parameters():
            i.requires_grad = False

    def make_encoder_block(self, in_channels, out_channels, first=False):
        if first:
            stride = 1
        else:
            stride = 2

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input, feature_tensor, c_tag_class, link, require=0): # 0: all, 1: main, 2: main, guide
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        # ==============================
        # it's about color variant tag set
        # temporally, don't think about noise z

        # c_tag_class = torch.cat((c_tag_class, link), 1)
        c_tag_class = link

        # c_tag_tensor = self.Linear(c_tag_class)
        # c_tag_tensor = c_tag_tensor.view(-1, 32, self.bottom_h, self.bottom_h)
        # c_tag_tensor = self.colorConv(c_tag_tensor)

        # c_se_tensor = self.colorFC(c_tag_class)
        c_se_tensor = self.set_transformer(c_tag_class)

        # ==============================
        # Convolution Layer for Feature Tensor

        # if self.net_opt['cit']:
        #     feature_tensor = self.featureConv(feature_tensor)
        #     concat_tensor = torch.cat([out5, feature_tensor, c_tag_tensor], 1)
        # else:
        #     concat_tensor = torch.cat([out5, c_tag_tensor], 1)

        if self.net_opt['cit']:
            feature_tensor = self.featureConv(feature_tensor)
            concat_tensor = torch.cat([out5, feature_tensor], 1)
        else:
            concat_tensor = out5

        out4_ske = self.ske_deconv1(concat_tensor)
        out4_prime = self.deconv1(concat_tensor, c_se_tensor)

        # ==============================
        # Deconv layers

        out3_ske = self.ske_deconv2(torch.cat([out4_ske, out4], 1))
        out3_prime = self.deconv2(torch.cat([out4_prime, out4_ske, out4], 1), c_se_tensor)

        out2_ske = self.ske_deconv3(torch.cat([out3_ske, out3], 1))
        out2_prime = self.deconv3(torch.cat([out3_prime, out3_ske, out3], 1), c_se_tensor)

        out1_ske = self.ske_deconv4(torch.cat([out2_ske, out2], 1))
        out1_prime = self.deconv4(torch.cat([out2_prime, out2_ske, out2], 1), c_se_tensor)

        full_output = self.deconv5(torch.cat([out1_prime, out1_ske, out1], 1))

        if require == 1:
            return full_output

        # ==============================
        # out4_prime should be input of Guide Decoder

        if self.net_opt['guide']:
            decoder_output = self.deconv_for_decoder(out4_prime)
        else:
            decoder_output = full_output

        if require == 2:
            return full_output, decoder_output

        skeleton_output = self.ske_deconv5(torch.cat([out1_ske, out1], 1))

        return full_output, decoder_output, skeleton_output

class Discriminator(nn.Module):
    def __init__(self, input_dim=6, output_dim=1, input_size=256, cv_class_num=115, iv_class_num=370, net_opt=DEFAULT_NET_OPT):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        # self.cv_class_num = cv_class_num + 23 * 19
        self.cv_class_num = 23 * 19
        self.iv_class_num = iv_class_num
        self.cardinality = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 2, 0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = self._make_block_1(32, 64)
        self.conv3 = self._make_block_1(64, 128)
        self.conv4 = self._make_block_1(128, 256)
        self.conv5 = self._make_block_1(256, 512)
        self.conv6 = self._make_block_3(512, 512)
        self.conv7 = self._make_block_3(512, 512)
        self.conv8 = self._make_block_3(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.cit_judge = nn.Sequential(
            nn.Linear(512, self.iv_class_num),
            nn.Sigmoid()
        )

        self.cvt_judge = nn.Sequential(
            nn.Linear(512, self.cv_class_num),
            nn.Sigmoid()
        )

        self.adv_judge = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_block_1(self, inplanes, planes):
        return nn.Sequential(
            SEResNeXt._make_layer(self, BottleneckX, planes//4, 2, inplanes=inplanes),
            nn.Conv2d(planes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _make_block_2(self, inplanes, planes):
        return nn.Sequential(
            SEResNeXt._make_layer(self, BottleneckX, planes//4, 2, inplanes=inplanes),
        )

    def _make_block_3(self, inplanes, planes):
        return nn.Sequential(
            SEResNeXt._make_layer(self, BottleneckX, planes//4, 1, inplanes=inplanes),
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        cit_judge = self.cit_judge(out)
        cvt_judge = self.cvt_judge(out)
        adv_judge = self.adv_judge(out)

        return adv_judge, cit_judge, cvt_judge
