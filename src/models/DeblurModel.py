import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from swin_transformer import swin_t

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, dilated=False):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.interpolate(out, scale_factor=2)


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # H/32, W/32
        self.up_block1 = nn.Sequential(
            BottleneckBlock(in_channels[-1], 256),
            TransitionBlock(in_channels[-1]+256, 128)
        )
        # H/16, W/16
        self.up_block2 = nn.Sequential(
            BottleneckBlock(in_channels[-2]+128, 128),
            TransitionBlock(in_channels[-2]+256, 128)
        )
        # H/8, W/8
        self.up_block3 = nn.Sequential(
            BottleneckBlock(in_channels[-3]+128, 64),
            TransitionBlock(in_channels[-3]+192, 64)
        )
        # H/4, W/4
        self.up_block4 = nn.Sequential(
            BottleneckBlock(in_channels[-4]+64, 32),
            TransitionBlock(in_channels[-4]+96, 32),
            BottleneckBlock(32, 16),
            TransitionBlock(48, 16),
        )
        # H, W
        self.conv_ref = nn.Conv2d(16+3, 20, 3, 1, 1)
        self.bn = nn.BatchNorm2d(20)
        self.relu = nn.LeakyReLU()
        self.conv_fin = nn.Conv2d(20, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, feats):
        # H/32, W/32
        x = torch.cat((self.up_block1(feats[-1]), feats[-2]), axis=1)
        # H/16, W/16
        x = torch.cat((self.up_block2(x), feats[-3]), axis=1)
        # H/8, W/8
        x = torch.cat((self.up_block3(x), feats[-4]), axis=1)
        # H/4, W/4
        x = torch.cat((self.up_block4(x), feats[-5]), axis=1)
        # H, W
        x = self.relu(self.bn(self.conv_ref(x)))
        x = self.conv_fin(x)
        x = self.tanh(x)
        
        return x

# Wrapper class of ResNet50
class resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=False)
        self.base = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.base(x)


class DeblurModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.feats = []

        if encoder == 'swin_t':
            self.encoder = swin_t()
            self.encoder.stage1.register_forward_hook(self.hook)
            self.encoder.stage2.register_forward_hook(self.hook)
            self.encoder.stage3.register_forward_hook(self.hook)
            self.encoder.stage4.register_forward_hook(self.hook)
            in_channels = [3,96,192,384,768]

        elif encoder == 'resnet50':
            self.encoder = resnet50()
            self.encoder.base[-4].register_forward_hook(self.hook)
            self.encoder.base[-3].register_forward_hook(self.hook)
            self.encoder.base[-2].register_forward_hook(self.hook)
            self.encoder.base[-1].register_forward_hook(self.hook)
            in_channels = [3,256,512,1024,2048]

        else:
            raise Exception("Not implemented")

        self.decoder = Decoder(in_channels)
    
    def hook(self, model, input, output):
        self.feats.append(output)

    def forward(self, x):
        self.feats = [x]
        _ = self.encoder(x)
        output = self.decoder(self.feats)
        return output

if __name__ == '__main__':
    model = DeblurModel('swin_t')
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
