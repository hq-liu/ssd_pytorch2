import torch
from torch import nn
from layers import multibox_layer
from torch.nn import functional as F
from torch.autograd import Variable


class SSD_300(nn.Module):
    def __init__(self, num_classes):
        super(SSD_300, self).__init__()
        self.num_classes = num_classes
        self.base = self.VGG16()

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

        num_anchors = [4, 6, 6, 6, 4, 4]
        in_planes = [512, 1024, 512, 256, 256, 256]
        self.multibox_layer = multibox_layer.multibox_layer(num_classes=num_classes,
                                                            num_anchors=num_anchors,
                                                            in_planes=in_planes)

    def forward(self, x):
        hs = []
        # h = self.base(x)
        ss = []
        for idx, layer in enumerate(self.base):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                ss.append(x)
        h = x
        hs.append(h)

        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)  # conv7

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)  # conv8_2

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)  # conv9_2

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)  # conv10_2

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)  # conv11_2

        loc_preds, conf_preds = self.multibox_layer(hs)
        return loc_preds, conf_preds, ss

    def VGG16(self):
        '''VGG16 layers.'''
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                layers += [nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)


if __name__ == '__main__':
    a = torch.rand(4, 3, 300, 300)
    a = Variable(a)
    model = SSD_300(21)
    b, c = model(a)
    print(b, c)
    