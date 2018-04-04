'''Convert pretrained VGG model to SSD.

VGG model download from PyTorch model zoo: https://download.pytorch.org/models/vgg16-397923af.pth
'''
import torch
from torch.autograd import Variable
from layers.ssd import SSD_300
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image


vgg = torch.load('./model/vgg16-397923af.pth')

ssd = SSD_300(5)
layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]

for layer_idx in layer_indices:
    ssd.base[layer_idx].weight.data = vgg['features.%d.weight' % layer_idx]
    ssd.base[layer_idx].bias.data = vgg['features.%d.bias' % layer_idx]

# [24,26,28]
ssd.conv5_1.weight.data = vgg['features.24.weight']
ssd.conv5_1.bias.data = vgg['features.24.bias']
ssd.conv5_2.weight.data = vgg['features.26.weight']
ssd.conv5_2.bias.data = vgg['features.26.bias']
ssd.conv5_3.weight.data = vgg['features.28.weight']
ssd.conv5_3.bias.data = vgg['features.28.bias']

# torch.save(ssd.state_dict(), 'ssd.pth')


class L2Norm2d(nn.Module):
    '''L2Norm layer across all channels.'''
    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        '''out = scale * x / sqrt(\sum x_i^2)'''
        x = torch.mul(x, self.scale)
        y = torch.rsqrt(torch.clamp(torch.sum(torch.pow(x, 2), dim=dim), min=1e-12)).unsqueeze(1)
        return x * y
        # return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)


root1 = 'D:\pycode\ssd_pytorch\data_utils\dataset\\1(1).jpg'
root2 = 'D:\pycode\ssd_pytorch\data_utils\dataset\\1(2).jpg'
a, b = Image.open(root1), Image.open(root2)
a, b = a.resize((300, 300)), b.resize((300, 300))
a, b = np.array(a)[None, :, :, :], np.array(b)[None, :, :, :]
img = np.concatenate((a, b), axis=0)/255-0.5
img = np.transpose(img, axes=(0, 3, 1, 2)).astype(np.float32)
img = torch.from_numpy(img)
img = Variable(img)
ss = []
data = torch.randn(3, 3, 300, 300)
data = Variable(data)
x = img
l2 = L2Norm2d(20)
for idx, layer in enumerate(ssd.base):
    x = layer(x)
    x = l2(x)
    if isinstance(layer, nn.Conv2d):
        print(layer)
        ss.append(x)
# data = torch.randn(3, 3, 300, 300)
# data = Variable(data)
# _, _, a = ssd(data)
print(len(ss))
for i in range(len(ss)):
    b = ss[i].data.numpy()
    b = np.reshape(b, newshape=[-1])
    plt.subplot(2, 5, i+1)
    plt.title('conv_'+str(i+1))
    plt.hist(b, 20)
    # plt.xticks(np.arange(-2.5, 2.5, 0.5))

plt.show()



