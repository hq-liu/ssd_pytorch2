from layers.ssd import SSD_300
from layers.multibox_loss import multibox_loss
from torch import optim
from data_utils.data_input import ListDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import itertools
import torch


def train():
    net = SSD_300(5)
    net.load_state_dict(torch.load('./layers/ssd.pth'))
    not_train_param = [net.base.parameters(), net.conv5_1.parameters(),
                       net.conv5_2.parameters(), net.conv5_3.parameters()]
    for param in itertools.chain(*not_train_param):
        param.requires_grad = False
    criterion = multibox_loss(5)
    train_param = [net.conv6.parameters(), net.conv7.parameters(),
                   net.conv8_1.parameters(), net.conv8_2.parameters(),
                   net.conv9_1.parameters(), net.conv9_2.parameters(),
                   net.conv10_1.parameters(), net.conv10_2.parameters(),
                   net.conv11_1.parameters(), net.conv11_2.parameters()]
    optimizer = optim.Adam(itertools.chain(*train_param), lr=1e-3)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    trainset = ListDataset(root='E:\\st',
                           list_file='./data_utils/data2.txt',
                           train=True,
                           transform=transform)
    print(len(trainset))
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
    train_loss = 0
    for ep in range(1, 11):
        print('ep:', ep)
        for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainloader):
            print(batch_idx, images.size(), loc_targets.size(), conf_targets.size())
            pass
            images = Variable(images)
            # print(images)
            loc_targets = Variable(loc_targets)
            conf_targets = Variable(conf_targets)
            optimizer.zero_grad()
            loc_preds, conf_preds = net(images)
            
            loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
            if loss.data[0] != 0:
                loss.backward()
                optimizer.step()
            train_loss += loss.data[0]
            print('%.3f %.3f' % (loss.data[0], train_loss/(batch_idx+1)))
        print('\n')
    torch.save(net.state_dict(), './layers/model/ssd300_4.pth')


if __name__ == '__main__':
    train()

