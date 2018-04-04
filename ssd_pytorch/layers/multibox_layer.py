from torch import nn
import torch


class multibox_layer(nn.Module):
    def __init__(self, num_classes, num_anchors, in_planes):
        """
        MultiBox layer
        :param num_classes: Number of the classes(#objects + 1 background)
        :param num_anchors: #default boxes of each feature maps
        :param in_planes: #input channels
        """
        super(multibox_layer, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_planes = in_planes

        self.conf_layers = nn.ModuleList()
        self.loc_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.conf_layers.append(nn.Conv2d(in_channels=in_planes[i], out_channels=self.num_anchors[i]*self.num_classes,
                                              kernel_size=3, padding=1))
            self.loc_layers.append(nn.Conv2d(in_channels=in_planes[i], out_channels=self.num_anchors[i]*4,
                                             kernel_size=3, padding=1))

    def forward(self, xs):
        """
        Forward function
        :param xs: input feature maps
        :return: loc_preds——[8732, 4] conf_preds——[8732, #classes]
        """
        y_confs, y_locs = [], []
        for i, x in enumerate(xs):
            y_conf = self.conf_layers[i](x)
            N = y_conf.size(0)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.num_classes)
            y_confs.append(y_conf)

            y_loc = self.loc_layers[i](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)
        conf_preds, loc_preds = torch.cat(y_confs, dim=1), torch.cat(y_locs, dim=1)
        return loc_preds, conf_preds