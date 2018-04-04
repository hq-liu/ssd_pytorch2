# import math
# import itertools
# scale = 300.
# steps = [s / scale for s in (8, 16, 32, 64, 100, 300)]  # step*feature_map_sizes = 300
# sizes = [s / scale for s in (30, 60, 111, 162, 213, 264, 315)]
# aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
# feature_map_sizes = (38, 19, 10, 5, 3, 1)
#
# num_layers = len(feature_map_sizes)
#
# boxes = []
# for i in range(num_layers):
#     fmsize = feature_map_sizes[i]
#     for h,w in itertools.product(range(fmsize), repeat=2):
#         cx = (w + 0.5)*steps[i]
#         cy = (h + 0.5)*steps[i]  # default box中心点
#
#         s = sizes[i]  # feature map index
#         boxes.append((cx, cy, s, s))
#
#         s = math.sqrt(sizes[i] * sizes[i+1])
#         boxes.append((cx, cy, s, s))
#
#         s = sizes[i]
#         for ar in aspect_ratios[i]:
#             boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
#             boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
#
# for i in boxes:
#     print(i)

import torch
a = torch.randn(3, 3, 2)
print(torch.mul(a, 2).size())
