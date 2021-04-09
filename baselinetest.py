
from src.nets import ST_GCN
import time
import torch
import numpy as np
from torch.backends import cudnn
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from src.graph import Graph
from src.utils import check_gpu


gcn_kernel_size =[5,2]
graph = Graph(max_hop=gcn_kernel_size[1])
# device = check_gpu([0,1])
device = check_gpu([0])
A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False).to(device)


a = ST_GCN((9,300,25,2), 120, A, 0.5, gcn_kernel_size)
checkpoint = torch.load('./models/baseline_NTUcset.pth.tar')


# a = ST_GCN((3,300,25,2), 60, A, 0.5, gcn_kernel_size)
# a = nn.DataParallel(a)
# checkpoint = torch.load('./models/org/baseline_NTUcset.pth.tar')

# checkpoint = torch.load('/home/peter/workspace/projects/1.closed/ECCV20/st-gcn/models/st_gcn.ntu-xsub.pt')
# a.load_state_dict(checkpoint['model'])
# a.module.load_state_dict(checkpoint)
# a.module.load_state_dict(checkpoint['model'])
a.load_state_dict(checkpoint['model'])

# stgcn.load_state_dict(checkpoint['model'])
