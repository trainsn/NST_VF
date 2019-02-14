import os
import pdb

from PIL import Image
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable

img = Image.open(os.path.join('content', 'lic.jpg')).convert('L')
img.save('grayscale.jpg')
T=transforms.Compose([transforms.ToTensor()])
P=transforms.Compose([transforms.ToPILImage()])

ten=torch.unbind(T(img))
rsize, csize = ten[0].shape
x=ten[0].unsqueeze(0).unsqueeze(0)

a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))

G_x=conv1(Variable(x)).data.view(1, rsize, csize)

b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
G_y=conv2(Variable(x)).data.view(1, rsize, csize)
pdb.set_trace()

G=torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
X=P(G)
X.save('fake_grad.jpg')