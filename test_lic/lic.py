import torch
from torch.autograd import Variable

def _advance(vx, vy, x, y, fx, fy, w, h):
    if vx>=0:
        tx = (1-fx)/vx
    else:
        tx = -fx/vx
    if vy>=0:
        ty = (1-fy)/vy
    else:
        ty = -fy/vy
    if tx<ty:
        if vx>=0:
            x+=1
            fx=0
        else:
            x-=1
            fx=1
        fy+=tx*vy
    else:
        if vy>=0:
            y+=1
            fy=0
        else:
            y-=1
            fy=1
        fx+=ty*vx
    if x>=w:
        x=w-1 # FIXME: other boundary conditions?
    if x<0:
        x=0 # FIXME: other boundary conditions?
    if y<0:
        y=0 # FIXME: other boundary conditions?
    if y>=h:
        y=h-1 # FIXME: other boundary conditions?
    return x, y, fx, fy

def line_integral_convolution(vectors, texture, kernel, cuda):
    h = vectors.shape[0]
    w = vectors.shape[1]
    kernellen = kernel.shape[0]

    sline = torch.zeros(h, w)
    if cuda:
        sline = sline.cuda()

    for i in range(h):
        for j in range(w):
            x = j
            y = i
            fx = 0.5
            fy = 0.5

            k = kernellen // 2
            sline[i, j] += texture[y, x].item() * kernel[k]
            while k < kernellen - 1:
                x, y, fx, fy = _advance(vectors[y, x, 0], vectors[y, x, 1],
                                        x, y, fx, fy, w, h)
                # print(i, j, k, y, x)
                k += 1
                sline[i, j] += texture[y, x].item() * kernel[k]

            x = j
            y = i
            fx = 0.5
            fy = 0.5

            k = kernellen // 2
            while k > 0:
                x, y, fx, fy = _advance(-vectors[y, x, 0], -vectors[y, x, 1],
                                        x, y, fx, fy, w, h)
                k -= 1
                # print(i, j, k, y, x)
                sline[i, j] += texture[y, x].item() * kernel[k]
            sline[i, j] /= float(kernel.sum())

    loss = torch.sum(torch.pow((sline - texture), 2))
    return loss




