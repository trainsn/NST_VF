import argparse
import numpy as np
from tqdm import tqdm, trange
import os
import pdb

import torch
from torch.optim import Adam
from torch.autograd import Variable

import utils
import lic

def optimize(args):

    # generate the vector field that we want to stylize
    size = args.content_size
    vectors = np.zeros((size, size, 2), dtype=np.float32)
    eps = 1e-7
    for y in range(size):
        for x in range(size):
            xx = float(x - size/2)
            yy = float(y - size/2)
            rsq = xx**2+yy**2
            if (rsq == 0):
                vectors[y, x, 0] = -1
                vectors[y, x, 1] = 1
            else:
                vectors[y, x, 0] = -yy/rsq if yy!=0 else eps
                vectors[y, x, 1] = xx/rsq  if xx!=0 else eps

    output_size = torch.Size([size, size])
    if args.cuda:
        output = Variable(torch.randn(output_size, device="cuda")*30 + 127, requires_grad=True)
    else:
        output = Variable(torch.randn(output_size) * 30 + 127, requires_grad=True)
    optimizer = Adam([output], lr=args.lr)

    tbar = trange(args.iters)
    for e in tbar:
        utils.clamp_batch(output, 0, 255)
        optimizer.zero_grad()
        kernellen = 15
        kernel = np.sin(np.arange(kernellen) * np.pi / kernellen)
        kernel = kernel.astype(np.float32)

        loss = lic.line_integral_convolution(vectors, output, kernel, args.cuda)
        loss.backward()
        optimizer.step()
        tbar.set_description(str(loss.data.cpu().numpy().item()))

        # save the image
        if ((e+1) % args.log_interval == 0):
            utils.tensor_save_gray_image(output.data, "output_iter_" + str(e+1) + ".jpg", args.cuda)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500,
                    help="number of training iterations, default is 500")
    parser.add_argument("--content-size", type=int, default=64,
                        help="size of the vector field. default is 512")
    parser.add_argument("--output-image", type=str, default="output.jpg",
                        help="path for saving the output image")
    parser.add_argument("--cuda", type=int, default=0,
                        help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--lr", type=float, default=1e1,
                        help="learning rate, default is 0.001")
    parser.add_argument("--log-interval", type=int, default=2,
                        help="number of images after which the training loss is logged, default is 50")

    args = parser.parse_args()
    optimize(args)

if __name__ == "__main__":
    main()