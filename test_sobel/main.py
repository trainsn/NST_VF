import argparse
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.autograd import Variable

import utils
from net import Sobel, CosineLoss
def optimize(args):
    # generate the vector field that we want to backward from
    size = args.content_size
    vectors = np.zeros((size, size, 2), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            xx = float(x - size / 2)
            yy = float(y - size / 2)
            rsq = xx ** 2 + yy ** 2
            if rsq == 0:
                vectors[y, x, 0] = 1
                vectors[y, x, 1] = 1
            else:
                vectors[y, x, 0] = -yy / rsq
                vectors[y, x, 1] = xx / rsq
    vectors = utils.tensor_load_vector_field(vectors)

    # load the sobel network
    sobel = Sobel()
    if args.cuda:
        vectors = vectors.cuda()
        sobel.cuda()

    # init optimizer
    vectors_size = vectors.data.size()
    output_size = np.asarray(vectors_size)
    output_size[1] = 1
    output_size = torch.Size(output_size)
    output = Variable(torch.randn(output_size, device="cuda") + 127, requires_grad=True)
    optimizer = Adam([output], lr=args.lr)
    cosine_loss =  CosineLoss()

    #optimize the images
    tbar = trange(args.iters)
    for e in tbar:
        utils.clamp_batch(output, 0, 255)
        optimizer.zero_grad()
        y = sobel(output)
        loss = cosine_loss(vectors, y)
        loss.backward()
        optimizer.step()
        if ((e+1) % args.log_interval == 0):
            print("iter: %d content_loss: %f" % (e, loss.item()))
        tbar.set_description(str(loss.data.cpu().numpy().item()))
    utils.tensor_save_gray_image(output.data[0], args.output_image, args.cuda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500,
                        help="number of training iterations, default is 500")
    parser.add_argument("--content-size", type=int, default=512,
                        help="size of the vector field. default is 512")
    parser.add_argument("--output-image", type=str, default="output.jpg",
                        help="path for saving the output image")
    parser.add_argument("--cuda", type=int, default=1,
                        help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--lr", type=float, default=1e1,
                        help="learning rate, default is 0.001")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="number of images after which the training loss is logged, default is 50")

    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    optimize(args)

if __name__ == "__main__":
   main()