import argparse
import numpy as np
from tqdm import tqdm, trange
import os
import pdb
import gc

import torch
from torch.optim import Adam
from torch.autograd import Variable

import utils
import lic
from net import Vgg16

def optimize(args):
    style_image = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style_image = style_image.unsqueeze(0)
    style_image = Variable(utils.preprocess_batch(style_image), requires_grad=False)
    # style_image = utils.subtract_imagenet_mean_batch(style_image)

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
            # vectors[y, x, 0] = -1
            # vectors[y, x, 1] = 1

    # load the pre-trained vgg-16 and extract features
    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, 'vgg16.weight')))
    if args.cuda:
        style_image = style_image.cuda()
        vgg.cuda()
    features_style = vgg(style_image)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # output_size = torch.Size([1, size, size])
    # output = torch.randn(output_size) * 80 + 127
    # if args.cuda:
    #     output = output.cuda()
    # output = output.expand(3, size, size)
    # output = Variable(output, requires_grad=True)
    output_size = torch.Size([3, size, size])
    output = Variable(torch.randn(output_size, device="cuda") * 80 + 127, requires_grad=True)
    optimizer = Adam([output], lr=args.lr)
    mse_loss = torch.nn.MSELoss()

    loss = []
    tbar = trange(args.iters)
    for e in tbar:
        utils.clamp_batch(output, 0, 255)
        optimizer.zero_grad()
        lic_input = output
        kernellen = 15
        kernel = np.sin(np.arange(kernellen) * np.pi / kernellen)
        kernel = kernel.astype(np.float32)

        loss.append(args.content_weight *
                    lic.line_integral_convolution(vectors, lic_input, kernel, args.cuda))

        # vgg_input = output.unsqueeze(0)
        # features_y = vgg(vgg_input)
        # style_loss = 0
        # for m in range(len(features_y)):
        #     gram_y = utils.gram_matrix(features_y[m])
        #     gram_s = Variable(gram_style[m].data, requires_grad=False)
        #     style_loss += args.style_weight * mse_loss(gram_y, gram_s)
        # style_loss.backward()
        # loss[e] += style_loss

        loss[e].backward()
        optimizer.step()
        tbar.set_description(str(loss[e].data.cpu().numpy().item()))

        # save the image
        if ((e+1) % args.log_interval == 0):
            # print("iter: %d content_loss: %f style_loss %f" % (e, loss[e].item(), style_loss.item()))
            utils.tensor_save_bgrimage(output.data, "output_iter_" + str(e+1) + ".jpg", args.cuda)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500,
                    help="number of training iterations, default is 500")
    parser.add_argument("--style-image", type=str, default="../optim/style_images/starry_night.jpg",
                        help="path to style-image")
    parser.add_argument("--content-size", type=int, default=128,
                        help="size of the vector field. default is 512")
    parser.add_argument("--style-size", type=int, default=None,
                        help="size of style image, default is the original size of style image")
    parser.add_argument("--output-image", type=str, default="output.jpg",
                        help="path for saving the output image")
    parser.add_argument("--vgg-model-dir", type=str, default="../optim/models/",
                        help="directory for vgg, if model is not present in the directory it is downloaded")
    parser.add_argument("--cuda", type=int, default=0,
                        help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--content-weight", type=float, default=1,
                        help="weight for content-loss, default is 3.0")
    parser.add_argument("--style-weight", type=float, default=5,
                        help="weight for style-loss, default is 5.0")
    parser.add_argument("--lr", type=float, default=1e2,
                        help="learning rate, default is 0.001")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of images after which the training loss is logged, default is 50")

    args = parser.parse_args()
    optimize(args)

if __name__ == "__main__":
    main()