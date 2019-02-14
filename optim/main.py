import os
import numpy as np
import math
from tqdm import tqdm, trange
import argparse
import pdb

import torch
from torch.optim import Adam
from torch.autograd import Variable

import utils
from net import Vgg16
from transformer_net import TransformerNet

def optimize(args):
    style_image = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style_image = style_image.unsqueeze(0)
    style_image = Variable(utils.preprocess_batch(style_image), requires_grad=False)
    style_image = utils.subtract_imagenet_mean_batch(style_image)

    # generate the vector field that we want to stylize
    size = args.content_size
    vectors = np.zeros((size, size, 2), dtype=np.float32)
    angles = np.zeros((size, size), dtype=np.float32)

    vortex_spacing = 0.5
    extra_factor = 2.

    a = np.array([1, 0]) * vortex_spacing
    b = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)]) * vortex_spacing
    rnv = int(2 * extra_factor / vortex_spacing)
    vortices = [n * a + m * b for n in range(-rnv, rnv) for m in range(-rnv, rnv)]
    vortices = [(x, y) for (x, y) in vortices if -extra_factor < x < extra_factor and -extra_factor < y < extra_factor]

    xs = np.linspace(-1, 1, size).astype(np.float32)[None, :]
    ys = np.linspace(-1, 1, size).astype(np.float32)[:, None]

    for (x, y) in vortices:
        rsq = (xs - x) ** 2 + (ys - y) ** 2
        vectors[..., 0] += (ys - y) / rsq
        vectors[..., 1] += -(xs - x) / rsq

    for y in range(size):
        for x in range(size):
            angles[y, x] = math.atan(vectors[y, x, 1] / vectors[y, x, 0]) * 180 / math.pi

    # for y in range(size):
    #     for x in range(size):
    #         xx = float(x - size / 2)
    #         yy = float(y - size / 2)
    #         rsq = xx ** 2 + yy ** 2
    #         if (rsq == 0):
    #             vectors[y, x, 0] = 0
    #             vectors[y, x, 1] = 0
    #         else:
    #             vectors[y, x, 0] = -yy / rsq
    #             vectors[y, x, 1] = xx / rsq
    #             angles[y, x] = math.atan(vectors[y, x, 1] / vectors[y, x, 0]) * 180 / math.pi
            # angles[y, x] = 45
    angles = utils.tensor_load_vector_field(angles)
    angles = angles.unsqueeze(0)
    angles = Variable(angles, requires_grad=False)

    # load the pre-trained vgg-16 and extract features
    vgg = Vgg16()
    utils.init_vgg16(args.vgg_model_dir)
    vgg.load_state_dict(torch.load(os.path.join(args.vgg_model_dir, 'vgg16.weight')))
    if args.cuda:
        style_image = style_image.cuda()
        vgg.cuda()
    features_style = vgg(style_image)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # load the transformer net and extract features
    transformer = TransformerNet()
    transformer.load_state_dict(torch.load(args.transformer_model_path))
    if args.cuda:
        angles = angles.cuda()
        transformer.cuda()

    # init optimizer
    angles_size = angles.data.size()
    output_size = np.asarray(angles_size)
    output_size[1] = 3
    output_size = torch.Size(output_size)
    output = Variable(torch.randn(output_size, device="cuda"), requires_grad=True)
    optimizer = Adam([output], lr=args.lr)
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    # optimize the images
    tbar = trange(args.iters)
    for e in tbar:
        utils.imagenet_clamp_batch(output, 0, 255)
        optimizer.zero_grad()
        transformer_input = utils.gray_bgr_batch(output)
        transformer_y = transformer(transformer_input)
        content_loss = args.content_weight * l1_loss(angles, transformer_y)

        vgg_input = output
        features_y = vgg(vgg_input)
        style_loss = 0
        for m in range(len(features_y)):
            gram_y = utils.gram_matrix(features_y[m])
            gram_s = Variable(gram_style[m].data, requires_grad=False)
            style_loss += args.style_weight * mse_loss(gram_y, gram_s)

        total_loss = content_loss + style_loss
        # total_loss = content_loss
        if ((e+1) % args.log_interval == 0):
            print("iter: %d content_loss: %f style_loss %f" % (e, content_loss.item(), style_loss.item()))
        total_loss.backward()
        optimizer.step()
        tbar.set_description(str(total_loss.data.cpu().numpy().item()))
    # save the image
    output = utils.add_imagenet_mean_batch_device(output, args.cuda)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500,
                        help="number of training iterations, default is 500")
    parser.add_argument("--style-image", type=str, default="style_images/22.jpg",
                        help="path to style-image")
    parser.add_argument("--content-size", type=int, default=512,
                        help="size of the vector field. default is 512")
    parser.add_argument("--style-size", type=int, default=None,
                        help="size of style image, default is the original size of style image")
    parser.add_argument("--output-image", type=str, default="output.jpg",
                        help="path for saving the output image")
    parser.add_argument("--transformer-model-path", type=str,default="../save_models/epoch_5.model",
                        help="path for transformer net, trained before")
    parser.add_argument("--vgg-model-dir", type=str, default="models/",
                        help="directory for vgg, if model is not present in the directory it is downloaded")
    parser.add_argument("--cuda", type=int, default=1,
                        help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--content-weight", type=float, default=1,
                        help="weight for content-loss, default is 3.0")
    parser.add_argument("--style-weight", type=float, default=5,
                        help="weight for style-loss, default is 5.0")
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