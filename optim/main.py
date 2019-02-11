import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import argparse

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

def optimize(args):


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500,
                        help="number of training iterations, default is 500")
    parser.add_argument("--style-image", type=str, default="style_images/candy.jpg",
                        help="path to style-image")
    parser.add_argument("--content-size", type=int, default=512,
                        help="size of the vector field. default is 512")
    parser.add_argument("--output-image", type=str, default="output.jpg",
                        help="path for saving the output image")
    parser.add_argument("--transformer-model-path", type=str,default="../save_models/",
                        help="path for transformer net, trained before")
    parser.add_argument("--vgg-model-dir", type=str, default="models/",
                        help="directory for vgg, if model is not present in the directory it is downloaded")
    parser.add_argument("--cuda", type=int, defaul=1,
                        help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--content-weight", type=float, default=3.0,
                        help="weight for content-loss, default is 3.0")
    parser.add_argument("--style-weight", type=float, default=5.0,
                        help="weight for style-loss, default is 5.0")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate, default is 0.001")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="number of images after which the training loss is logged, default is 50")

    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    optimize(args)


if __name__ == "__main__":
   main()