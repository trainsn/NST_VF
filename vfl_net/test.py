import pyximport
pyximport.install()

import argparse
import pdb
import re
from PIL import Image
import scipy.misc
import numpy as np
import math

import torch
from torchvision import transforms

import utils
from transformer_net import TransformerNet
import dataset
import lic_internal

def lic(vectors, output_name):
    rsize, csize, _ = vectors.shape

    kernellen=20
    kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
    kernel = kernel.astype(np.float32)
    texture = np.random.rand(rsize, csize).astype(np.float32)

    image = lic_internal.line_integral_convolution(vectors, texture, kernel)
    scipy.misc.imsave(output_name, image)


def vectorize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = Image.open(args.content_image).convert('L')
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    content_image = utils.subtract_imagenet_mean_batch(content_image)
    content_image = content_image.to(device)

    with torch.no_grad():
        vectorize_model = TransformerNet()
        state_dict = torch.load(args.saved_model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                pdb.set_trace()
                del state_dict[k]
        vectorize_model.load_state_dict(state_dict)
        vectorize_model.to(device)
        output = vectorize_model(content_image)

    target = dataset.hdf5_loader(args.target_vector)
    target_transform = transforms.ToTensor()
    target = target_transform(target)
    target = target.unsqueeze(0).to(device)

    cosine_loss = torch.nn.CosineEmbeddingLoss()
    label = torch.ones(1, 1, args.size, args.size).to(device)
    loss = cosine_loss(output, target, label)
    print(loss.item())

    pdb.set_trace()
    output = output.cpu().clone().numpy()[0].transpose(1, 2, 0)
    lic(output, "output.jpg")
    target = target.cpu().clone().numpy()[0].transpose(1, 2, 0)
    lic(target, "target.jpg")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-image", type=str, default="../datasets/train_gray/COCO_train2014_000000035427.jpg",
                        help="path to the original image")
    parser.add_argument("--size", type=int, default=512,
                        help="size the the image and vector field")
    parser.add_argument("--target-vector", type=str, default="../datasets/vector_fields/COCO_train2014_000000035427.h5",
                        help="path to the target vector field")
    parser.add_argument("--saved-model", type=str, default="../save_models/epoch_5.model",
                        help="saved model to be used for vectorize the image. ")
    parser.add_argument("--cuda", type=int, default=1,
                        help="set it to 1 for running on GPU, 0 for CPU")

    args = parser.parse_args()
    vectorize(args)

if __name__ == "__main__":
    main()