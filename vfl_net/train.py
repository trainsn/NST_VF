import argparse
import os
import sys
import time
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from dataset import VFDataset
from transformer_net import TransformerNet, CosineLoss

import pdb

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    target_transform = transforms.ToTensor()

    train_dataset = VFDataset(args.dataset, transform, target_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    if args.load_model is not None:
        transformer.load_state_dict(torch.load(args.load_model))
    optimizer = Adam(transformer.parameters(), args.lr)
    # mse_loss = torch.nn.MSELoss()
    cosine_loss = CosineLoss()

    # log_file = open(args.log_file, "w")

    for e in range(args.epochs):
        transformer.train()
        agg_loss = 0.
        count = 0
        for batch_id, (x, vf) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = utils.subtract_imagenet_mean_batch(x)
            x = x.to(device)
            y = transformer(x)
            vf = vf.to(device)

            # loss = mse_loss(y, vf)
            loss = cosine_loss(y, vf)
            loss.backward()
            optimizer.step()

            agg_loss += loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_loss / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5,
                        help="number of training epochs, default is 5")
    parser.add_argument("--batch-size", type=int, default=7,
                        help="batch size for training, default is 7")
    parser.add_argument("--dataset", type=str, default="../datasets",
                        help="path to training dataset, the path should point to a folder containing "
                             "train_data and ground truth vector fields")
    parser.add_argument("--save-model-dir", type=str, default="../save_models",
                        help="path to folder where trained model will be saved")
    parser.add_argument("--checkpoint-model-dir", type=str, default="../save_models",
                        help="path to folder where checkpoints of trained model will be saved")
    parser.add_argument("--load-model", type=str, default=None,
                        help="path to the model param that want to be loaded")
    parser.add_argument("--image-size", type=int, default=512,
                        help="size of training image, default is 512 X 512")
    parser.add_argument("--cuda", type=int, default=1,
                        help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate, default is 1e-3")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="number of images after which the training loss is logged, default is 50")
    parser.add_argument("--checkpoint-interval", type=int, default=200,
                        help="number of batches after which a checkpoint of the trained model will be created")
    parser.add_argument("--log-file", type=str, default="log.txt",
                        help="store the log of training process")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
