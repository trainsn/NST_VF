from torch.utils.data import Dataset

from PIL import Image

import os
import h5py

def img_loader(path):
    img = Image.open(path).convert('L')
    return img

def hdf5_loader(path):
    f = h5py.File(path, 'r')
    a_group_key = list(f.keys())[0]
    data = f[a_group_key][:]
    return data

class VFDataset(Dataset):
    def __init__(self, root, transfrom=None, target_transform=None,
                 loader=img_loader, target_loader=hdf5_loader):
        txt = 'img_names.txt'
        fh = open(os.path.join(root, txt), 'r')
        imgs = []
        for line in fh:
            imgs.append(line)

        self.root = root
        self.imgs = imgs
        self.transform = transfrom
        self.target_transform = target_transform
        self.loader = loader
        self.target_loader = target_loader

    def __getitem__(self, index):
        inputDir = os.path.join(self.root, "train_gray")
        vfDir = os.path.join(self.root, "vector_fields")

        fn = self.imgs[index]
        vf_fn = fn[:fn.rfind('.')] + '.h5'
        img = self.loader(os.path.join(inputDir, fn))
        vector_field = self.target_loader(os.path.join(vfDir, vf_fn))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            vector_field = self.target_transform(vector_field)

        return img, vector_field

    def __len__(self):
        return len(self.imgs)





