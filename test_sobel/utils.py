import torch
from torch.autograd import Variable

from PIL import Image
import pdb

def tensor_load_vector_field(vectors):
    vectors = vectors.transpose(2, 0, 1)
    vectors = torch.from_numpy(vectors).float()
    vectors = vectors.unsqueeze(0)
    vectors = Variable(vectors, requires_grad=False)
    return vectors

def clamp_batch(batch, low, high):
    batch[:,:,:,:].data.clamp_(low, high)

def tensor_save_gray_image(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img[0].astype('uint8')
    pdb.set_trace()
    img = Image.fromarray(img)
    img.save(filename)