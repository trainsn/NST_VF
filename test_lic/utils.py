from PIL import Image

def clamp_batch(batch, low, high):
    batch[:,:].data.clamp_(low, high)

def tensor_save_gray_image(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)