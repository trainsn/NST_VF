from torch.autograd import Variable

def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNt mean pixel-wise from a gray image"""
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, :, :, :] = 0.114*103.939 + 0.587*116.779 + 0.299*123.680
    return batch - Variable(mean)