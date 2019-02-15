import pyximport
pyximport.install()

import h5py
import numpy as np
import pdb
import os
import lic_internal
import scipy.misc

def lic(vectors, output_name):
    rsize, csize, _ = vectors.shape
    eps = 1e-7
    for x in range(rsize):
        for y in range(csize):
            if vectors[x, y, 0] == 0:
                vectors[x, y, 0] = eps
            if vectors[x, y, 1] == 0:
                vectors[x, y, 1] = eps

    kernellen=20
    kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
    kernel = kernel.astype(np.float32)
    texture = np.random.rand(rsize, csize).astype(np.float32)

    image = lic_internal.line_integral_convolution(vectors, texture, kernel)
    scipy.misc.imsave(output_name, image)

filename = 'vectorline.h5'
f = h5py.File(os.path.join(filename), 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
data = np.asarray(data)
lic(data, "test_output.jpg")
pdb.set_trace()