import h5py
import numpy as np
import pdb
filename = 'h5ex_d_rdwr.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
data = np.asarray(data)
pdb.set_trace()