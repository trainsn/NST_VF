from os import listdir
from PIL import Image
import os

fns = [fn for fn in listdir('train_sub') if fn.endswith('.jpg')]
fns.sort()

outfile = open('img_names.txt', 'w')
for fn in fns:
    # im = Image.open(fn).convert('L').resize((512, 512), Image.BICUBIC)
    # im.save(os.path.join("train_gray", fn))

    outfile.write('%s\n' % fn)
