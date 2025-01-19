import os
import imageio

p = '/home/jicredt_data/dsj/CDdata/BANDON/test/labels_unch0ch1ig255'

for file in os.listdir(p):
    print(file)
    img = imageio.imread(os.path.join(p, file))
    print(set(img.flatten()))