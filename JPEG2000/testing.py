# Poorly constructed testing file don't look

import transforms, quantization, partition, entropy_parallel
from skimage import io, color
from scipy.signal import convolve
import numpy as np
import time

file = 'images/deep-blue-cubism.tif'

img = io.imread(file)

start = time.time()
# y, cb, cr = transforms.rgb_to_ycc(img)
ycc_img = color.rgb2ycbcr(img)
y, cb, cr = ycc_img[:,:,0], ycc_img[:,:,1], ycc_img[:,:,2]

coeff_list = [transforms.DWT2D(y, 3), transforms.DWT2D(cb, 3), transforms.DWT2D(cr, 3)]

quant = quantization.quantize_all(coeff_list, 100)
part = partition.partition_all(quant)
entropy = entropy_parallel.entropy_encode_all(part, 'test.bin')

print(len(part))