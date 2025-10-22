import transforms, baseline
from skimage import io, color
from scipy.signal import convolve
import numpy as np

file = 'images/deep-blue-cubism.tif'

img = io.imread(file)
y, cb, cr = transforms.rgb_to_ycc(img)
coeff_list = [transforms.DWT2D(y), transforms.DWT2D(cb), transforms.DWT2D(cr)]