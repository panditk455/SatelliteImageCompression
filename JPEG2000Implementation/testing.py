# Poorly constructed testing file don't look

import transforms, quantization, partition, entropy_parallel, baseline
from skimage import io, color, util, metrics
import numpy as np
import time

FILE = 'images/deep-blue-cubism.tif'
# FILE = 'images/airplane.bmp'

def main():
    quality = 100
    block_size = 64

    img = io.imread(FILE)

    start = time.time()

    # io.imsave('original.png', img)

    colors = baseline.getYCbCrArrays(FILE)
    dwt_coeffs = baseline.DWTAll(colors)
    quantized = quantization.quantize_all(dwt_coeffs, quality)
    part = partition.partition_all(quantized, block_size)
    entropy_parallel.entropy_encode_all(part, 'test.bin')



    de_entro = entropy_parallel.entropy_decode_all('test.bin')
    de_part = partition.reverse_partition(de_entro)
    de_quant = quantization.dequantize_all(de_part, quality)
    idwt = baseline.DecodeAll(de_quant)
    recon = baseline.reconstructRGB(idwt)
    # print(metrics.peak_signal_noise_ratio(img, recon))

    # quantized2 = quantization.quantize_all(dwt_coeffs, 0)
    # dequantized2 = quantization.dequantize_all(quantized2, 0)
    # idwt2 = baseline.DecodeAll(dequantized2)
    # recon2 = baseline.reconstructRGB(idwt2)

    # print(metrics.peak_signal_noise_ratio(img, recon2))
    
    io.imsave('recon.png', recon)
    

    # print(len(part))
    print(f"Done in {time.time() - start:.3f}s")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
