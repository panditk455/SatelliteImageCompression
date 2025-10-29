import baseline, quantization
from skimage import io, util, color, metrics
import time

file = 'images/deep-blue-cubism.tif'
img = io.imread(file)
# io.imsave('original.png', img)

colors = baseline.getYCbCrArrays(file)
dwt_coeffs = baseline.DWTAll(colors)
quantized = quantization.quantize_all(dwt_coeffs, 100)



dequantized = quantization.dequantize_all(quantized, 100)
idwt = baseline.DecodeAll(dequantized)
recon = baseline.reconstructRGB(idwt)
print(metrics.peak_signal_noise_ratio(img, recon))

quantized2 = quantization.quantize_all(dwt_coeffs, 0)
dequantized2 = quantization.dequantize_all(quantized2, 0)
idwt2 = baseline.DecodeAll(dequantized2)
recon2 = baseline.reconstructRGB(idwt2)

print(metrics.peak_signal_noise_ratio(img, recon2))