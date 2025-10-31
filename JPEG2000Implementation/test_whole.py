from PIL import Image, features
import os
import sys
import math
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim
import transforms, quantization, partition, entropy_parallel, baseline
from skimage import io, color, util, metrics
import time


def ensure_rgb(img: Image.Image) -> Image.Image:
    # ensures image is in standard 3-channel RGB format
    return img.convert("RGB") if img.mode != "RGB" else img


def image_to_array(img: Image.Image) -> np.ndarray:
    # HxWxC uint8 describes how an image is stored as a NumPy array: 
    # H is the height (rows)
    # W is the width (columns)
    # C is the number of color channels (like 3 for RGB)
    # Each pixel channel is stored as an unsigned 8-bit integer (uint8) with values 0–255.
    # A 480×640 RGB image → shape (480, 640, 3)
    return np.asarray(img, dtype=np.uint8)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    # Mean Squared Error over all pixels and channels
    # float64 ensures precise decimal math instead of integer rounding
    
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = a - b
    return float(np.mean(diff * diff))


def psnr(mse_val: float, max_val: float = 255.0) -> float:
    
    if mse_val <= 0.0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse_val)


def bytes_to_bits(nbytes: int) -> int:
    return nbytes * 8

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(ssim(a, b, channel_axis=-1, data_range=255))

def convert_to_jpeg2000(input_path: str, output_dir: str, quality: int = 10, jpeg_size_bytes: int = 0) -> str:
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{name}.bin")

    with Image.open(input_path) as im:
        im = ensure_rgb(im)
        arr = np.array(im)

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
    

    return out_path


# Metrics: 
def analyze_pair(original_path: str, compressed_path: str, block_size: int) -> dict:

    original_bytes = os.path.getsize(original_path)
    compressed_bytes = os.path.getsize(compressed_path)
    original_bits = bytes_to_bits(original_bytes)
    compressed_bits = bytes_to_bits(compressed_bytes)

    ratio = (compressed_bits / original_bits) if original_bits else 0.0
    
    if original_bits:
        savings = 1.0 - ratio 
    else:
        0.0

    # MSE / PSNR / SSIM metrics
    
    with Image.open(original_path) as im_orig:
        im_orig = ensure_rgb(im_orig)
        arr_orig = image_to_array(im_orig)

    with Image.open(compressed_path) as im_comp:
        im_comp = ensure_rgb(im_comp)
        
        if im_comp.size != im_orig.size:
            im_comp = im_comp.resize(im_orig.size, Image.Resampling.LANCZOS)
        arr_comp = image_to_array(im_comp)

    mse_val = mse(arr_orig, arr_comp)
    psnr_db = psnr(mse_val)
    ssim_val = compute_ssim(arr_orig, arr_comp)

    return {"block_size": block_size, "file": os.path.basename(original_path), "compressed_file": os.path.basename(compressed_path), "original_bits": original_bits,
        "compressed_bits": compressed_bits, "compression_ratio": ratio, "space_savings": savings, "mse": mse_val,  "psnr": psnr_db, "ssim": ssim_val,}


def print_table(rows):
    if not rows:
        print("No files analyzed.")
        return

    headers = ["block_size", "file", "compressed_file", "original_bits", "compressed_bits", "compression_ratio", "space_savings", "mse", "psnr", "ssim"]

# 4 decimal places for now:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))

# pretty table formatting:
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(fmt(r.get(h))))

    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))

    for r in rows:
        print(" | ".join(fmt(r.get(h)).ljust(widths[h]) for h in headers))



def main():
    parser = argparse.ArgumentParser(description = "Convert other form of images to JPEG and JPEG2000.")
    parser.add_argument("inputs", nargs = "+")
    parser.add_argument("--outdir", required = True)
    parser.add_argument("--block_size", type = int, default = 64)
    parser.add_argument("--quality", type = int, default = 75)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok = True)
    results = []

    for inp in args.inputs:
        jp2_path = convert_to_jpeg2000(inp, args.outdir, args.quality, args.block_size)
        results.append(analyze_pair(inp, jp2_path, args.block_size))

    print_table(results)


if __name__ == "__main__":
    main()

# type for the purpose of testing:

# python3 test_whole.py images/re-entry.tif images/fanned-out.tif images/irritated.tif images/desert-ribbons.tif images/deep-blue-cubism.tif\ --outdir output_folder \ --block_size 64 \ --quality 20 