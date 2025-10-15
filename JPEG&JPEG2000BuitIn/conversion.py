# Author: Kritika Pandit
# Dates: Sunday Oct 5 - Oct 10, 2025

# Author: Leon Liang
# Dates: Saturday Oct 11 - Oct 14, 2025
# Implemented JPEG 2000, compression rate control, and SSIM

from PIL import Image, features
import os
import glymur as gly
import sys
import math
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim
# import imagecodecs


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



def convert_to_jpeg(input_path: str, output_dir: str, quality: int = 10) -> str:
  
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{name}.jpeg")

    with Image.open(input_path) as im:
        im = ensure_rgb(im)
        im.save(out_path, "JPEG", quality = quality, optimize = True)

    size_bytes = os.path.getsize(out_path)
    return out_path, size_bytes


# Modified by Leon Liang
def convert_to_jpeg2000(input_path: str, output_dir: str, quality: int = 10, jpeg_size_bytes: int = 0) -> str:

    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{name}.jp2")

    with Image.open(input_path) as im:
        im = ensure_rgb(im)
        arr = np.array(im)

    uncompressed_bytes = arr.size * arr.itemsize  # size for JP2 input
    cratio = max(1.0, uncompressed_bytes / jpeg_size_bytes)

    # # JP2 compression ratio 
    # # higher ratio = more compression
    # # cratio = int(round(200 / quality))
    # # cratio = quality
    # # cratio = int(round(1000 / quality))

    # Encode to JP2 (lossy 9/7 by default via irreversible=True)
    gly.Jp2k(
        out_path,
        data=arr,
        cratios=[cratio],
        irreversible=True,   # lossy. Set False for lossless.
        prog="LRCP",         # layer-resolution-component-position progression\
        # numres=6,
        # cbsize=(64, 64),
        # tilesize=(1024, 1024),
    )

    return out_path


# Metrics: 

def analyze_pair(original_path: str, compressed_path: str) -> dict:

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

    return { "file": os.path.basename(original_path), "compressed_file": os.path.basename(compressed_path), "original_bits": original_bits,
        "compressed_bits": compressed_bits, "compression_ratio": ratio, "space_savings": savings, "mse": mse_val,  "psnr": psnr_db, "ssim": ssim_val,}


def print_table(rows):
    if not rows:
        print("No files analyzed.")
        return

    headers = ["file", "compressed_file", "original_bits", "compressed_bits", "compression_ratio", "space_savings", "mse", "psnr", "ssim"]

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
    parser.add_argument("--quality", type = int, default = 10)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok = True)
    results = []

    for inp in args.inputs:
        
        jpeg_path, jpeg_size = convert_to_jpeg(inp, args.outdir, args.quality)
        results.append(analyze_pair(inp, jpeg_path))

        jp2_path = convert_to_jpeg2000(inp, args.outdir, args.quality, jpeg_size)
        results.append(analyze_pair(inp, jp2_path))

    print_table(results)


if __name__ == "__main__":
    main()

# type for the purpose of testing:
    
# python3 conversion.py images/airplane.bmp images/inputfile5.tif images/boats.bmp images/goldhill.bmp \ --outdir output_folder \ --quality 20

# python3 conversion.py images/test1.tif images/fanned-out.tif images/irritated.tif images/desert-ribbons.tif images/deep-blue-cubism.tif\ --outdir output_folder \ --quality 20 