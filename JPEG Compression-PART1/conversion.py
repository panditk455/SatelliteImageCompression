# Author: Kritika Pandit
# Dates: Sunday Oct 5 - Oct 10, 2025

from PIL import Image, features
import os
# import glymur
import sys
import math
import argparse
import numpy as np
import imagecodecs


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


def convert_to_jpeg(input_path: str, output_dir: str, quality: int = 10) -> str:
  
  
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{name}.jpeg")

    with Image.open(input_path) as im:
        im = ensure_rgb(im)
        im.save(out_path, "JPEG", quality = quality, optimize = True)
    return out_path


#       this part did not work as i kept getting bug for it:
# def convert_to_jpeg2000(input_path, output_dir, quality=20):
#     os.makedirs(output_dir, exist_ok = True)
#     name = os.path.splitext(os.path.basename(input_path))[0]
#     out_path = os.path.join(output_dir, f"{name}.jp2")

#     with Image.open(input_path) as im:
#         arr = np.asarray(im.convert("RGB"), dtype = np.uint8)
        
#     # as we are only interested in the Lossy algorithms:
#     # imagecodecs: set a lossy compression ratio (>1) and the lossy 9/7 wavelet
#     cratio = max(2.0, 50.0 - 0.4 * quality)  # higher = smaller file, lower quality
#     jp2_bytes = imagecodecs.jpeg2k_encode(arr, cratio = cratio, irreversible = True)

#     with open(out_path, "wb") as f:
#         f.write(jp2_bytes)
        
#     return out_path


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

    # MSE / PSNR metrics
    
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

    return { "file": os.path.basename(original_path), "compressed_file": os.path.basename(compressed_path), "original_bits": original_bits,
        "compressed_bits": compressed_bits, "compression_ratio": ratio, "space_savings": savings, "mse": mse_val,  "psnr": psnr_db, }


def print_table(rows):
    if not rows:
        print("No files analyzed.")
        return

    headers = ["file", "compressed_file", "original_bits", "compressed_bits", "compression_ratio", "space_savings", "mse", "psnr"]

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
        
        jpeg_path = convert_to_jpeg(inp, args.outdir, args.quality)
        results.append(analyze_pair(inp, jpeg_path))

        # jp2_path = convert_to_jpeg2000(inp, args.outdir, args.quality)
        # results.append(analyze_pair(inp, jp2_path))

    print_table(results)


if __name__ == "__main__":
    main()

# type for the purpose of testing:
    
# python3 conversion.py images/airplane.bmp images/inputfile5.tif images/boats.bmp images/goldhill.bmp \
# >   --outdir output_folder --quality 20