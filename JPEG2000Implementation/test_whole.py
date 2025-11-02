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
import matplotlib.pyplot as plt


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

def convert_to_jpeg2000(input_path: str, output_dir: str, quality: int = 75, block_size: int = 64) -> str:
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_path))[0]
    bin_path = os.path.join(output_dir, f"{name}.bin")
    image_path = os.path.join(output_dir, f"{name}.png")

    start = time.time()

    # io.imsave('original.png', img)

    colors = baseline.getYCbCrArrays(input_path)
    dwt_coeffs = baseline.DWTAll(colors)
    quantized = quantization.quantize_all(dwt_coeffs, quality)
    part = partition.partition_all(quantized, block_size)
    entropy_parallel.entropy_encode_all(part, bin_path)

    de_entro = entropy_parallel.entropy_decode_all(bin_path)
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
    
    io.imsave(image_path, recon)
    

    # print(len(part))
    runtime = time.time() - start
    print(f"{name} (block {block_size}): Done in {runtime:.3f}s")

    return bin_path, image_path, runtime


# Metrics: 
def analyze_pair(original_path: str, bin_path: str, compressed_path: str, block_size: int, runtime, quality) -> dict:

    original_bytes = os.path.getsize(original_path)
    compressed_bytes = os.path.getsize(bin_path)
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

    return {"block_size": block_size, "quality": quality, "file": os.path.basename(original_path), "compressed_file": os.path.basename(compressed_path), "original_bits": original_bits,
        "compressed_bits": compressed_bits, "compression_ratio": ratio, "space_savings": savings, "runtime": runtime, "mse": mse_val,  "psnr": psnr_db, "ssim": ssim_val,}


def print_table(rows):
    if not rows:
        print("No files analyzed.")
        return

    headers = ["block_size", "quality", "file", "compressed_file", "original_bits", "compressed_bits", "compression_ratio", "space_savings", "runtime", "mse", "psnr", "ssim"]

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


def plot_results(results, outdir):
    import matplotlib.pyplot as plt
    import numpy as np

    # Group results by image filename
    grouped = {}
    for r in results:
        fname = r["file"]
        grouped.setdefault(fname, []).append(r)

    # Sort each group by block size
    for fname in grouped:
        grouped[fname].sort(key=lambda x: x["block_size"])

    # Use modern Matplotlib colormap API (no deprecation warning)
    cmap = plt.colormaps.get_cmap("tab10")
    n_colors = len(grouped)

    # ---- Compression Ratio Plot ----
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        color = cmap(i / max(1, n_colors - 1))  # evenly spaced colors
        plt.plot(
            [v["block_size"] for v in vals],
            [v["compression_ratio"] for v in vals],
            marker="o",
            linewidth=2,
            markersize=6,
            color=color,
            label=fname,
        )

    plt.xlabel("Block Size", fontsize=12)
    plt.ylabel("Compression Ratio (compressed/original)", fontsize=12)
    plt.title("Compression Ratio vs Block Size", fontsize=14, fontweight="bold")
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compression_ratio_vs_blocksize.png"), dpi=300)
    plt.close()

    # ---- Runtime Plot ----
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        color = cmap(i / max(1, n_colors - 1))
        plt.plot(
            [v["block_size"] for v in vals],
            [v["runtime"] for v in vals],
            marker="s",
            linewidth=2,
            markersize=6,
            color=color,
            label=fname,
        )

    plt.xlabel("Block Size", fontsize=12)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.title("Runtime vs Block Size", fontsize=14, fontweight="bold")
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "runtime_vs_blocksize.png"), dpi=300)
    plt.close()

    print(f"Saved plots to {outdir}")

def plot_results_quality(results, outdir):
    import os
    import matplotlib.pyplot as plt

    # Group by image filename
    grouped = {}
    for r in results:
        fname = r["file"]
        grouped.setdefault(fname, []).append(r)

    # Sort each group by quality for nice lines
    for fname in grouped:
        grouped[fname].sort(key=lambda x: x["quality"])

    # Colormap with distinct colors
    cmap = plt.colormaps.get_cmap("tab10")
    n_colors = max(1, len(grouped))

    # ---------- Runtime vs Quality ----------
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        color = cmap(i / max(1, n_colors - 1))
        qualities = [v["quality"] for v in vals]
        runtimes  = [v["runtime"] for v in vals]
        plt.plot(qualities, runtimes, marker="s", linewidth=2, markersize=6, color=color, label=fname)

    plt.xlabel("Quality", fontsize=12)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.title("Runtime vs Quality (fixed block size)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "runtime_vs_quality.png"), dpi=300)
    plt.close()

    # ---------- Compression Ratio vs Quality ----------
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        color = cmap(i / max(1, n_colors - 1))
        qualities = [v["quality"] for v in vals]
        ratios    = [v["compression_ratio"] for v in vals]
        plt.plot(qualities, ratios, marker="o", linewidth=2, markersize=6, color=color, label=fname)

    plt.xlabel("Quality", fontsize=12)
    plt.ylabel("Compression Ratio (compressed/original)", fontsize=12)
    plt.title("Compression Ratio vs Quality (fixed block size)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compression_ratio_vs_quality.png"), dpi=300)
    plt.close()

    # ---------- PSNR vs Quality ----------
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        color = cmap(i / max(1, n_colors - 1))
        qualities = [v["quality"] for v in vals]
        psnrs     = [v["psnr"] for v in vals]
        plt.plot(qualities, psnrs, marker="^", linewidth=2, markersize=6, color=color, label=fname)

    plt.xlabel("Quality", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title("PSNR vs Quality (fixed block size)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "psnr_vs_quality.png"), dpi=300)
    plt.close()

    print(f"Saved quality plots to {outdir}")


def main():
    parser = argparse.ArgumentParser(description = "Convert other form of images to JPEG and JPEG2000.")
    parser.add_argument("inputs", nargs = "+")
    parser.add_argument("--outdir", required = True)
    parser.add_argument("--block_sizes", nargs = "+", type = int, default = [64])
    parser.add_argument("--qualities", nargs = "+", type = int, default = [75])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok = True)
    results = []

    for inp in args.inputs:
        for block_size in args.block_sizes:
            for quality in args.qualities:
                bin_out_path, image_out_path, runtime = convert_to_jpeg2000(inp, args.outdir, quality, block_size)
                results.append(analyze_pair(inp, bin_out_path, image_out_path, block_size, runtime, quality))

    print_table(results)
    # plot_results(results, args.outdir)
    plot_results_quality(results, args.outdir)


if __name__ == "__main__":
    main()

# type for the purpose of testing:
    
# python3 test_whole.py images/airplane.bmp --outdir output_folder --block_sizes 16 32 64 128 --qualities 75 100

# python3 test_whole.py images/re-entry.tif images/fanned-out.tif images/irritated.tif images/desert-ribbons.tif images/deep-blue-cubism.tif\ --outdir output_folder \ --block_sizes 16 32 64 \ --qualities 20 50 80