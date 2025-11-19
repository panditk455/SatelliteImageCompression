# JPEG2000 Full Pipeline and Analysis
# Authors: Leon Liang and Lucas Schattenmann
# Date: November 19, 2025

from PIL import Image, features
import os
import sys
import math
import argparse
import numpy as np
from typing import Dict, List, Tuple, Union
from skimage.metrics import structural_similarity as ssim
import pywt
from skimage import color
import time
import pickle
import zlib
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import BaseFrequencyTable

# =============================  IMAGE PRE/POST-PROCESSING  ============================= #

def preprocess_image(file):
    # Read image and convert to YCbCr colorspace
    img = Image.open(file)
    img = image_to_array(ensure_rgb(img))
    ycc = color.rgb2ycbcr(img)

    # Create 2D arrays for Y, Cb, Cr coeffs
    y, cb, cr = ycc[:,:,0], ycc[:,:,1], ycc[:,:,2]

    # Return as list of colorspace coefficients
    return [y, cb, cr]

def postprocess_image(arrays):
    image = np.stack(arrays, axis=2)
    rgb_img = color.ycbcr2rgb(image)
    return (rgb_img*255).astype(np.uint8)

# =============================  END OF IMAGE PRE/POST-PROCESSING ============================= #

# =============================  DISCRETE WAVELET TRANSFORM ============================= #

def dwt_all(list, n=3):
    return [dwt(array, n) for array in list]

def dwt(array, n=3):
    coeff_dict = {"levels": []}

    # Run 2D Haar wavelet transform
    coeffs = pywt.wavedec2(array, "haar", level=n)
    coeff_dict["LL"] = coeffs[0]

    # Put in compatible format for quantization
    for i in range(n):
        tuple = coeffs[i+1]
        coeff_dict["levels"].append({"LH": tuple[0], "HL": tuple[1], "HH": tuple[2]})

    return coeff_dict

def idwt_all(coeff_list):
    return [idwt(coeffs) for coeffs in coeff_list]

def idwt(coeff_dict):
    # Find level of deconstruction
    n = len(coeff_dict["levels"])

    # Ensure compatible format
    coeff_list = [coeff_dict["LL"]]
    for i in range(n):
        level = coeff_dict["levels"][i]
        coeff_list.append((level["LH"], level["HL"], level["HH"]))

    # Run inverse DWT
    coeffs = pywt.waverec2(coeff_list, "haar")

    return coeffs

# =============================  END OF DISCRETE WAVELET TRANSFORM ============================= #

# =============================  QUANTIZATION & DEQUANTIZATION ============================= #

# Helper function: Linear Interpolation
# a: minimum, b: maximum, t: quality
def linear_interpolation(a, b, t):
    return a + (b - a) * t / 100

# Calculating the step sizes and deadzones based on the number of DWT level and user defined quality factor
def calc_params(num_levels: int, quality: float) -> Tuple[Dict[Tuple[int, str], float], float, Dict[str, float]]:
    """
    Return:
      deltas: {(level, band): delta} for bands in {"LH","HL","HH"}
      delta_LL: delta for LL
      deadzones: {"LL": dz_LL, "other": dz_ow}
    quality in [0,100]: 100 = best quality, 0 = lowest quality
    """
    quality = float(np.clip(quality, 0.0, 100.0))

    # Delta
    global_delta = linear_interpolation(1.8, 0.6, quality)   

    base_delta = 1.0 * global_delta
    delta_LL = 0.7 * global_delta

    first_delta = linear_interpolation(1.8, 1.2, quality)   # low Q -> stronger growth, high Q -> gentler

    deltas: Dict[Tuple[int, str], float] = {}
    for level in range(1, num_levels + 1):
        # level 1 (finest) should get biggest factor
        t = (num_levels - level) / max(1, (num_levels - 1)) * 100 # t = 0 at finest, 100 at coarsest
        level_factor = linear_interpolation(first_delta, 1.0, t)

        for b in ("LH", "HL", "HH"):
            deltas[(level, b)] = base_delta * level_factor
            if b == "HH":
                deltas[(level, b)] = deltas[(level, b)] * 1.3

    # Deadzone
    dz_ow = linear_interpolation(1.8, 1.2, quality)  # low quality: larger deadzone; high quality: smaller
    dz_LL = linear_interpolation(1.3, 1.0, quality)  

    deadzones = {"LL": float(dz_LL), "other": float(dz_ow)}
    return deltas, float(delta_LL), deadzones

# Quantizing a coefficient array (x)
def quantize(x: np.ndarray, delta: float, deadzone: float = 1.5) -> np.ndarray:
    if delta <= 0:
        raise ValueError("delta must be > 0")
    a = np.abs(x)
    q = np.where(a >= deadzone * delta,
                 np.floor((a - (deadzone - 1) * delta) / delta),
                 0.0)
    return (np.sign(x) * q).astype(np.int32)

# Quantizing single component
def quantize_component(comp: Dict, quality: float) -> Dict:
    num_levels = len(comp["levels"])
    deltas, delta_LL, deadzones = calc_params(num_levels = num_levels, quality = quality)

    dz_ow = deadzones.get("other", deadzones.get("other", 1.5))
    dz_LL = deadzones.get("LL", dz_ow)

    out_levels = []
    for level, bands in enumerate(comp["levels"], start = 1):
        out_levels.append({
            "LH": quantize(bands["LH"], deltas.get((level, "LH"), 1.0), dz_ow),
            "HL": quantize(bands["HL"], deltas.get((level, "HL"), 1.0), dz_ow),
            "HH": quantize(bands["HH"], deltas.get((level, "HH"), 1.0), dz_ow),
        })
    out_LL = quantize(comp["LL"], delta_LL, dz_LL)
    return {"levels": out_levels, "LL": out_LL}


# Quantizing all coefficients
def quantize_all(dwt_result: List[Dict], quality: float) -> List[Dict]:
    return [quantize_component(c, quality) for c in dwt_result]

# Dequangtizing a coefficient array (x)
def dequantize(q: np.ndarray, delta: float, deadzone: float = 1.5, out_dtype=np.float32) -> np.ndarray:
    if delta <= 0:
        raise ValueError("delta must be > 0")
    q = np.asarray(q)
    s = np.sign(q).astype(np.float32)
    a = np.abs(q).astype(np.float32)

    # midpoint reconstruction for nonzero bins; zero stays zero
    mag = np.where(a > 0.0, (deadzone - 0.5 + a) * delta, 0.0).astype(np.float32)
    xhat = (s * mag).astype(out_dtype)
    return xhat

# Dequantizing single component
def dequantize_component(comp_q: Dict, quality: float) -> Dict:
    num_levels = len(comp_q["levels"])
    deltas, delta_LL, deadzones = calc_params(num_levels=num_levels, quality=quality)

    dz_ow = deadzones.get("other", 1.5)
    dz_LL = deadzones.get("LL", dz_ow)

    out_levels = []
    for level, bands in enumerate(comp_q["levels"], start=1):
        out_levels.append({
            "LH": dequantize(bands["LH"], deltas.get((level, "LH"), 1.0), dz_ow),
            "HL": dequantize(bands["HL"], deltas.get((level, "HL"), 1.0), dz_ow),
            "HH": dequantize(bands["HH"], deltas.get((level, "HH"), 1.0), dz_ow),
        })
    out_LL = dequantize(comp_q["LL"], delta_LL, dz_LL)
    return {"levels": out_levels, "LL": out_LL}

# Dequantizing all components
def dequantize_all(result: List[Dict], quality: float) -> List[Dict]:
    return [dequantize_component(c, quality) for c in result]

# =============================  END OF QUANTIZATION & DEQUANTIZATION  ============================= #

# ============================= PARTITIONING & REVERSE PARTITIONING ============================= #

# Splits a 2D array into blocks of user-defined size (64x64 by default)
# Returns a list of tuples ((i, j), block_array), where (i,j) is the left top coordinate of the block
def partition(array: np.ndarray, block_size: int = 64) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    h, w = array.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = array[i:i+block_size, j:j+block_size]
            blocks.append(((i, j), block))
    return blocks

# Partition all component results from the quantization part into square blocks
# Return format: {
#                   'component': int,             (Y:0, Cb:1, Cr:2)
#                   'level': int,                 (DWT level)
#                   'band': str,                  (LL, HL, LH, or HH)
#                   'position': (i, j),           (The left top coordinate of the block)
#                   'shape': (h, w),              (Size of the block)
#                   'data': np.ndarray            (Actual data)
#               }
def partition_all(quantized_result: List[Dict], block_size: int = 64) -> List[Dict]:
    all_blocks = []
    for comp_idx, comp in enumerate(quantized_result):
        # LL
        for (i, j), block in partition(comp["LL"], block_size):
            all_blocks.append({
                'component': comp_idx,
                'level': 0,
                'band': 'LL',
                'position': (i, j),
                'shape': block.shape,
                'data': block
            })

        # HL, LH, and HH
        for level_idx, level_bands in enumerate(comp["levels"], start=1):
            for band in ("LH", "HL", "HH"):
                for (i, j), block in partition(level_bands[band], block_size):
                    all_blocks.append({
                        'component': comp_idx,
                        'level': level_idx,
                        'band': band,
                        'position': (i, j),
                        'shape': block.shape,
                        'data': block
                    })

    return all_blocks

# Reconstructs the original quantized_result structure from a list of blocks.
def reverse_partition(blocks: List[Dict]) -> List[Dict]:
    # Determine how many components exist
    num_components = 3  # Y, Cb, Cr
    result = [{"LL": None, "levels": []} for i in range(num_components)]

    for comp_idx in range(num_components):
        # Group all blocks in each component
        comp_blocks = [b for b in blocks if b["component"] == comp_idx]
        if not comp_blocks:
            continue

        # Rebuild LL 
        ll_blocks = [b for b in comp_blocks if b["band"] == "LL"]
        if ll_blocks:
            h_max = max(b["position"][0] + b["shape"][0] for b in ll_blocks)
            w_max = max(b["position"][1] + b["shape"][1] for b in ll_blocks)
            LL = np.zeros((h_max, w_max), dtype=ll_blocks[0]["data"].dtype)
            for b in ll_blocks:
                i, j = b["position"]
                h, w = b["shape"]
                LL[i:i+h, j:j+w] = b["data"]
            result[comp_idx]["LL"] = LL

        # Rebuild HL, LH, HH bands for each level
        max_level = max((b["level"] for b in comp_blocks), default=0)
        result[comp_idx]["levels"] = [
            {"LH": None, "HL": None, "HH": None} for i in range(max_level)
        ]

        for level in range(1, max_level + 1):
            for band in ("LH", "HL", "HH"):
                band_blocks = [b for b in comp_blocks
                               if b["band"] == band and b["level"] == level]
                if not band_blocks:
                    continue

                h_max = max(b["position"][0] + b["shape"][0] for b in band_blocks)
                w_max = max(b["position"][1] + b["shape"][1] for b in band_blocks)
                band_array = np.zeros((h_max, w_max), dtype=band_blocks[0]["data"].dtype)
                for b in band_blocks:
                    i, j = b["position"]
                    h, w = b["shape"]
                    band_array[i:i+h, j:j+w] = b["data"]

                result[comp_idx]["levels"][level - 1][band] = band_array

    return result

# ============================= ENF OF PARTITIONING & REVERSE PARTITIONING ============================= #

# ============================= ENTROPYCODING ENCODING & DECODING ============================= #


# Zigzag encode: z = 2|x| - 1 if x < 0 else 2x
def zigzag_encode_arr(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.int32, copy=False)
    return np.where(a >= 0, 2 * a, -2 * a - 1).astype(np.int64, copy=False)

# Zigzag decode: x = z//2 if even else -(z//2) - 1
def zigzag_decode_arr(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.int64, copy=False)
    return np.where((z & 1) == 0, z // 2, -(z // 2) - 1).astype(np.int32, copy=False)

def normalize_band(band_val):
    if isinstance(band_val, bytes):
        return band_val.decode(errors="ignore")
    if isinstance(band_val, str):
        return band_val
    return int(band_val)

def make_model_dict(alphabet_size: int) -> BaseFrequencyTable:
    A = max(2, int(alphabet_size))
    counts = {i: 1 for i in range(A)}
    return BaseFrequencyTable(counts)

# Tuple formats
# AC block: ('ac', comp, level, band, pos_y, pos_x, h, w, #unique_vals, encoded_bytes, unique_vals)
# RAW block: ('raw', comp, level, band, pos_y, pos_x, h, w, raw_zlib_bytes). 
# Created for uniform blocks: blocks with only one unique coefficient. Therefore cannot apply arithmetic encoder to it.
ACBlock = Tuple[str, int, int, Union[str, int], int, int, int, int, int, bytes, int, List[int]]
RAWBlock = Tuple[str, int, int, Union[str, int], int, int, int, int, bytes]
BlockTuple = Union[ACBlock, RAWBlock]

# Encoding one block and return a tuple with encoded data and side information needed for decoding
def encode_block(block: Dict, pivot: int) -> BlockTuple:
    comp = int(block['component'])
    lvl = int(block['level'])
    bnd = normalize_band(block['band'])
    posy, posx = map(int, block['position'])
    h, w = map(int, block['shape'])
    data = block['data']

    flat = data.reshape(-1)
    zz = zigzag_encode_arr(flat)
    n = zz.size

    if n == 0:
        raw_bytes = zlib.compress(np.array([], dtype=np.int32).tobytes(), level=6)
        return ('raw', comp, lvl, bnd, posy, posx, h, w, raw_bytes)

    unique_vals_np = np.unique(zz)
    unique_vals = unique_vals_np.astype(int).tolist()
    K = len(unique_vals)

    if K <= pivot: # pivot shoul dbe an integer larger or equal to 1
    # store the real data; zlib compresses uniform blocks extremely well
        raw_bytes = zlib.compress(data.astype(np.int32, copy=False).tobytes(), level=6)
        return ('raw', comp, lvl, bnd, posy, posx, h, w, raw_bytes)

    val2idx = {v: i for i, v in enumerate(unique_vals)}
    idx_list = [val2idx[int(v)] for v in zz.tolist()]  # 0-based
    A = K

    model = make_model_dict(A)
    coder = AECompressor(model)

    try:
        bits_list = coder.compress(idx_list)
        bit_count = len(bits_list)

        if bit_count > 0:
            bits = np.fromiter(bits_list, dtype=np.uint8, count=bit_count)
            enc_bytes = np.packbits(bits, bitorder="little").tobytes()
        else:
            enc_bytes = b""

        return ('ac', comp, lvl, bnd, posy, posx, h, w, K, enc_bytes, bit_count, unique_vals)

    except Exception:
        raw_bytes = zlib.compress(data.astype(np.int32, copy=False).tobytes(), level=6)
        return ('raw', comp, lvl, bnd, posy, posx, h, w, raw_bytes)

# Decoding one block
def decode_block(tup: BlockTuple) -> Dict:
    kind = tup[0]

    if kind == 'raw':
        _, comp, lvl, bnd, posy, posx, h, w, raw_bytes = tup
        buf = zlib.decompress(raw_bytes)

        # Make decoding deterministic
        if buf:
            arr = np.frombuffer(buf, dtype=np.int32)
        else:
            # Only allow truly empty blocks (h*w == 0); otherwise fill zeros
            arr = np.zeros((h * w,), dtype=np.int32)

        data2d = arr.reshape((h, w))

        return {
            'component': comp, 'level': lvl, 'band': bnd,
            'position': (posy, posx), 'shape': [h, w], 'data': data2d,
        }

    _, comp, lvl, bnd, posy, posx, h, w, K, payload, bit_count, unique_vals = tup
    n = h * w

    A = K
    model = make_model_dict(A)
    coder = AECompressor(model)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little") if payload else np.array([], np.uint8)
    bits = bits[:bit_count]  # trim to the real encoded bit length
    bits_list = bits.tolist()
    emitted_syms = coder.decompress(bits_list, n)
    idx_list = emitted_syms  # no offset now
    vals = np.array([unique_vals[i] for i in idx_list], dtype=np.int64)
    coeffs = zigzag_decode_arr(vals)
    data2d = coeffs.reshape((h, w))


    return {
        'component': comp, 'level': lvl, 'band': bnd,
        'position': (posy, posx), 'shape': [h, w], 'data': data2d,
    }

# Encoding all blocks in parallel
def entropy_encode_all(blocks: List[Dict], output_path: str, pivot: List[int]):
    with ProcessPoolExecutor() as executor:
        encoded = list(executor.map(encode_block, blocks, repeat(pivot), chunksize=64))
    with open(output_path, 'wb') as f:
        pickle.dump(encoded, f, protocol=pickle.HIGHEST_PROTOCOL)

# Decoding all blocks in parallel
def entropy_decode_all(input_path: str) -> List[Dict]:
    with open(input_path, 'rb') as f:
        encoded = pickle.load(f)
    with ProcessPoolExecutor() as executor:
        decoded = list(executor.map(decode_block, encoded, chunksize=64))
    return decoded


# ============================= END OF ENTROPYCODING & ENCODING & DECODING ============================= #


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

def convert_to_jpeg2000(input_path: str, output_dir: str, dwt_level: int = 1, quality: int = 75, block_size: int = 64, pivot: int = 1) -> str:
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_path))[0]
    bin_path = os.path.join(output_dir, f"{name}.bin")
    image_path = os.path.join(output_dir, f"{name}.png")

    start = time.time()

    colors = preprocess_image(input_path)
    dwt_coeffs = dwt_all(colors, dwt_level)
    quantized = quantize_all(dwt_coeffs, quality)
    part = partition_all(quantized, block_size)
    entropy_encode_all(part, bin_path, pivot)

    de_entro = entropy_decode_all(bin_path)
    de_part = reverse_partition(de_entro)
    de_quant = dequantize_all(de_part, quality)
    idwt = idwt_all(de_quant)
    recon = postprocess_image(idwt)
    
    # io.imsave(image_path, recon)
    runtime = time.time() - start

    print(f"{name} (dwt {dwt_level}) (block {block_size}) (quality {quality}) (pivot {pivot}): Done in {runtime:.3f}s")

    return bin_path, image_path, runtime

# Metrics: 
def analyze_pair(original_path: str, bin_path: str, compressed_path: str, dwt_level, block_size: int, runtime, quality, pivot: int) -> dict:

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

    return {"dwt": dwt_level, "block_size": block_size, "quality": quality, "pivot": pivot, "file": os.path.basename(original_path), "compressed_file": os.path.basename(compressed_path), "original_bits": original_bits,
        "compressed_bits": compressed_bits, "compression_ratio": ratio, "space_savings": savings, "runtime": runtime, "mse": mse_val,  "psnr": psnr_db, "ssim": ssim_val,}


def print_table(rows):
    if not rows:
        print("No files analyzed.")
        return

    headers = ["dwt", "block_size", "quality", "file", "compressed_file", "original_bits", "compressed_bits", "compression_ratio", "space_savings", "runtime", "mse", "psnr", "ssim"]

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

# DWT deconstruction experiments
def plot_results_dwt(results, outdir):
    import matplotlib.pyplot as plt
    import numpy as np

    # Group results by image filename
    grouped = {}
    for r in results:
        fname = r["file"]
        grouped.setdefault(fname, []).append(r)

    # Sort each group by DWT level
    for fname in grouped:
        grouped[fname].sort(key=lambda x: x["dwt"])

    # Use modern Matplotlib colormap API (no deprecation warning)
    cmap = plt.colormaps.get_cmap("tab10")
    n_colors = len(grouped)

    # Compression Ratio vs DWT
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        color = cmap(i / max(1, n_colors - 1))  # evenly spaced colors
        plt.plot(
            [v["dwt"] for v in vals],
            [v["compression_ratio"] for v in vals],
            marker=".",
            linewidth=2,
            markersize=6,
            color=color,
            label=fname,
        )

    plt.xlabel("Level of Deconstruction", fontsize=12)
    plt.ylabel("Compression Ratio (compressed/original)", fontsize=12)
    plt.title("Compression Ratio vs DWT Level", fontsize=14, fontweight="bold")
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compression_ratio_vs_dwt_level.png"), dpi=300)
    plt.close()

    # Runtime vs DWT
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        color = cmap(i / max(1, n_colors - 1))
        plt.plot(
            [v["dwt"] for v in vals],
            [v["runtime"] for v in vals],
            marker=".",
            linewidth=2,
            markersize=6,
            color=color,
            label=fname,
        )

    plt.xlabel("Level of Deconstruction", fontsize=12)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.title("Runtime vs DWT Level", fontsize=14, fontweight="bold")
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "runtime_vs_dwt_level.png"), dpi=300)
    plt.close()

    # PSNR vs DWT
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        color = cmap(i / max(1, n_colors - 1))
        qualities = [v["dwt"] for v in vals]
        psnrs     = [v["psnr"] for v in vals]
        plt.plot(qualities, psnrs, marker=".", linewidth=2, markersize=6, color=color, label=fname)

    plt.xlabel("Level of Deconstruction", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title("PSNR vs DWT Level", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "psnr_vs_dwt_level.png"), dpi=300)
    plt.close()

    print(f"Saved DWT plots to {outdir}")

# Block size experiment graph
def plot_results_blocksize(results, outdir):
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

    # Compression Ratio Plot
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

    # Runtime Plot 
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

# Quality experiment graph
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

    # Runtime vs Quality
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

    # Compression Ratio vs Quality 
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

    # PSNR vs Quality 
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

# Pivot experiment graph
def plot_results_pivot(results, outdir):
    import os
    import matplotlib.pyplot as plt

    # Group results by image filename
    grouped = {}
    for r in results:
        fname = r["file"]
        grouped.setdefault(fname, []).append(r)

    # Sort each group by pivot for consistent plotting
    for fname in grouped:
        grouped[fname].sort(key=lambda x: x["pivot"])

    cmap = plt.colormaps.get_cmap("tab10")
    n_colors = max(1, len(grouped))

    # Runtime vs Pivot
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        pivots = [v["pivot"] for v in vals]
        runtimes = [v["runtime"] for v in vals]
        color = cmap(i / max(1, n_colors - 1))
        plt.plot(pivots, runtimes, marker="s", linewidth=2, markersize=6, color=color, label=fname)

    plt.xlabel("Pivot", fontsize=12)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.title("Runtime vs Pivot", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "runtime_vs_pivot.png"), dpi=300)
    plt.close()

    # PSNR vs Pivot
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        pivots = [v["pivot"] for v in vals]
        psnrs = [v["psnr"] for v in vals]
        color = cmap(i / max(1, n_colors - 1))
        plt.plot(pivots, psnrs, marker="^", linewidth=2, markersize=6, color=color, label=fname)

    plt.xlabel("Pivot", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title("PSNR vs Pivot", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "psnr_vs_pivot.png"), dpi=300)
    plt.close()

    # Compression Ratio vs Pivot
    plt.figure(figsize=(8, 6))
    for i, (fname, vals) in enumerate(grouped.items()):
        pivots = [v["pivot"] for v in vals]
        ratios = [v["compression_ratio"] for v in vals]
        color = cmap(i / max(1, n_colors - 1))
        plt.plot(pivots, ratios, marker="o", linewidth=2, markersize=6, color=color, label=fname)

    plt.xlabel("Pivot", fontsize=12)
    plt.ylabel("Compression Ratio (compressed/original)", fontsize=12)
    plt.title("Compression Ratio vs Pivot", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Image", loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compression_ratio_vs_pivot.png"), dpi=300)
    plt.close()

    print(f"Saved pivot plots to {outdir}")

def main():
    parser = argparse.ArgumentParser(description = "Convert other form of images to JPEG and JPEG2000.")
    parser.add_argument("inputs", nargs = "+")
    parser.add_argument("--outdir", required = True)
    parser.add_argument("--dwt_levels", nargs = "+", type = int, default = [1])
    parser.add_argument("--block_sizes", nargs = "+", type = int, default = [64])
    parser.add_argument("--qualities", nargs = "+", type = int, default = [75])
    parser.add_argument("--pivots", nargs = "+", type = int, default = [1])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok = True)
    results = []

    for inp in args.inputs:
        for dwt_level in args.dwt_levels:
            for block_size in args.block_sizes:
                for quality in args.qualities:
                    for pivot in args.pivots:
                        bin_out_path, image_out_path, runtime = convert_to_jpeg2000(inp, args.outdir, dwt_level, quality, block_size, pivot)
                        results.append(analyze_pair(inp, bin_out_path, image_out_path, dwt_level, block_size, runtime, quality, pivot))

    print_table(results)
    plot_results_dwt(results, args.outdir)
    plot_results_blocksize(results, args.outdir)
    plot_results_quality(results, args.outdir)
    plot_results_pivot(results, args.outdir)

if __name__ == "__main__":
    main()