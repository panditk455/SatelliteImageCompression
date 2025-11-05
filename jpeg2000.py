import skimage as ski
import pywt
import numpy as np
from typing import Dict, List, Tuple, Union
import zlib
from concurrent.futures import ProcessPoolExecutor
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import BaseFrequencyTable
import pickle
import argparse

def color_convert(file):
    # Read image and convert to YCbCr colorspace
    img = ski.io.imread(file)
    ycc = ski.color.rgb2ycbcr(img)

    # Create 2D arrays for Y, Cb, Cr coeffs
    y, cb, cr = ycc[:,:,0], ycc[:,:,1], ycc[:,:,2]

    # Return as list of colorspace coefficients
    return [y, cb, cr]

def dwt_all(list, wavelet, n=3):
    return [dwt(array, wavelet, n) for array in list]

def dwt(array, wavelet, n=3):
    coeff_dict = {"levels": []}

    # Run 2D CDF wavelet transform
    coeffs = pywt.wavedec2(array, wavelet, level=n)
    coeff_dict["LL"] = coeffs[0]

    for i in range(n):
        tuple = coeffs[i+1]
        coeff_dict["levels"].append({"LH": tuple[0], "HL": tuple[1], "HH": tuple[2]})

    return coeff_dict

# a: minimum, b: maximum, t: quality
def linear_interpolation(a, b, t):
    return a + (b - a) * t / 100

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

def encode_block(block: Dict) -> BlockTuple:
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

    if K <= 1:
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

def entropy_encode_all(blocks: List[Dict], output_path: str):
    with ProcessPoolExecutor() as executor:
        encoded = list(executor.map(encode_block, blocks, chunksize=64))
        
    # encoded = list(map(encode_block, blocks))
    with open(output_path, 'wb') as f:
        pickle.dump(encoded, f, protocol=pickle.HIGHEST_PROTOCOL)

def entropy_decode_all(input_path: str) -> List[Dict]:
    with open(input_path, 'rb') as f:
        encoded = pickle.load(f)
    with ProcessPoolExecutor() as executor:
        decoded = list(executor.map(decode_block, encoded, chunksize=64))
    # decoded = list(map(decode_block, encoded))
    return decoded

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

def idwt_all(list, wavelet):
    return [idwt(coeffs, wavelet) for coeffs in list]

def idwt(dict, wavelet):
    n = len(dict["levels"])

    coeff_list = [dict["LL"]]
    for i in range(n):
        level = dict["levels"][i]
        coeff_list.append((level["LH"], level["HL"], level["HH"]))

    coeffs = pywt.waverec2(coeff_list, wavelet)

    return coeffs

def reconstructRGB(arrays):
    image = np.stack(arrays, axis=2)
    rgb_img = ski.color.ycbcr2rgb(image)
    return (rgb_img*255).astype(np.uint8)

def full_encoding(input, output, wavelet="haar", quality=100, block_size=64, n=3):
    color = color_convert(input)
    dwt = dwt_all(color, wavelet, n)
    quant = quantize_all(dwt, quality)
    part = partition_all(quant, block_size)
    entropy_encode_all(part, output)

def full_decoding(input, output, wavelet="haar", quality=100):
    decode = entropy_decode_all(input)
    depart = reverse_partition(decode)
    dequant = dequantize_all(depart, quality)
    idwt = idwt_all(dequant, wavelet)
    original = reconstructRGB(idwt)
    ski.io.imsave(output, original)

def main():
    parser = argparse.ArgumentParser(description="JPEG 2000 Pipeline")

    parser.add_argument("input", help="Input image (any format)")
    parser.add_argument("--wavelet", default="haar", help="Wavelet for DWT")
    parser.add_argument("--quality", type=int, default=100, help="Quality (0-100)")
    parser.add_argument("--block_size", type=int, default=64, help="Partitioning block size")
    parser.add_argument("--output", default="output.bin", help="Output filename")
    parser.add_argument("--recon", default="recon.png", help="Reconstructed image filename")
    args = parser.parse_args()

    print("Beginning encoding process...")
    full_encoding(args.input, args.output, args.wavelet, args.quality, args.block_size)
    print(f"Output has been saved to {args.output}.\nBeginning decoding process...")
    full_decoding(args.output, args.recon, args.wavelet, args.quality)
    print(f"Decoded image has been saved to {args.recon}.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()