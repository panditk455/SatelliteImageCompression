# Author: Leon Liang
# Entropy coding and saving output as binary file in JPEG2000 pipeline

from typing import List, Dict, Tuple, Union
import pickle
import numpy as np
import zlib
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import BaseFrequencyTable

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
