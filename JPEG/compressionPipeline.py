"""
JPEG Full pipeline implementation: color space conversion, chroma subsampling, DCT, quantization, and entropy coding.
Authors: Kritika Pandit, Anika Rajbhandary
"""
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import cv2

# Color & Subsampling ---------------------------------------------

def ensure_rgb(img: Image.Image) -> Image.Image:
    """
    Making sure the image is in plain RGB format (3 color channels).
    We convert any other format to the form of  'RGB'
    so the rest of the code can assume the same format.
    """
    return img.convert("RGB")

def rgb_to_ycbcr_arrays(img: Image.Image):
    """
    Convert an RGB image into the YCbCr color space and split it into 3 arrays:
      - Y  = brightness (luma) information
      - Cb = blue-color difference (chroma) information
      - Cr = red-color difference (chroma) information
    """
    
    ycbcr = img.convert("YCbCr")
    arr = np.asarray(ycbcr, dtype = np.uint8)
    Y, Cb, Cr = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return Y, Cb, Cr

def downsample_420(channel_u8: np.ndarray, method: str = "nearest") -> np.ndarray:
    """
    4:2:0 chroma downsampling: reduce width and height by 2.
    method:
      - 'nearest' : pick nearest neighbor (fast; sharp)
      - 'average' : area/box filter (smoother; slightly better quality)
      - '444'     : no downsampling (pass-through)
    """
    if method == "444":
        return channel_u8

    H, W = channel_u8.shape
    # Target 4:2:0 size is roughly half in both dims (OpenCV picks exact sizes cleanly)
    target_w = max(1, W // 2)
    target_h = max(1, H // 2)

    if method == "nearest":
        return cv2.resize(channel_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    elif method == "average":
        # INTER_AREA is OpenCV’s go-to for downsampling (box-like averaging)
        return cv2.resize(channel_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError("downsample_420 method must be 'nearest', 'average', or '444'")

def upsample_nn(channel_small: np.ndarray, target_shape: Tuple[int, int], method = "nearest"):
    """
    Bring a downsampled chroma channel back up to the target size
    using nearest-neighbor upsampling (repeat rows/columns).

    This doesn't invent new detail—it just stretches the smaller image back
    to match the Y (luma) resolution so we can combine channels again.
    """
    H_target, W_target = target_shape

    # If already correct size (e.g., 4:4:4), just return
    if channel_small.shape == (H_target, W_target):
        return channel_small

    if method == "nearest":
        return cv2.resize(channel_small, (W_target, H_target), interpolation=cv2.INTER_NEAREST)
    
    elif method == "average" or method == "444":  # treat 444 like linear when resizing (won't be hit if sizes match)
        return cv2.resize(channel_small, (W_target, H_target), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError("method must be 'nearest', 'average', or '444'")
    

def ycbcr_to_rgb_image(Y, Cb, Cr):
    """
    Combine Y, Cb, and Cr 2D arrays back into a single image and convert
    from YCbCr color space to standard RGB for saving/viewing.
    """
    # Ensure channels are uint8
    Y = np.clip(Y, 0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)

    ycbcr = np.stack([Y, Cb, Cr], axis = 2).astype(np.uint8)
    return Image.fromarray(ycbcr, mode = "YCbCr").convert("RGB")

def save_channel_as_grayscale_png(array, filename):
    """Scales array data to 0-255 and saves as a grayscale PNG."""
    # Ensure array is in the 0-255 range for proper visualization
    scaled_array = np.clip(array, 0, 255).astype(np.uint8)
    img = Image.fromarray(scaled_array, mode='L')
    img.save(filename)

# DCT and Quantization ---------------------------------------------

# Quantization Tables
# JPEG standard quantization tables
# Source: https://www.sciencedirect.com/topics/computer-science/quantization-table
# Same table seen in official JPEG source code (jcparam.c): 
# https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/jcparam.c

# These 8x8 tables say how much we "shrink" each DCT frequency in a block.
# Lower numbers = keep more detail; higher numbers = throw away more detail.
# Luminance (Y) uses a different table from chroma (Cb/Cr) because the eye
# is more sensitive to brightness detail than to color detail.

luma_quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

chroma_quantization_table = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 27, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


# Flat Quantization Table for experiment
flat_quantization_table = np.full((8, 8), 99)

# Quality scaling for QTables
def scale_qtable(base_qt: np.ndarray, quality: int) -> np.ndarray:
    """
    Scale a base 8x8 quantization table according to  'quality' (1..100). This will be used for the analysis part too
    Lower quality -> larger table values -> stronger quantization (more loss, higher compression).
    """
    quality = max(1, min(quality, 100))
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    qtable = (base_qt * scale + 50) // 100
    return np.clip(qtable, 1, 255).astype(int)

def print_matrix(matrix, title):
    """
    Function to print a matrix with a given title.
    """
    
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(f"{val:7.1f}" for val in row))
    print()


def pad_to_multiple_of_8(arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Padding a 2D array (image channel) so its height (H) and width (W) are multiples of 8.
    JPEG works on 8x8 blocks, so this helps for a  clean tiling.
    Padding is done by repeating the last row/column(edge padding) to avoid sharp borders.
    
    This function, returns:
      - padded array
      - original (H, W) so we can crop back after decoding
    """
    
    H, W = arr.shape
    H_pad = (8 - (H % 8)) % 8
    W_pad = (8 - (W % 8)) % 8
    if H_pad == 0 and W_pad == 0:
        return arr, (H, W)

    arr2 = arr
    if H_pad:
        pad_bottom = arr[-1:, :].repeat(H_pad, axis = 0)
        arr2 = np.vstack([arr, pad_bottom])

    arr3 = arr2
    if W_pad:
        pad_right = arr2[:, -1:].repeat(W_pad, axis = 1)
        arr3 = np.hstack([arr2, pad_right])

    return arr3, (H, W)


def unpad_to_shape(arr: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    """
    Cropping a previously padded array back to its original height and width.(reverse of the padding!)
    Use the (H, W) returned by pad_to_multiple_of_8.
    """
    
    H, W = original_shape
    return arr[:H, :W]

def block_view_8x8(channel: np.ndarray) -> Tuple[List[List[np.ndarray]], Tuple[int, int]]:
    """
    Spliting  a 2D channel into 8x8 blocks (without copying data unnecessarily).
    THis is basically how tht  JPEG processes images—block by block.

    This function returns:
      - blocks: a 2D list blocks[i][j] = the (i,j) 8x8 block
      - (h_blocks, w_blocks): number of blocks vertically and horizontally (height and the width!)
    """
    
    H, W = channel.shape
    h_blocks, w_blocks = H // 8, W // 8
    blocks = []
    
    for i in range(h_blocks):
        row = []
        for j in range(w_blocks):
            # Slicing  out the 8x8 region for this block
            row.append(channel[i*8 :(i+1)*8, j*8 : (j+1)*8])
        blocks.append(row)
        
    return blocks, (h_blocks, w_blocks)

def merge_blocks_8x8(blocks: List[List[np.ndarray]], dims: Tuple[int, int]) -> np.ndarray:
    
    """
    Reassembling a full 2D channel from its 8x8 blocks.
    This is the inverse of block_view_8x8: placing the  each block back in its position.
    """
    
    h_blocks, w_blocks = dims
    H, W = h_blocks*8, w_blocks*8
    out = np.zeros((H, W), dtype = np.float32)
    for i in range(h_blocks):
        for j in range(w_blocks):
            out[i*8 : (i+1)*8, j*8 : (j+1)*8] = blocks[i][j]
    return out

N = 8


def get_fixed_dct_factors(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Factors and cosine basis for more efficient DCT/IDCT computation.

    For context, the 1D DCT formula is:
        DCT[u] = C(u) * sum_{x=0}^{N-1} signal[x] * cos[(2x + 1) * u * pi / (2N)]

        where C(u) is the scaling factor:
            C(0) = 1/sqrt(N)
            C(u) = sqrt(2/N) for u > 0
    N is 8 in JPEG
    """
    # Using the same 8 basis functions for all rows/columns
    factors = np.zeros(N)
    # C(u): C(0)=1/sqrt(N), C(u)=sqrt(2/N) for u>0
    factors[0] = 1.0 / np.sqrt(N)
    factors[1:] = np.sqrt(2.0 / N)
    
    # Cosine basis functions (still needs to be multiplied with input signal later)
    cos_basis = np.zeros((N, N))
    for u in range(N):
        for x in range(N):
            cos_basis[u, x] = np.cos((2 * x + 1) * u * np.pi / (2 * N))
    return factors, cos_basis

# Constants for DCT/IDCT
SCALING_FACTORS, COSINE_BASIS = get_fixed_dct_factors(N)

def apply_1d_dct(signal: np.ndarray) -> np.ndarray:
    """
    Computes the 1D Discrete Cosine Transform (DCT) for a 1D signal (row or column).
    Uses the pre-calculated COSINE_BASIS and SCALING_FACTORS.
    """
    l = len(signal)
    output = np.zeros(l)
    vector = signal.astype(np.float64)

    for u in range(l):
        # Calculate the dot product of signal and bases function for frequency u
        sum_val = np.sum(vector * COSINE_BASIS[u, :])
        # Apply the pre-calculated scaling factor
        output[u] = SCALING_FACTORS[u] * sum_val
    return output

def DCT_2d(signal: np.ndarray) -> np.ndarray:
    """
    Two passes of 1D DCT to compute the 2D DCT using the separability property.
    """
    signal = signal.astype(np.float64)

    # Apply 1D DCT to each row
    res = np.apply_along_axis(apply_1d_dct, axis=1, arr=signal)

    # Transpose (flip) and apply to columns
    res = res.T
    res = np.apply_along_axis(apply_1d_dct, axis=1, arr=res)

    # Transpose back to the original
    return res.T


def dct_inv_1d(signal: np.ndarray) -> np.ndarray:
    """
    Computes the 1D Inverse Discrete Cosine Transform (IDCT) for an 8-element vector.
    Uses the pre-calculated factors for O(N) efficiency.

    Equation for 1D Inverse DCT:
        signal[x] = sum_{u=0}^{N-1} C(u) * DCT[u] * cos[(2x + 1) * u * pi / (2N)]
    """
    l = len(signal)
    res = np.zeros(l)
    for x in range(l):
        # Sum(C(u) * signal[u] * cos[u, x])
        sum_terms = SCALING_FACTORS * signal * COSINE_BASIS[:, x]
        res[x] = np.sum(sum_terms)
        
    return res

def IDCT_2d(coefficients: np.ndarray) -> np.ndarray:
    """
    Computes the 2D IDCT using the separable property (two passes of 1D IDCT).
    """
    coefficients = coefficients.astype(np.float64)
    res = np.apply_along_axis(dct_inv_1d, axis=1, arr=coefficients)

    # np transpose the result to apply to columns
    res = res.T
    res = np.apply_along_axis(dct_inv_1d, axis=1, arr=res)
    return res.T


def quantize(dct_block, qtable):
    """
    Standard Uniform Scalar Quantization (USQ).
    Quantize a DCT 8x8 block by dividing each coefficient by a matching
    number from the 8x8 quantization table, then rounding to integers.
    Bigger table values → more aggressive compression (more loss).
    """
    return np.round(dct_block / qtable).astype(int)

def dequantize(quantized_block, qtable):
    """
    Standard USQ Dequantization.
    Reverse quantization by multiplying the integer coefficients back
    by the quantization table. This approximates the original DCT block,
    but the earlier rounding means some detail is permanently lost.
    """
    return (quantized_block * qtable).astype(np.float32)

# Zig-Zag and RLE (Start of Entropy Encoding) ---------------------------------------------

def zigzag_indices(n =8):
    """
    Create the visiting order for an nxn block in JPEG's zig-zag pattern.
    Why zig-zag? After DCT, important (low-frequency) coefficients are near
    the top-left and less important are bottom-right. Zig-zag walks from
    low to high frequency so long runs of zeros appear at the end, which
    makes run-length encoding (RLE) very effective.
    """
#  x  0  1  2  3  4  5  6  7  
#  0  0  1  5  6 14 15 27 28
#  1  2  4  7 13 16 26 29 42
#  2  3  8 12 17 25 30 41 43
#  3  9 11 18 24 31 40 44 53
#  4 10 19 23 32 39 45 52 54
#  5 20 22 33 38 46 51 55 60
#  6 21 34 37 47 50 56 59 61
#  7 35 36 48 49 57 58 62 63

    idx = []
    for s in range(2*n - 1):
        if s % 2 == 0:  # even diagonal: down-left
            r = 0 if s < n else s - n + 1
            c = s if s < n else n - 1
            while r < n and c >= 0:
                idx.append((r, c))
                r += 1; c -= 1
        else:  # odd diagonal: up-right
            r = s if s < n else n - 1
            c = 0 if s < n else s - n + 1
            while r >= 0 and c < n:
                idx.append((r, c))
                r -= 1; c += 1
    return idx

# Precomputing the  zig-zag order for 8x8 ( this is the standard JPEG block size)
zz = zigzag_indices(8)

def zigzag_flat(block_8x8: np.ndarray) -> List[int]:
    """
    Read an 8x8 block's values in zig-zag order to produce a 1D list of 64 ints.
    """
    return [int(block_8x8[r, c]) for (r, c) in zz]

def izigzag_flat(coeffs_64: List[int]) -> np.ndarray:
    """
    Inverse of zigzag_flatten: take a 64-length list and place values back
    into an 8x8 block using the zig-zag positions.
    """
    out = np.zeros((8, 8), dtype = np.float32)
    for k, (r, c) in enumerate( zz):
        out[r, c] = coeffs_64[k]
    return out

def rle_encode_block(coeffs_64: List[int], prev_dc: int) -> Tuple[List[Tuple], int]:
    """
     RLE for one block's 64 zig-zag coefficients.
    Output is a token list like:
       [('DC', dc_diff), ('AC', run, val), ... , ('EOB',)]
       
    - DC is stored as a difference from the previous block's DC (saves bits).
    - AC uses (run, val) pairs where 'run' counts consecutive zeros before
      a nonzero 'val'. Long zero tails end with EOB (End Of Block).
    - For very long zero runs, JPEG uses ZRL (Zero Run Length = 16 zeros).
      We do that by tsking the ('AC', 16, 0) chunks when needed.
      
    This function returns the tokens and this block's DC to use as next block's prev_dc.
    """
    dc = coeffs_64[0]
    dc_diff = dc - prev_dc
    tokens = [('DC', dc_diff)]

    # Encode AC coefficients (positions 1..63)
    run = 0
    for v in coeffs_64[1:]:
        if v == 0:
            run += 1
            # We defer emitting until we see a nonzero or reach the end.
        else:
            # Break long zero run into 16-zero chunks like JPEG's ZRL.
            while run > 15:
                tokens.append(('AC', 16, 0))
                run -= 16
            tokens.append(('AC', run, int(v)))
            run = 0

    # If the block ends with zeros, close with EOB
    if run > 0:
        tokens.append(('EOB',))
    return tokens, dc

def rle_decode_block(tokens: List[Tuple], prev_dc: int) -> Tuple[List[int], int]:
    """
    Inverse of rle_encode_block. 
    Rebuilding  the 64 zig-zag coefficients from tokens.
    - First token must be ('DC', diff); recover absolute DC using prev_dc.
    - Then place ACs, skipping 'run' zeros before each nonzero value.
    - Stop at EOB or when we've filled 64 entries.
    Returns the recovered coeffs list and the absolute DC for the next block's prev_dc.
    """
    coeffs = [0]*64

    # DC
    assert tokens[0][0] == 'DC'
    dc = prev_dc + int(tokens[0][1])
    coeffs[0] = dc

    # AC
    k = 1  # index into positions 1..63
    for t in tokens[1:]:
        if t[0] == 'EOB':
            break
        if t[0] == 'AC':
            run, val = int(t[1]), int(t[2])
            k += run  # skippin the 'run' zeros
            
            if k >= 64:
                break
            
            coeffs[k] = val
            k += 1
        else:
            # if it is  Unknown token kind; ignoring it.
            pass

    return coeffs, dc

# Huffman Coding (More entropy coding) ---------------------------------------------

class HuffmanNode:
    """
    Same as for the Huffman Assignment
    Node for a Huffman tree.
      - Leaf:   node.symbol is a symbol string, no children
      - Branch: node.symbol is None, has left/right children
      
    The 'freq' field is used when building the tree: higher freq ->>> shorter code.
    """
    __slots__ = ("symbol", "freq", "left", "right")

    def __init__(self, symbol = None, freq = 0, left = None, right = None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        # Needed so nodes can live in a priority queue ordered by frequency.
        return self.freq < other.freq


def build_huffman_codes(symbols: List[Any]) -> Dict[str, str]:
    """
    Build a Huffman code table (symbol -> bitstring) from observed symbol frequencies.

    symbols : list
    Stream of symbols (maybe tokens like ('DC', diff), ('AC', run, val), ('EOB',)).

    This function returns
    codes : dict[str, str]
        Mapping from repr(symbol) ->> bitstring (e.g., "('AC', 0, 3)" --> "00101").

   
    - We use repr(symbol) as the  key so tuples are handled consistently.
    - If there's only one unique symbol, we still assign it a code "0" to be valid.
    """
    
    import heapq

    def key(s: Any) -> str:
        return repr(s)

    #  Counting frequencies
    freq: Dict[str, int] = {}
    for s in symbols:
        k = key(s)
        freq[k] = freq.get(k, 0) + 1

    # Initializing a min-heap of leaf nodes
    heap: List[HuffmanNode] = [HuffmanNode(symbol = k, freq = v) for k, v in freq.items()]
    heapq.heapify(heap)

    # Edge case: only one unique symbol
    if len(heap) == 1:
        only = heap.pop()
        return {only.symbol: "0"}

    #  Building the tree by merging the two least frequent nodes repeatedly
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        parent = HuffmanNode(freq = a.freq + b.freq, left = a, right = b)
        heapq.heappush(heap, parent)

    root = heap[0]

    # Walking the tree and assign codes: left = 0, right = 1
    codes: Dict[str, str] = {}

    def assign_codes(node: HuffmanNode, prefix: str) -> None:
        if node.symbol is not None:             # leaf
            codes[node.symbol] = prefix or "0"  # ensure non-empty
            return
        
        assign_codes(node.left,  prefix + "0")
        assign_codes(node.right, prefix + "1")

    assign_codes(root, "")
    return codes


def huffman_encode(tokens: List[Any], huffman_codes: Dict[str, str]) -> str:
    """
    Encode a list of tokens into a single bitstring using the provided Huffman codes.

    - We convert each token to repr(token) so it matches the code table key.
    """
    def key(s: Any) -> str:
        return repr(s)
    
    return "".join(huffman_codes[key(t)] for t in tokens)


def huffman_decode(bitstring: str, huffman_codes: Dict[str, str]) -> List[Any]:
    """
    Decode a bitstring back into tokens using the code table.

    Implementation details:
    - We invert the code table (bits → repr(symbol)) and scan the bitstring,
      growing a prefix until it matches a code; then we emit that symbol.
    - We use eval() to convert the repr-string back to its original Python object
      (e.g., "('AC', 0, 3)" → ('AC', 0, 3)). For untrusted inputs, replace eval
      with a safer parser.
    """
    # Build inverse map: "0101" ->> "('AC', 0, 3)"
    inv: Dict[str, str] = {bits: sym for sym, bits in huffman_codes.items()}

    out: List[Any] = []
    prefix = ""
    
    for bit in bitstring:
        prefix += bit
        
        if prefix in inv:
            sym_repr = inv[prefix]
            out.append(eval(sym_repr))  # turn repr back into the original tuple
            prefix = ""                 # reset and continue scanning
    # If 'prefix' is non-empty here, the stream ended mid-symbol; we ignore gracefully.
    return out

# Bitstring Packing/Unpacking ---------------------------------------------

def pack_bitstring_to_bytes(bitstring: str) -> Tuple[bytes, int]:
    """
    Packs a string of '0's and '1's into a raw bytes object. Need to make sure that exact arbitrary bitstrings can be stored.
    If the bitstring length is not a multiple of 8, pad with '0's at the end to fill the last byte.

    """
    num_bits = len(bitstring)
    
    # Pad the string with '0's at the end to make its length a multiple of 8
    padding = (8 - (num_bits % 8)) % 8
    bitstring += '0' * padding
    
    # Convert 8-bit chunks into bytes
    b_arr = bytearray()
    for i in range(0, len(bitstring), 8):
        byte_str = bitstring[i : i+8]
        b_arr.append(int(byte_str, 2))
        
    return bytes(b_arr), num_bits

def unpack_bytes_to_bitstring(packed_bytes: bytes, num_bits: int) -> str:
    """
    Unpacks raw bytes back into a string of '0's and '1's.
    'num_bits' is used to truncate the final padded byte.
    """
    bitstring = ""
    for byte in packed_bytes:
        bitstring += f'{byte:08b}'
        
    # Remove the padding
    return bitstring[:num_bits]

# Channel encode/decode helper methods ---------------------------------------------

def encode_channel(channel_u8: np.ndarray, base_qt: np.ndarray, quality: int, 
                   quantization_method: str = "standard", 
                   print_first_block: bool = False, collect_stats: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Encode a single channel (Y, Cb or Cr) all the way to a Huffman-compressed bitstring.

    Steps per 8x8 block:
      1) Shift by -128 (center around 0) -> 
      2) DCT -> 
      3) Quantize (quality-scaled table, selectable method)
      4) Zig-zag reorder ->>
      5) RLE 
    After all blocks: build Huffman codes from token frequencies and encode the stream.

    This function returns:(This will give the information for it to be able to be decoded and be converted into a RGB oimage)
    meta : dict { 'h_blocks', 'w_blocks',            # block grid size, the height and the width
          'original_shape', 'padded_shape',  # for unpadding after decode
          'huffman_codes',                   # dict repr(symbol) -> bits
          'bitstring',                       # compressed token stream
          'dc_init',                         # baseline = 0
          'qtable',                          # the   8x8 quantization table (as list of lists)
          'quantization_method'              # 'standard' or 'flat'
          }
    """
    
    # The 'debug' dictionary stores intermediate results from the first 8x8 block during encoding to help visualize
    # and verify each processing step. It saves the original, shifted, DCT-transformed, and quantized versions of the block for inspection.
    # This allows you to check that each stage of the JPEG compression pipeline is working correctly without affecting the final encoded output.

    debug: Dict[str, Any] = {}

    # Tile into 8×8 blocks (padding to multiple of 8 with edge pixels)
    padded, original_shape = pad_to_multiple_of_8(channel_u8)
    blocks, (h_blocks, w_blocks) = block_view_8x8(padded)

    # Scaling the quantization table by requested quality
    qtable = scale_qtable(base_qt, quality)
    
    # Collect all tokens from all blocks to get the good of a Huffman codes
    all_tokens: List[Any] = []
    # Collect all quantized coefficients for entropy calculation
    all_coeffs_flat: List[int] = []
    prev_dc = 0  # for DC differential coding across blocks

    # Stats containers
    stats = None
    if collect_stats:
        stats = {
            "num_blocks": 0,
            "occupancy_counts": np.zeros((8, 8), dtype=np.int64),  # count of non-zero per (u,v)
            "zero_run_hist": {}  # run_length -> count
        }

        def _accum_zero_runs(ac_coeffs_63: List[int]):
            run = 0
            for v in ac_coeffs_63:
                if v == 0:
                    run += 1
                else:
                    if run > 0:
                        stats["zero_run_hist"][run] = stats["zero_run_hist"].get(run, 0) + 1
                        run = 0
            if run > 0:
                stats["zero_run_hist"][run] = stats["zero_run_hist"].get(run, 0) + 1

    for bi in range(h_blocks):
        for bj in range(w_blocks):
            block = blocks[bi][bj].astype(np.float32) - 128.0

            if bi == 0 and bj == 0 and print_first_block:
                debug["orig_block"] = blocks[bi][bj].copy()
                debug["shifted_block"] = block.copy()

            dct_block = DCT_2d(block)
            if bi == 0 and bj == 0 and print_first_block:
                debug["dct_block"] = dct_block.copy()

            q_block = quantize(dct_block, qtable)
            if bi == 0 and bj == 0 and print_first_block:
                debug["quantized_block"] = q_block.copy()

            # ----- stats: occupancy per coefficient position -----
            if collect_stats:
                stats["num_blocks"] += 1
                stats["occupancy_counts"] += (q_block != 0).astype(np.int64)

            zz_list = zigzag_flat(q_block)

            # ----- stats: zero-run histogram over the AC stream -----
            if collect_stats:
                _accum_zero_runs(zz_list[1:])  # ACs only

            tokens, prev_dc = rle_encode_block(zz_list, prev_dc)
            all_tokens.extend(tokens)
    
    # Getting the  Huffman codes for this channel and encode the token stream
    huff_codes = build_huffman_codes(all_tokens)
    bitstring_str = huffman_encode(all_tokens, huff_codes)

    packed_bytes, num_bits = pack_bitstring_to_bytes(bitstring_str)

    meta: Dict[str, Any] = {
        "h_blocks": h_blocks,
        "w_blocks": w_blocks,
        "original_shape": original_shape,
        "padded_shape": padded.shape,
        "huffman_codes": huff_codes,    
        "packed_bitstream": packed_bytes,
        "num_bits": num_bits,
        "dc_init": 0,
        "qtable": qtable.tolist(),
        "quantization_method": quantization_method,
        }
    if collect_stats:
        # Convert numpy to lists for JSON friendliness; keep counts as plain types
        meta["stats"] = {
            "num_blocks": int(stats["num_blocks"]),
            "occupancy_counts": stats["occupancy_counts"].tolist(),
            "zero_run_hist": {str(int(k)): int(v) for k, v in stats["zero_run_hist"].items()}}
    return meta, debug


def decode_channel(meta: Dict[str, Any], print_first_block: bool = False) -> np.ndarray:
    """
    Decode a single channel from the stored metadata:
      Huffman bits -> tokens -> inverse RLE -> inverse zig-zag ->  dequant ->  IDCT -> +128 -> back to [0,255] for making it RGB.
    Finally, merge 8x8 blocks and unpad back to original shape.
    """
    
    h_blocks: int = meta["h_blocks"]
    w_blocks: int = meta["w_blocks"]
    padded_shape = tuple(meta["padded_shape"]) 
    huff_codes: Dict[str, str] = meta["huffman_codes"]
    qtable = np.array(meta["qtable"], dtype=int)
    qtable_base = np.array(meta["qtable"], dtype=int)
    packed_bytes: bytes = meta["packed_bitstream"]
    num_bits: int = meta["num_bits"]
    quantization_method: str = meta.get("quantization_method", "standard")


    # Unpack bytes to bitstring
    bitstring = unpack_bytes_to_bitstring(packed_bytes, num_bits)

    # Huffman decode to token stream
    tokens = huffman_decode(bitstring, huff_codes)
    # Rebuilding the  8×8 blocks from tokens
    blocks: List[List[np.ndarray]] = []
    prev_dc = 0
    t_idx = 0

    debug: Dict[str, Any] = {}

    for bi in range(h_blocks):
        row: List[np.ndarray] = []
        for bj in range(w_blocks):
            block_qtable = qtable_base
            
            # Each block must start with a DC token, the first (left-top) value in the grid
            if t_idx >= len(tokens) or tokens[t_idx][0] != 'DC':
                raise ValueError("Corrupt stream: expected 'DC' token at start of block")

            block_tokens = [tokens[t_idx]]
            t_idx += 1

            # Keep getting the  AC/EOB tokens until EOB or until the block is fully filled.
            ac_nonzeros = 0
            while t_idx < len(tokens):
                tok = tokens[t_idx]
                
                if tok[0] == 'DC':
                    break
                
                block_tokens.append(tok)
                t_idx += 1 # Consume this token
                
                if tok[0] == 'EOB':
                    # This was the EOB token, so this block is done.
                    break

            # Inversing the RLE to 64 zig-zag coefficients
            coeffs, prev_dc = rle_decode_block(block_tokens, prev_dc)

            # Placing it  back into 8×8, dequantize, inverse DCT, re-center and back to the RGB
            q_block = izigzag_flat(coeffs)
            
            deq = dequantize(q_block, block_qtable)

            spatial = IDCT_2d(deq) + 128.0
            spatial = np.clip(spatial, 0, 255)

            if bi == 0 and bj == 0 and print_first_block:
                debug["decoded_quant_block"] = q_block.copy()
                debug["decoded_dequant"] = deq.copy()
                debug["decoded_idct"] = spatial.copy()

            row.append(spatial.astype(np.float32))
        blocks.append(row)
        

    #  Merging the  blocks and unpadding to the original channel size
    reconstructed = merge_blocks_8x8(blocks, (h_blocks, w_blocks))
    return unpad_to_shape(reconstructed, tuple(meta["original_shape"])).astype(np.uint8)



# Full JPEG encode /decode pipelines ---------------------------------------------

def jpeg_encode_pipeline(source, config, show_first_block=False):    
    """
    Loading the  RGB
    RGB -> YCbCr
    4:2:0 on Cb, Cr
    DCT -> Quantize -> ZigZag -> RLE -> Huffman
    This function returns a dict  with three per-channel streams + image geometry that will help for it to be reconstructed.
    """

    quality = config['quality']
    quantization_method = config['quantization_method']
    chroma_method = config['chroma_method']
    collect_stats = config.get("collect_stats", False)

    if isinstance(source, str):
        img = ensure_rgb(Image.open(source))
    elif isinstance(source, Image.Image):
        img = ensure_rgb(source)
    else:
        raise ValueError(f"source must be a file path (str) or PIL.Image object, but got {type(source)}")

    Y, Cb, Cr = rgb_to_ycbcr_arrays(img)

    H, W = Y.shape

    # Subsample chroma
    Cb_ds = downsample_420(Cb, chroma_method)
    Cr_ds = downsample_420(Cr, chroma_method)

    if quantization_method == "standard":
        y_base_table = luma_quantization_table
        c_base_table = chroma_quantization_table
    else:
        y_base_table = flat_quantization_table
        c_base_table = flat_quantization_table

    y_meta, y_dbg = encode_channel(Y, y_base_table, quality, 
                                   quantization_method=quantization_method,
                                   print_first_block=show_first_block, collect_stats=collect_stats)
    cb_meta, _ = encode_channel(Cb_ds, c_base_table, quality, 
                                quantization_method=quantization_method,
                                print_first_block=False, collect_stats=collect_stats)
    cr_meta, _ = encode_channel(Cr_ds, c_base_table, quality, 
                                quantization_method=quantization_method,
                                print_first_block=False, collect_stats=collect_stats)

    # This is foer tbe debugging purpose only as it  prints for the very first Y block
    if show_first_block:
        print_matrix(y_dbg["orig_block"], "Original 8x8 Block (Y, pixels)")
        print_matrix(y_dbg["shifted_block"], "Shifted Block (Y)")
        print_matrix(y_dbg["dct_block"], "DCT Coefficients (Y)")
        print_matrix(y_dbg["quantized_block"], "Quantized DCT (Y)")

    meta = {  
        "width": W,
        "height": H,
        "chroma_method": chroma_method,
        "quality": quality,
        "Y": y_meta,
        "Cb": cb_meta,
        "Cr": cr_meta, 
        "Y_original": Y,
        "Cb_original": Cb_ds,
        "Cr_original": Cr_ds,
        "config": config
        }
    
    return meta

def jpeg_decode_pipeline(meta: Dict[str, Any]) -> Image.Image:
    """
    Huffman decode -> RLE inverse -> inverse zig-zag -> dequant -> IDCT
    Upsample Cb/Cr back to Y dims
    YCbCr -> RGB
    """
    W, H = meta["width"], meta["height"]


    Y = decode_channel(meta["Y"], print_first_block=True)
    Cb_ds = decode_channel(meta["Cb"])
    Cr_ds = decode_channel(meta["Cr"])

    # Upsampling the chroma back to Y size
    Cb = upsample_nn(Cb_ds, (H, W), meta["chroma_method"])
    Cr = upsample_nn(Cr_ds, (H, W), meta["chroma_method"])
    
    return ycbcr_to_rgb_image(Y, Cb, Cr)


# Main methods that runs all of the code, does the encoding as well as the decoding !

def main():
    parser = argparse.ArgumentParser(description="JPEG Pipeline")
    parser.add_argument("input", help = "Input image (bmp, tif, jpg, png, etc. any raw file which are large in size)")
    
    parser.add_argument("--quality", type  = int, default = 50, help = "Quality (1-95))")
    parser.add_argument("--method", choices=["nearest", "average", "444"], default = "nearest", help = "Chroma 4:2:0 downsampling method, choose one of them")
    parser.add_argument("--qmethod", choices=["standard", "flat"], default="standard", help="Quantization method (standard USQ or flat)")
    parser.add_argument("--out", default = "reconstructed_image.png", help = "Output: reconstructed RGB image")
    args = parser.parse_args()
    
    test_config = {
        'quality': args.quality,
        'quantization_method': args.qmethod,
        'chroma_method': args.method
    }

    meta = jpeg_encode_pipeline(args.input, config=test_config, show_first_block = True)
    recon = jpeg_decode_pipeline(meta)
    recon.save(args.out)

    print(f"\nSaved reconstructed image is saved in the file {args.out}")

if __name__ == "__main__":
    test_config = {
        'quality': 5,
        'quantization_method': 'standard',
        'chroma_method': 'nearest',
    }
    
    print("Testing...")
    meta = jpeg_encode_pipeline(
        'images/desert-ribbons.tif',
        config=test_config,
        show_first_block=True
    )
    print("Testing standard decode")
    recon_standard = jpeg_decode_pipeline(meta)
    recon_standard.save('reconstructed_standard.png')
    print("Saved to reconstructed_standard.png")
