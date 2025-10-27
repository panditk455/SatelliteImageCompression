from typing import List, Tuple, Dict
import numpy as np

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