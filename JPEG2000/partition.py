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