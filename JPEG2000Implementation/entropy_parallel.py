from adaptive_arithmetic_coding import ArithmeticEncoder, ArithmeticDecoder, Model
import io
import numpy as np
from typing import List, Tuple, Dict

# Convert numpy array block to flat byte-like list of symbols
def flatten_block_to_symbols(block: np.ndarray) -> List[int]:
    # Shift values to make them non-negative
    flat = block.flatten()
    offset = abs(flat.min()) if flat.min() < 0 else 0
    return flat + offset, offset

# Entropy encode each block
def entropy_encode_blocks(blocks: List[Dict]) -> List[Dict]:
    encoded_blocks = []

    for block in blocks:
        data = block['data']
        flat_data, offset = flatten_block_to_symbols(data)

        # Create a symbol model from the data
        model = Model(alphabet=range(0, int(flat_data.max()) + 2))  # Adaptive model
        output_stream = io.BytesIO()
        encoder = ArithmeticEncoder(model, output_stream)

        for symbol in flat_data:
            encoder.write(symbol)
        encoder.finish()

        compressed_bytes = output_stream.getvalue()

        encoded_blocks.append({
            'component': block['component'],
            'level': block['level'],
            'band': block['band'],
            'position': block['position'],
            'shape': block['shape'],
            'offset': offset,
            'encoded': compressed_bytes,
        })

    return encoded_blocks


def entropy_decode_block(encoded: bytes, shape: Tuple[int, int], offset: int, alphabet_range: int) -> np.ndarray:
    model = Model(alphabet=range(alphabet_range + 1))
    input_stream = io.BytesIO(encoded)
    decoder = ArithmeticDecoder(model, input_stream)

    num_values = shape[0] * shape[1]
    decoded = np.array([decoder.read() for _ in range(num_values)]) - offset
    return decoded.reshape(shape)
