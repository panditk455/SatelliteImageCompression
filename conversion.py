import numpy as np
from PIL import Image
import scipy.fftpack
from io import BytesIO

# JPEG standard quantization tables
# Source: https://www.sciencedirect.com/topics/computer-science/quantization-table
# Same table seen in official JPEG source code (jcparam.c): 
# https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/jcparam.c

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


def print_matrix(matrix, title):
    """
    Function to print a matrix with a given title.
    """
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(f"{val:6.1f}" for val in row))
    print()

def dct_2d(block):
    """
    2D Discrete Cosine Transform using scipy's optimized DCT.

    Input: 8x8 block of pixel values (shifted to range [-128, 127])
    Output: 8x8 block of DCT coefficients
    """
    return scipy.fft.dct(block, type=2, norm ='ortho')

def dct_2d_unoptimized(block):
    """
    2D Discrete Cosine Transform (unoptimized version for reference).
    Based on the mathematical definition of DCT-II:
    DCT(i,j) = (1/4) * C(i) * C(j) * Σ Σ f(x,y) * cos(π*i*(2x+1)/16) * cos(π*j*(2y+1)/16)
    Where C(u) = 1/√2 if u=0, else 1
    Iterating over each possible (i,j) basis function to compute all 64 DCT coefficients.

    Input: 8x8 block of pixel values (shifted to range [-128, 127])
    Output: 8x8 block of DCT coefficients
    """
    for i in range(8):
        for j in range(8):
            total = 0
            for x in range(8):
                for y in range(8):
                   total += block[x, y] * np.cos((2*x + 1) * i * np.pi / 16) * np.cos((2*y + 1) * j * np.pi / 16)
            ci = 1 / np.sqrt(2) if i == 0 else 1
            cj = 1 / np.sqrt(2) if j == 0 else 1
            block[i, j] = 0.25 * ci * cj * total
    return block

# TODO: Try to optimize the above DCT reference implementation using techniques that scipy uses (without using scipy directly)

def quantize(dct_block, qtable):
    """
    Q(u,v) = round(F(u,v) / QT(u,v))

    Input: 8x8 block of DCT coefficients, 8x8 quantization table
    Output: 8x8 block of quantized coefficients
    """
    return np.round(dct_block / qtable).astype(int)

def dequantize(quantized_block, qtable):
    """
    F'(u,v) = Q(u,v) * QT(u,v)

    Input: 8x8 block of quantized coefficients, 8x8 quantization table
    Output: 8x8 block of dequantized DCT coefficients
    """
    return quantized_block * qtable

def idct_2d(block):
    """
    2D Inverse Discrete Cosine Transform using scipy's optimized IDCT.
    """
    return scipy.fft.idct(block, type=2, norm = 'ortho')

def jpeg_encode_pipeline(input_path, quality=50, channel='Y'):
    """
    - Load image
    - Convert to YCbCr color space (not doing chroma subsampling)
    - Split into 8x8 blocks
    - Apply DCT to each block
    - Quantize DCT coefficients
    - RLE / Huffman Encoding (not implemented yet though)

    Input: Image path, quality (default 50), channel to process ('Y', 'Cb', or 'Cr')
    Output: List of quantized blocks, quantization table used, dimensions in blocks (h_blocks, w_blocks)
    """

    # Load image and make sure it is in RGB colorspace
    img = Image.open(input_path)
    img_rgb = img.convert('RGB')

    # Convert to YCbCr color space
    img_ycbcr = img_rgb.convert('YCbCr')
    img_array = np.array(img_ycbcr, dtype=np.float32)

    # Select the appropriate channel
    if channel == 'Y':
        selected_channel = img_array[:, :, 0]
        quantization_table = luma_quantization_table
    elif channel == 'Cb':
        selected_channel = img_array[:, :, 1]
        quantization_table = chroma_quantization_table
    elif channel == 'Cr':
        selected_channel = img_array[:, :, 2]
        quantization_table = chroma_quantization_table
    else:
        raise ValueError("Channel must be 'Y', 'Cb', or 'Cr'")

    # Adjust quantization table based on quality
    # Reference (JPEG source code jcparam.c): https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/jcparam.c
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2
    qtable = (quantization_table * scale + 50) / 100
    qtable = np.clip(qtable, 1, 255).astype(int)

    print(f"\nQuantization table (quality={quality}):")
    print_matrix(qtable, "Quantization Table")

    # Process image in 8x8 blocks
    h, w = selected_channel.shape

    # TODO: Handle images not multiple of 8 in width/height, bc this would skew the CR
    h_blocks = h // 8
    w_blocks = w // 8

    quantized_blocks = []

    for i in range(h_blocks):
        row_blocks = []
        for j in range(w_blocks):
            block = selected_channel[i*8:(i+1)*8, j*8:(j+1)*8]

            # Shift pixel values from [0, 255] to [-128, 127] so DCT centers around 0
            block_shifted = block - 128

            # Print matrices/details for first block only
            if i == 0 and j == 0:
                print_matrix(block, "Original 8x8 Block (pixels)")
                print_matrix(block_shifted, "Shifted Block")

            dct_block = dct_2d(block_shifted)

            if i == 0 and j == 0:
                print_matrix(dct_block, "DCT Coefficients")

            quantized = quantize(dct_block, qtable)

            if i == 0 and j == 0:
                print_matrix(quantized, "Quantized DCT Coefficients")
            row_blocks.append(quantized)
        quantized_blocks.append(row_blocks)
    return quantized_blocks, qtable, (h_blocks, w_blocks)

#TODO: Implement Huffman encoding and decoding for the quantized coefficients

def jpeg_decode_pipeline(quantized_blocks, qtable, block_dims):
    """
    - Dequantize coefficients
    - Apply inverse DCT
    - Shift values back to [0, 255]
    - Reconstruct image

    Input: List of quantized blocks, quantization table used, dimensions in blocks (h_blocks, w_blocks)
    Output: Reconstructed channel as a 2D numpy array
    """
    h_blocks, w_blocks = block_dims
    h = h_blocks * 8
    w = w_blocks * 8

    reconstructed = np.zeros((h, w))

    for i in range(h_blocks):
        for j in range(w_blocks):
            quantized = quantized_blocks[i][j]

            # Show details for first block
            if i == 0 and j == 0:
                print_matrix(quantized, "Quantized DCT (from encoder)")

            dequantized = dequantize(quantized, qtable)

            if i == 0 and j == 0:
                print_matrix(dequantized, "Dequantized DCT Coefficients")

            reconstructed_block = idct_2d(dequantized)

            if i == 0 and j == 0:
                print_matrix(reconstructed_block, "After IDCT (still centered)")

            reconstructed_block = reconstructed_block + 128
            reconstructed_block = np.clip(reconstructed_block, 0, 255)

            if i == 0 and j == 0:
                print_matrix(reconstructed_block, "Reconstructed Block (pixels)")

            reconstructed[i*8:(i+1)*8, j*8:(j+1)*8] = reconstructed_block
    return reconstructed

if __name__ == "__main__":
    input_file = "images/deep-blue-cubism.tif"

    # Y Channel
    quantized, qtable, dims = jpeg_encode_pipeline(input_file)
    reconstructed_y = jpeg_decode_pipeline(quantized, qtable, dims)
    reconstructed_img = Image.fromarray(reconstructed_y.astype(np.uint8), mode='L')
    reconstructed_img.save("y_channel.png")

    # Cb Channel
    quantized, qtable, dims = jpeg_encode_pipeline(input_file, channel='Cb')
    reconstructed_cb = jpeg_decode_pipeline(quantized, qtable, dims)
    reconstructed_img = Image.fromarray(reconstructed_cb.astype(np.uint8), mode='L')
    reconstructed_img.save("cb_channel.png")

    # Cr Channel
    quantized, qtable, dims = jpeg_encode_pipeline(input_file, channel='Cr')
    reconstructed_cr = jpeg_decode_pipeline(quantized, qtable, dims)
    reconstructed_img = Image.fromarray(reconstructed_cr.astype(np.uint8), mode='L')
    reconstructed_img.save("cr_channel.png")

    # Combine channels back to YCbCr and convert to RGB
    h, w = reconstructed_y.shape
    reconstructed_ycbcr = np.zeros((h, w, 3), dtype=np.uint8)
    reconstructed_ycbcr[:, :, 0] = reconstructed_y
    reconstructed_ycbcr[:, :, 1] = reconstructed_cb
    reconstructed_ycbcr[:, :, 2] = reconstructed_cr
    reconstructed_rgb = Image.fromarray(reconstructed_ycbcr, mode='YCbCr').convert('RGB')
    # Full JPEG pipeline implemented (basically)!
    reconstructed_rgb.save("reconstructed_image.png")
