# part1.py
# This contains the first 2 part of the JPEG Compression which is the Color Space conversion as well as the Chroma Downsampling


import os
import sys
import argparse
import numpy as np
from PIL import Image

# Convert an RGB image to Y, Cb, Cr numpy arrays.

def rgb_to_ycbcr_arrays(img: Image.Image):

    ycbcr = img.convert("YCbCr")
    arr = np.asarray(ycbcr, dtype = np.uint8)
    
    Y, Cb, Cr = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return Y, Cb, Cr

# Downsample a chroma channel to 4:2:0

def downsample_420(channel_2d: np.ndarray, method = "nearest"):

    # channel_2d.shape is a NumPy telling the size of  image
    # unpack it into H and W to use later when resizing, averaging, or reconstructing the image.
    
    H, W = channel_2d.shape
    H_even, W_even = H - (H % 2), W - (W % 2)
    c = channel_2d[:H_even, :W_even]

    if method == "nearest":
        # Take every 2nd row and every 2nd column.
        # This means we "skip" one pixel each time in both directions,
        # effectively keeping only 1 pixel out of every 4 (reduces width & height by half).
        # It's called "nearest" because we just pick one nearby pixel instead of averaging.
        return c[::2, ::2]
    
    elif method == "average":
            # This method averages 2x2 pixel blocks to represent each color region more accurately.
            # Reshape the 2D array (H_even x W_even) into smaller 4D blocks:
            # - (H_even // 2) groups every 2 rows together
            # - (W_even // 2) groups every 2 columns together
            # So each 2x2 block of pixels becomes one small group we can average.
            # Example:
            # Original:  [[a, b], [c, d]]
            # Reshaped:  [[[[a, b], [c, d]]]]
            
        c4 = c.reshape(H_even // 2, 2, W_even // 2, 2)
        return c4.mean(axis = (1, 3)).astype(np.uint8)
    
    else:
        raise ValueError("method must be 'nearest' or 'average' and not nonw of it")

# Nearest-neighbor upsample by repeating rows and columns.

def upsample_nn(channel_small: np.ndarray, target_shape):
    
    h, w = channel_small.shape
    H, W = target_shape
    up = np.repeat(np.repeat(channel_small, 2, axis = 0), 2, axis = 1)
    
    return up[:H, :W]


# Combine Y, Cb, Cr arrays and convert back to RGB

def ycbcr_to_rgb_image(Y, Cb, Cr):

    ycbcr = np.stack([Y, Cb, Cr], axis = 2).astype(np.uint8)
    return Image.fromarray(ycbcr, mode="YCbCr").convert("RGB")



def ensure_rgb(img):
    return img.convert("RGB")


# Function for processing an image. using everything we created up and saving and seeing how it looks:
# RGB->YCbCr->Downsample->Reconstruct-> See all the reults saved in outpit_part1 to see how things are and what it actually means with it
def process_one_image(in_path: str, outdir = "output_part1", method = "nearest"):

    os.makedirs(outdir, exist_ok = True)
    base = os.path.splitext(os.path.basename(in_path))[0]

    img = ensure_rgb(Image.open(in_path))
    W, H = img.size

    Y, Cb, Cr = rgb_to_ycbcr_arrays(img)
    Cb_ds, Cr_ds = downsample_420(Cb, method), downsample_420(Cr, method)

    Cb_up, Cr_up = upsample_nn(Cb_ds, Y.shape), upsample_nn(Cr_ds, Y.shape)
    recon = ycbcr_to_rgb_image(Y, Cb_up, Cr_up)

    # Final reconstructed image after chroma downsampling and upsampling, we are see shows overall visual result in the output file!)
    recon.save(os.path.join(outdir, f"{base}_reconstructed_420_{method}.png"))

    # Y- luminance channel , grayscale, the whole brightness information  is saved as well as fine details preserved
    Image.fromarray(Y).save(os.path.join(outdir, f"{base}_Y.png"))

    # Cb channel - original color info before downsampling
    Image.fromarray(Cb).save(os.path.join(outdir, f"{base}_Cb_full.png"))

    #  Cr, red-difference chroma channel - original
    Image.fromarray(Cr).save(os.path.join(outdir, f"{base}_Cr_full.png"))

    # Cb channel after 4:2:0 downsampling, upsampled again for visualization (shows color data loss)
    Cb_ds_vis = Image.fromarray(upsample_nn(Cb_ds, Y.shape))

    # Cr channel after 4:2:0 downsamplin. upsampled again for visualization (shows color data loss)
    Cr_ds_vis = Image.fromarray(upsample_nn(Cr_ds, Y.shape))
    
    # Save visualized downsampled chroma channels for analysis (blurry color info)
    
    Cb_ds_vis.save(os.path.join(outdir, f"{base}_Cb_420view.png"))
    Cr_ds_vis.save(os.path.join(outdir, f"{base}_Cr_420view.png"))




# run it 
def main():
    parser = argparse.ArgumentParser(description = "JPEG Color conversion+ Chroma dOWNSAMpling.")
    parser.add_argument("images", nargs = "+", help = "Input image files (.bmp, .tif, .jpg, etc.)")
    parser.add_argument("--method", choices = ["nearest", "average"], default = "nearest")
    args = parser.parse_args()


    for p in args.images:
        try:
            process_one_image(p, "output_part1", args.method)
        except Exception as e:
            print(f"[ERROR] {p}: {e}", file = sys.stderr)


if __name__ == "__main__":
    main()



#How to run this file, parser says it
# python3 part1.py images/airplane.bmp --method nearest for preprocessing 1 image with nearest 

# python3 part1.py images/airplane.bmp --method average,
# python3 part1.py images/airplane.bmp , does the nearest by default
# python3 part1.py images/airplane.bmp images/boats.bmp images/goldhill.bmp images/inputfile5.tif --method average does this for all files

