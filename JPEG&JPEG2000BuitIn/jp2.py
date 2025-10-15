from PIL import Image
import glymur as gly
import numpy as np

# Read a TIFF or BMP
im = Image.open("re-entry.tif")
# Ensure 8/16-bit per channel; convert if needed
if im.mode not in ("L", "RGB", "I;16", "I;16B"):
    im = im.convert("RGB")

arr = np.array(im)

# Encode to JP2
jp2 = gly.Jp2k(
    "output.jp2",
    data=arr,
    psnr=None,                 # or set e.g. [38, 42, 45] per layer
    cratios=[20],              # compression ratios for quality layers
    irreversible=True,         # use 9/7; set False for lossless 5/3
    numres=6,                  # DWT levels (resolutions)
    cbsize=(64, 64),           # code-block size
    tilesize=(1024, 1024),     # tiling if image is large
    prog="LRCP"                # progression order
)

# Decode back
decoded = gly.Jp2k("output.jp2")[:]
Image.fromarray(decoded).save("roundtrip.png")
