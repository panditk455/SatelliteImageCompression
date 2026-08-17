# JPEG & JPEG‑2000

Authors: **Kritika Pandit**, **Anika Rajbhandary**, **Leon Liang**, and **Lucas Schattenmann**

This repository contains a **library implementation** and **manual implementation** of both **JPEG** and **JPEG‑2000** pipelines and **scripts for metrics** to analyze compression performance on **satellite imagery**.

---

## What’s in the repo?

### **Conversion & Metrics**

The first script provides conversion to **baseline JPEG** and **JPEG‑2000 (JP2)** formats. It evaluates compression efficiency and distortion using the following metrics:

- **Mean Squared Error (MSE)**
- **Peak Signal‑to‑Noise Ratio (PSNR)**
- **Compression ratio (CR)**
- **Structural Similarity Index (SSIM)**

---

### **JPEG Pipeline**

Implements a full step‑by‑step JPEG encoder and decoder from scratch. It compresses images into the JPEG domain and reconstructs them to verify loss and quality.

**Encoding:**

- **RGB → YCbCr** color conversion
- **4:2:0 Chroma Downsampling** (nearest or average)
- **8×8 DCT‑II (ortho)** transformation
- **Quantization** using scaled luminance/chrominance tables
- **Zig‑zag ordering -> Run‑Length Encoding (RLE) -> Huffman encoding**

**Decoding:**

- **Inverse Huffman -> Inverse RLE -> Inverse Zig‑zag**
- **Dequantization -> Inverse DCT -> YCbCr -> RGB reconstruction**

---

### **JPEG‑2000 Pipeline**

Implements a full step‑by‑step JPEG‑2000 (JP2) encoder and decoder, It uses discrete wavelet transforms (DWT) instead of block‑based DCT.

**Encoding:**

- **RGB** -> **YCbCr color conversion**
- **Discrete Wavelet Transform (DWT)**
- **Quantization of sub‑bands**
- **Bitplane coding via Embedded Block Coding with Optimized Truncation (EBCOT)**
- **Entropy coding (arithmetic coding) producing final bitstream**

**Decoding:**

- **Inverse entropy decoding** -> **Inverse bitplane reconstruction**
- **Inverse quantization** -> **Inverse DWT**
- **Inverse YCbCr** -> **RGB reconstruction**

---

## Install

_Work in progress:_ will be finalized once all files are merged to the main branch.

Set up for local testing:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pillow scipy imagecodecs opencv-python scikit-image pywavelets arithmetic-compressor pandas matplotlib glymur
```

---

## Run the custom JPEG pipeline

```bash
python3 conversion.py path/to/image.png --quality 50 --method average -qmethod standard --out reconstructed.png
```

**Arguments:**

- `--quality` -> scales quantization tables (1-95)
- `--method` -> chroma downsampling (`nearest` or `average` or `444`)
- `--qmethod` -> quantization table (`standard` or `large_flat` or `small_flat`)
- `--out` -> name of reconstructed image output

### **Outputs:**

- A reconstructed RGB image
- Optional console printout of the **first Y block** at each compression stage (for debugging)

## Compression Evalutation

Runs batches of compression tests using the custom JPEG pipeline compressionPipeline.py file and saves results to a CSV.

```bash
python3 compressionEvaluator.py
```

## Run the custom JPEG2000 pipeline

```bash
python3 jpeg2000pipeline.py image.tif --outdir output_folder --dwt_levels 1 3 5 --block_sizes 16 32 64 --qualities 20 50 80 --pivots 1 5 10
```

**Arguments:**

- `--outdir` -> Output directory for analysis results
- `--dwt_levels` -> List of DWT levels (0-11; default 1)
- `--block_sizes` -> List of block sizes in partitioning (default 64x64)
- `--qualities` -> List of quality levels for quantization (default 75)
- `--pivots` -> TBA (default 1)

### **Outputs:**

- Plots comparing metrics across tweaks in portions of pipeline

---

## Run the built-in codec converter+ metrics

```bash
python3 conversion.py images/input1.bmp images/input2.tif --outdir output_folder --quality 20
```

Prints a table summarizing compression statistics (bits, MSE, PSNR, ratio, and savings).

---

## Notes & Limits

---

## Repository Sketch

TBD

---

## License

To be added

---
