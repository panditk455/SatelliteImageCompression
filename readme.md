# JPEG & JPEG‑2000 (Work‑in‑Progress) 

Authors: **Kritika Pandit**, **Anika Rajbhandari**, **Leon Liang**, and **Lucas Schattenmann**

This repository contains a **library implementation** and  **manual implementation** of both **JPEG** and **JPEG‑2000** pipelines and **scripts for metrics** to analyze compression performance on **satellite imagery**.

---

## What’s in the repo?


### **Conversion & Metrics**

The first script provides conversion to **baseline JPEG** and **JPEG‑2000 (JP2)** formats. It evaluates compression using the following metrics:

* **Mean Squared Error (MSE)**
* **Peak Signal‑to‑Noise Ratio (PSNR)**
* **Compression ratio**
* **Space savings**

---


### **JPEG Pipeline**

Implements a full step‑by‑step JPEG encoder and decoder from scratch. It compresses images into the JPEG domain and reconstructs them to verify loss and quality.

**Encoding:**

* **RGB → YCbCr** color conversion
* **4:2:0 Chroma Downsampling** (nearest or average)
* **8×8 DCT‑II (ortho)** transformation
* **Quantization** using scaled luminance/chrominance tables
* **Zig‑zag ordering -> Run‑Length Encoding (RLE) -> Huffman encoding**

**Decoding:**

* **Inverse Huffman -> Inverse RLE ->  Inverse Zig‑zag**
* **Dequantization -> Inverse DCT -> YCbCr -> RGB reconstruction**

---


### **JPEG‑2000 Pipeline**

Implements a full step‑by‑step JPEG‑2000 (JP2) encoder and decoder,  It uses discrete wavelet transforms (DWT) instead of block‑based DCT.

**Encoding:**
**RGB** ->  **YCbCr color conversion**
**Discrete Wavelet Transform (DWT)**
**Quantization of sub‑bands**
**Bitplane coding via Embedded Block Coding with Optimized Truncation (EBCOT)**
**Entropy coding (arithmetic coding) producing final bitstream**

**Decoding:**

**Inverse entropy decoding** -> **Inverse bitplane reconstruction**
**Inverse quantization** -> **Inverse DWT**
**Inverse YCbCr** ->  **RGB reconstruction**

---

## Install

*Work in progress:* will be finalized once all files are merged to the main branch.

Set up for local testing:

```bash
python3 -m venv .venv && source .venv/bin/activate 
pip install numpy pillow scipy imagecodecs
```

---

## Run the custom JPEG pipeline

```bash
python conversion.py path/to/image.png --quality 50 --method average --out reconstructed.png
```

**Arguments:**

* `--quality` -> scales quantization tables (1-95)
* `--method` -> chroma downsampling (`nearest` or `average`)
* `--out` -> name of reconstructed image output

### **Outputs:**

* A reconstructed RGB image
* Optional console printout of the **first Y block** at each compression stage (for debugging)

---

## Run the converter + metrics

```bash
python conversion.py images/input1.bmp images/input2.tif --outdir output_folder --quality 20
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
