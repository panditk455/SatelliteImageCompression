# plotting.py


"""
Plotting the JPEG Experiments:

- Rate–distortion (RD) curves from the CSV produced by CompressionEvaluator
- Coefficient occupancy heatmaps from per-channel stats JSONs
- Zero-run length histograms and mean zero-run length vs. quality

These functions generate the figures used in the JPEG sections of the paper.
Used Gemini to standardize the plot sizing and improve axis and legend labeling for better visualization quality.
"""

import json
import glob
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# RATE–DISTORTION CURVES
def plot_rd_curves(csv_path: str, out_dir: str = "figs") -> None:
    """
    Plot PSNR-BPP and SSIM-BPP RD curves averaged across images,
    grouped by chroma_method.

    Outputs:
      - rd_psnr_vs_bpp.png
      - rd_ssim_vs_bpp.png
    """

    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(csv_path)
    
    if "chroma_method" not in df.columns:
        raise ValueError("CSV must contain column 'chroma_method'.")
    chroma_col = "chroma_method"

    # Aggregate across images
    g = ( df.groupby(["type", chroma_col, "quality_setting"], as_index = False).agg(
              mean_psnr=("psnr_db", "mean"),
              mean_ssim=("ssim", "mean"),
              mean_bpp=("bpp", "mean"),))

    # PSNR vs BPP
    plt.figure()
    for chroma in ("444", "nearest", "average"):
        sub = g[
            (g["type"] == "JPEG") &
            (g[chroma_col] == chroma)
        ].sort_values("mean_bpp")

        if not sub.empty:
            plt.plot(
                sub["mean_bpp"],
                sub["mean_psnr"],
                marker="o",
                label = chroma,
            )

    plt.xlabel("Bits per pixel (BPP)")
    plt.ylabel("PSNR (dB)")
    plt.title("RD Curve: PSNR vs BPP (averaged across images)")
    plt.legend(title = "Chroma method")
    plt.grid(True, linestyle = "--")
    plt.tight_layout()
    plt.savefig(out / "rd_psnr_vs_bpp.png", dpi = 200)
    plt.close()

    # SSIM vs BPP
    plt.figure()
    for chroma in ("444", "nearest", "average"):
        sub = g[(g["type"] == "JPEG") &
            (g[chroma_col] == chroma)].sort_values("mean_bpp")

        if not sub.empty:
            plt.plot(
                sub["mean_bpp"],
                sub["mean_ssim"],
                marker="o",
                label=chroma,
            )

    plt.xlabel("Bits per pixel (BPP)")
    plt.ylabel("SSIM")
    plt.title("RD Curve: SSIM vs BPP (averaged across images)")
    plt.legend(title = "Chroma method")
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plt.savefig(out / "rd_ssim_vs_bpp.png", dpi=200)
    plt.close()


# STATS JSON HELPERS to make the plots.

def load_stats(stats_json_glob):
    """
    Load per-channel stats JSONs produced by CompressionEvaluator.

    Fields:
      - channel
      - chroma_method
      - quality
      - occupancy_counts  (8x8)
      - num_blocks
      - zero_run_hist     {run_length: count}

    Returns:
      list of (channel, method, quality, occupancy_matrix, num_blocks, zero_run_hist_dict)
    """
    paths = glob.glob(stats_json_glob)
    if not paths:
        return []

    recs = []
    for p in paths:
        with open(p, "r") as f:
            d = json.load(f)
        ch = d.get("channel")
        subs = d.get("chroma_method")
        q = int(d.get("quality", -1))

        occ = np.array(d["occupancy_counts"], dtype = np.float64)
        n = float(d["num_blocks"])

        zr_raw = d.get("zero_run_hist", {})
        zr = {int(k): int(v) for k, v in zr_raw.items()}

        recs.append((ch, subs, q, occ, n, zr))
    return recs


def _agg_occupancy(recs, quality, channels = ("Y", "Cb", "Cr"), methods = ("444", "nearest", "average"),):
    """
    For a given quality, compute the average occupancy matrix P(non-zero)
    per (channel, method).
    """
    out = {}
    for ch in channels:
        for m in methods:
            occ_sum = np.zeros((8, 8), dtype = np.float64)
            n_sum = 0.0
            for r_ch, r_m, r_q, occ, n, _ in recs:
                if r_ch == ch and r_m == m and r_q == quality:
                    occ_sum += occ
                    n_sum += n
            if n_sum > 0:
                out[(ch, m)] = occ_sum / n_sum
    return out


def _agg_zerorun(recs, quality, max_run = 63, channels = ("Y", "Cb", "Cr"), methods = ("444", "nearest", "average"),):
    """
    For a given quality, aggregate zero-run histograms per (channel, method).
    """
    out = {}
    for ch in channels:
        for m in methods:
            counts = np.zeros(max_run + 1, dtype = np.int64)
            for r_ch, r_m, r_q, _, __, zr in recs:
                if r_ch == ch and r_m == m and r_q == quality:
                    for k, v in zr.items():
                        if 0 <= k <= max_run:
                            counts[k] += v
            if counts.sum() > 0:
                out[(ch, m)] = counts
    return out


# COEFFICIENT OCCUPANCY HEATMAPS for differnt chroma methods and differnt Channels:
def occupancy_grid_all_channels( stats_json_glob, quality = 50, channels = ("Y", "Cb", "Cr"),
                                methods = ("444", "nearest", "average"), out_dir = "figs",):
    """
    Heatmap grid: channels x methods of P(non-zero) at a given quality.

    Outputs (for different Q values):
      - occupancy_grid_all_Q10.png
      - occupancy_grid_all_Q95.png
    """
    outp = Path(out_dir)
    outp.mkdir(parents = True, exist_ok = True)

    recs = load_stats(stats_json_glob)
    pnz = _agg_occupancy(recs, quality, channels = channels, methods = methods)

    if not pnz:
        print(f"No occupancy for Q = {quality}")
        return

    vmin, vmax = 0.0, 1.0
    fig, axes = plt.subplots(
        len(channels),
        len(methods),
        figsize = (12, 10),
        constrained_layout = True,)
    last_im = None

    for i, ch in enumerate(channels):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            mat = pnz.get((ch, m))
            if mat is None:
                ax.axis("off")
                continue
            im = ax.imshow(mat, origin = "upper", vmin = vmin, vmax = vmax)
            last_im = im
            if i == 0:
                ax.set_title(m)
            if j == 0:
                ax.set_ylabel(ch)
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))

    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax = axes.ravel().tolist(),
            fraction = 0.02,
            pad = 0.02,)
        
        cbar.set_label("P(non-zero)")

    fig.suptitle(f"Coefficient Occupancy - All Channels x Methods, Q = {quality}")
    fn = outp / f"occupancy_grid_all_Q{quality}.png"
    plt.savefig(fn, dpi = 200)
    plt.close(fig)
    print(f"Saved {fn}")


# ZERO-RUN DISTRIBUTION VISUALS
def zerorun_grid_all_channels(stats_json_glob, quality=50,channels = ("Y", "Cb", "Cr"),
                              methods=("444", "nearest", "average"), max_run = 63,
                              normalize = True, out_dir = "figs",):
    """
    rows = channels, cols = methods, bars = zero-run histograms.
    Outputs (for different Quality (Q) values):
    eg:
      - zerorun_grid_all_Q10.png
      - zerorun_grid_all_Q95.png
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    recs = load_stats(stats_json_glob)
    agg = _agg_zerorun(
        recs,
        quality,
        max_run = max_run,
        channels = channels,
        methods = methods,
    )

    if not agg:
        print(f"No zero-run data for Q = {quality}")
        return

    fig, axes = plt.subplots(
        len(channels),
        len(methods),
        figsize=(12, 10),
        sharey = "row",
        constrained_layout = True,
    )

    runs = np.arange(max_run + 1)

    for i, ch in enumerate(channels):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            counts = agg.get((ch, m))
            if counts is None or counts.sum() == 0:
                ax.axis("off")
                continue
            data = counts / counts.sum() if normalize else counts
            ax.bar(runs, data, width=0.9)
            if i == 0:
                ax.set_title(m)
            if j == 0:
                ax.set_ylabel(
                    f"{ch}\n" + ("Proportion" if normalize else "Count")
                )
            ax.set_xlabel("Zero-run length")
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(f"Zero-run Distribution - All Channels x Methods, Q = {quality}")
    fn = outp / f"zerorun_grid_all_Q{quality}.png"
    plt.savefig(fn, dpi=200)
    plt.close(fig)
    print(f"Saved {fn}")


def zerorun_mean_vs_quality( stats_json_glob, channels=("Y", "Cb", "Cr"), methods = ("444", "nearest", "average"),  # ignored, kept for compatibility accoriding to the Stats
    max_run = 63, out_dir="figs", ):
    """
    For each channel, plot mean zero-run length vs JPEG quality,
    aggregating across ALL chroma methods.

    Output:
      - zerorun_mean_vs_quality_all_channels_vertical.png
    """
    outp = Path(out_dir)
    outp.mkdir(parents = True, exist_ok = True)

    recs = load_stats(stats_json_glob)
    if not recs:
        print("No stats JSONs found for zerorun_mean_vs_quality")
        return

    #  JPEG qualities
    qualities = sorted({q for (_, _, q, _, _, _) in recs if q >= 0})
    runs = np.arange(max_run + 1)

    # Computing the mean zero-run lengths
    means_by_channel = {ch: [] for ch in channels}

    for ch in channels:
        for Q in qualities:
            counts = np.zeros(max_run + 1, dtype = np.int64)
            
            # Aggregating across ALL subsampling methods
            for r_ch, r_m, r_q, _, __, zr in recs:
                if r_ch == ch and r_q == Q:
                    for k, v in zr.items():
                        k = int(k)
                        if 0 <= k <= max_run:
                            counts[k] += int(v)

            total = counts.sum()
            if total == 0:
                means_by_channel[ch].append(np.nan)
            else:
                means_by_channel[ch].append((runs * counts).sum() / float(total))

    # Plot vertically: 3 rows
    fig, axes = plt.subplots(
        len(channels),
        1,
        figsize=(7, 12),
        sharex = True,
        constrained_layout = True,
    )

    if len(channels) == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        means = means_by_channel[ch]
        if all(np.isnan(means)):
            ax.axis("off")
            continue

        ax.plot(qualities, means, marker="o")
        ax.set_title(f"{ch}", fontsize=13)
        ax.set_ylabel("Mean zero-run length")
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("JPEG Quality")

    fig.suptitle(
        "Mean zero-run length vs Quality (averaged across all chroma methods)",
        fontsize = 15,
        y=1.02,
    )

    fn = outp / "zerorun_mean_vs_quality_all_channels_vertical.png"
    plt.savefig(fn, dpi = 200)
    plt.close(fig)
    print(f"Saved {fn}")


if __name__ == "__main__":
    # Calling functions from scripts or notebooks as it would take really long to process all methods, quality levels and other methods
    pass

# Example calls:
# python3 -c "import plotting as p; p.plot_rd_curves('compression_results/compression_results_20251116_221752.csv')"
# python3 -c "import plotting as p; p.occupancy_grid_all_channels('compression_results/stats/*_stats_*.json', quality=10)"
# python3 -c "import plotting as p; p.occupancy_grid_all_channels('compression_results/stats/*_stats_*.json', quality=95)"
# python3 -c "import plotting as p; p.zerorun_grid_all_channels('compression_results/stats/*_stats_*.json', quality=10)"
# python3 -c "import plotting as p; p.zerorun_grid_all_channels('compression_results/stats/*_stats_*.json', quality=95)"
# python3 -c "import plotting as p; p.zerorun_mean_vs_quality('compression_results/stats/*_stats_*.json')"


"""
Plotting functions for visualizing results from compression experiments.
Note: these functions read from specific CSV files generated during the experiments.
Should generalize to be run on a singular results csv in future as needed.
"""

new_dir = 'Plots'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

# Heatmaps for representing spatial data, frequency data, and quantized frequency data ----------------
pixel_block = np.array([
    [52.0, 139.0, 162.0, 116.0, 97.0, 107.0, 107.0, 108.0],
    [89.0, 153.0, 163.0, 127.0, 110.0, 114.0, 118.0, 127.0],
    [139.0, 173.0, 165.0, 142.0, 130.0, 124.0, 120.0, 127.0],
    [179.0, 187.0, 169.0, 145.0, 120.0, 102.0, 109.0, 133.0],
    [191.0, 192.0, 170.0, 123.0, 84.0, 74.0, 99.0, 140.0],
    [190.0, 186.0, 157.0, 95.0, 69.0, 81.0, 105.0, 139.0],
    [189.0, 170.0, 131.0, 82.0, 73.0, 94.0, 113.0, 143.0],
    [179.0, 136.0, 88.0, 59.0, 60.0, 88.0, 132.0, 161.0]
])

dct_coefficients = np.array([
    [-5.8, 132.6, 123.7, -83.2, -38.8, -26.3, 3.4, -1.5],
    [10.4, -36.2, -144.4, -45.6, -38.6, -15.5, -6.4, -2.6],
    [-79.5, -82.8, -1.4, 10.5, -25.3, -1.0, 0.7, 0.8],
    [-26.4, 27.0, 2.7, -32.0, -19.8, -3.9, 7.3, 1.5],
    [-16.2, -2.8, 6.1, -5.5, 10.2, 5.3, -3.6, 0.5],
    [11.3, 9.0, -15.6, 9.7, -1.8, -3.5, 1.4, -1.0],
    [-5.0, -5.0, 0.7, 4.5, -3.6, 1.1, -0.6, 0.7],
    [1.0, 3.6, -2.9, -0.5, -0.5, -1.1, 0.4, -0.7]
])

quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def plot_heatmap(data: np.ndarray, title: str, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Rate-Distortion Curve for Quality Experiments 
def rate_distortion_curve(csv_path: str, save_path: str):
    df = pd.read_csv(csv_path)
    unique_images = df['image_name'].unique()

    plt.figure(figsize=(10, 6))

    for image in unique_images:
        image_df = df[df['image_name'] == image]
        image_df = image_df.sort_values(by='bpp')
        plt.plot(image_df['bpp'], image_df['mse'], label=image, marker='o')

    plt.xlabel('Rate (Bits Per Pixel - bpp)')
    plt.ylabel('Distortion (Mean Squared Error - MSE)')
    plt.title('Rate-Distortion Curve (MSE vs. bpp)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

# Comparing the Quantization methods

def quantization_comparison(csvs: list[str]):
    dfs = [pd.read_csv(csv) for csv in csvs]
    unique_methods = pd.concat(dfs)['quantization_method'].unique()

    plt.figure(figsize=(10, 6))
    counter = 0
    for method in unique_methods:
        method_df = pd.concat(dfs)[pd.concat(dfs)['quantization_method'] == method]

        method_df = method_df.sort_values(by='bpp')
        plt.plot(method_df['bpp'], method_df['psnr_db'], label=method, marker='o')
        plt.xlabel('Rate (Bits Per Pixel - bpp)')
        plt.ylabel('Distortion (Peak Signal-to-Noise Ratio - PSNR)')
        plt.title(f'Rate-Distortion Curve of {method_df["image_name"].iloc[0]} for Different Quantization Methods')

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        image_path = os.path.join(new_dir, f'quantization_distortion_curve{counter}.png')
        plt.savefig(image_path)

        plt.show()
        counter += 1


# BPP vs Spatial Complexity at fixed quality level 75
def bpp_vs_spatial_complexity(csv_path: str):
    df = pd.read_csv(csv_path)
    df_quality_75 = df[df['quality_setting'] == 75]

    plt.figure(figsize=(10, 6))

    x = df_quality_75['edge_density']
    y = df_quality_75['bpp']

    plt.scatter(x, y)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    #add trendline to plot
    plt.plot(x, p(x))

    plt.xlabel('Edge Density')
    plt.ylabel('Rate (Bits Per Pixel - bpp)')
    plt.title('BPP vs. Edge Density (Quality = 75)')

    plt.grid(True)
    plt.tight_layout()

    image_path = os.path.join(new_dir, 'edge_density_vs_bpp.png')
    plt.savefig(image_path)

    plt.show()


# Other plots for the JPEG Experiments:
def main():
    plot_heatmap(pixel_block, 'Spatial Domain Heatmap', os.path.join(new_dir, 'spatial_heatmap.png'))
    plot_heatmap(dct_coefficients, 'Frequency Domain Heatmap (DCT Coefficients)', os.path.join(new_dir, 'frequency_heatmap.png'))
    plot_heatmap(np.round(dct_coefficients / quantization_matrix).astype(int), 'Quantized Heatmap', os.path.join(new_dir, 'quantized_frequency_heatmap.png'))
