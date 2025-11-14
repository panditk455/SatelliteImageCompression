# plots.py
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Used Matplotlib, pandas and numpy documentation as well as  Gemini to figure out how to generate the plots as well as to edit it and debug it !

def plot_rd_curves(csv_path, out_dir="figs"):
    """
    Plot PSNR–BPP and SSIM–BPP rate–distortion curves averaged across images.
    """
    # Load the CSV file containing per-image metrics (PSNR, SSIM, BPP) and
    # compute average values grouped by subsampling method and quality level.
    # The function then generates two RD curves:
    #   1) PSNR vs BPP  — shows distortion relative to bitrate
    #   2) SSIM vs BPP  — perceptual quality vs bitrate
    # Each subsampling method (444, nearest, average) is plotted separately,
    # enabling a direct comparison of compression performance across methods.
    
    out = Path(out_dir)
    out.mkdir(exist_ok = True, parents = True)

    df = pd.read_csv(csv_path)

    g = df.groupby( ["type", "subsampling", "quality_setting"],  as_index = False
    ).agg(
        mean_psnr=("psnr_db", "mean"),
        mean_ssim=("ssim", "mean"),
        mean_bpp=("bpp", "mean"),
    )

    # PSNR vs BPP 
    plt.figure()
    for subs in ("444", "nearest", "average"):
        sub = g[ (g["type"] == "JPEG") & (g["subsampling"] == subs) ].sort_values("mean_bpp")

        if not sub.empty:
            plt.plot(sub["mean_bpp"], sub["mean_psnr"],  marker = "o", label = subs)

    plt.xlabel("Bits per pixel (BPP)")
    plt.ylabel("PSNR (dB)")
    plt.title("RD Curve: PSNR vs BPP (averaged across images)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha = 0.4)
    plt.tight_layout()
    plt.savefig(out / "rd_psnr_vs_bpp.png", dpi = 200)
    plt.close()

    # SSIM vs BPP 
    plt.figure()
    for subs in ("444", "nearest", "average"):
        sub = g[  (g["type"] == "JPEG") &  (g["subsampling"] == subs) ].sort_values("mean_bpp")

        if not sub.empty:
            plt.plot(sub["mean_bpp"], sub["mean_ssim"], marker = "o", label = subs)

    plt.xlabel("Bits per pixel (BPP)")
    plt.ylabel("SSIM")
    plt.title("RD Curve: SSIM vs BPP (averaged across images)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha = 0.4)
    plt.tight_layout()
    plt.savefig(out / "rd_ssim_vs_bpp.png", dpi = 200)
    plt.close()


def load_stats(stats_json_glob):
    """
    Load per-channel stats JSON files produced during compression evaluation.
    """
    # Search for all JSON files matching the pattern and extract the
    # statistics recorded during encoding: channel type (Y/Cb/Cr), chroma
    # subsampling method, quality setting, coefficient occupancy matrix,
    # total number of 8×8 blocks, and the zero-run histogram. These are
    # returned in a tuple format used by all plotting functions.
    
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


def _agg_occupancy(recs, quality, channels = ("Y", "Cb", "Cr"), methods = ("444", "nearest", "average")):
    """
    Compute averaged 8×8 coefficient occupancy (P(non-zero)) per channel/method.
    """
    # For each channel and subsampling method, this function aggregates all
    # occupancy matrices corresponding to the requested quality level. It sums
    # the non-zero counts across all images and normalizes by the total number
    # of 8×8 blocks to produce an average probability heatmap for P(non-zero).
    out = {}
    for ch in channels:
        for m in methods:
            occ_sum = np.zeros((8, 8), dtype=np.float64)
            n_sum = 0.0
            for r_ch, r_m, r_q, occ, n, _ in recs:
                if r_ch == ch and r_m == m and r_q == quality:
                    occ_sum += occ
                    n_sum += n
            if n_sum > 0:
                out[(ch, m)] = occ_sum / n_sum
    return out


def _agg_zerorun(recs,
                 quality,
                 max_run = 63,
                 channels = ("Y", "Cb", "Cr"),
                 methods = ("444", "nearest", "average")):
    """
    Aggregate zero-run histograms for each channel and subsampling method.
    """
    # Collect all zero-run histograms for the specified channel/method/quality.
    # The function sums zero-run counts (run_length → count) across all records
    # to produce a global histogram that characterizes how often each zero-run
    # length occurs across the dataset.
    
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


#  COEFFICIENT OCCUPANCY Heatmap:

def compare_occupancy_for_channel(stats_json_glob,
                                  channel = "Cb",
                                  quality = 50,
                                  methods = ("444", "nearest", "average"),
                                  out_dir = "figs"):
    """
    Plot occupancy heatmaps for a single channel across subsampling methods.
    """
    # Load all stats JSON files and compute the average P(non-zero) matrix for
    # the chosen channel. For each subsampling method, an 8×8 heatmap is drawn
    # showing how often each DCT coefficient survives quantization. This allows
    # direct comparison of frequency-domain energy retention across pipelines.
    
    outp = Path(out_dir)
    outp.mkdir(parents = True, exist_ok = True)

    recs = load_stats(stats_json_glob)
    pnz = _agg_occupancy(recs, quality, channels = (channel,), methods = methods)

    if not pnz:
        print(f"No occupancy for channel = {channel}, Q = {quality}")
        return

    vmin, vmax = 0.0, 1.0
    fig, axes = plt.subplots(1, len(methods),
                             figsize = (12, 4),
                             constrained_layout = True)
    if len(methods) == 1:
        axes = [axes]

    last_im = None
    for ax, m in zip(axes, methods):
        mat = pnz.get((channel, m))
        if mat is None:
            ax.axis("off")
            continue
        im = ax.imshow(mat, origin = "upper", vmin = vmin, vmax=vmax)
        last_im = im
        ax.set_title(f"{channel} — {m}")
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax = axes,
                            fraction = 0.046, pad = 0.04)
        cbar.set_label("P(non-zero)")

    fig.suptitle(f"Coefficient Occupancy — {channel}, Q = {quality}")
    fn = outp / f"compare_occupancy_{channel}_Q{quality}.png"
    plt.savefig(fn, dpi=200)
    plt.close(fig)
    print(f"Saved {fn}")


def occupancy_grid_all_channels(stats_json_glob,
                                quality = 50,
                                channels=("Y", "Cb", "Cr"),
                                methods=("444", "nearest", "average"),
                                out_dir="figs"):
    """
    Plot full grid of occupancy heatmaps for all channels and methods.
    """
    # Construct a grid of heatmaps where rows correspond to channels (Y, Cb, Cr)
    # and columns correspond to subsampling methods. Each cell visualizes the
    # 8×8 P(non-zero) pattern for that particular channel/method combination,
    # enabling a broad comparison of frequency preservation across the system.
    
    outp = Path(out_dir)
    outp.mkdir(parents = True, exist_ok = True)

    recs = load_stats(stats_json_glob)
    pnz = _agg_occupancy(recs, quality, channels = channels, methods = methods)

    if not pnz:
        print(f"No occupancy for Q = {quality}")
        return

    vmin, vmax = 0.0, 1.0
    fig, axes = plt.subplots(len(channels), len(methods),
                             figsize = (12, 10),
                             constrained_layout = True)
    last_im = None

    for i, ch in enumerate(channels):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            mat = pnz.get((ch, m))
            if mat is None:
                ax.axis("off")
                continue
            im = ax.imshow(mat, origin="upper", vmin = vmin, vmax = vmax)
            last_im = im
            if i == 0:
                ax.set_title(m)
            if j == 0:
                ax.set_ylabel(ch)
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))

    if last_im is not None:
        cbar = fig.colorbar(last_im,
                            ax = axes.ravel().tolist(),
                            fraction = 0.02, pad = 0.02)
        cbar.set_label("P(non-zero)")

    fig.suptitle(f"Coefficient Occupancy — All Channels × Methods, Q = {quality}")
    fn = outp / f"occupancy_grid_all_Q{quality}.png"
    plt.savefig(fn, dpi = 200)
    plt.close(fig)
    print(f"Saved {fn}")


#  ZERO-RUN DISTRIBUTION VISUALS

def compare_zerorun_for_channel(stats_json_glob,
                                channel="Cb",
                                quality = 50,
                                methods = ("444", "nearest", "average"),
                                max_run = 63,
                                normalize = True,
                                out_dir = "figs"):
    """
    Plot zero-run histograms for one channel across subsampling methods.
    """
    # Aggregate zero-run histograms for the selected channel and quality level.
    # The function then produces one histogram per subsampling method, showing
    # the distribution of consecutive zero lengths in the AC coefficient stream.
    # This highlights how each chroma method affects compressibility.
    
    outp = Path(out_dir)
    outp.mkdir(parents = True, exist_ok = True)

    recs = load_stats(stats_json_glob)
    agg = _agg_zerorun(recs, quality, max_run = max_run, channels = (channel,), methods = methods)

    if not agg:
        print(f"No zero-run data for {channel}, Q = {quality}")
        return

    fig, axes = plt.subplots(1, len(methods),
                             figsize=(12, 4),
                             sharey=True,
                             constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]

    runs = np.arange(max_run + 1)

    for ax, m in zip(axes, methods):
        counts = agg.get((channel, m))
        if counts is None or counts.sum() == 0:
            ax.axis("off")
            continue
        
        data = counts / counts.sum() if normalize else counts
        ax.bar(runs, data, width=0.9)
        ax.set_title(f"{channel} — {m}")
        ax.set_xlabel("Zero-run length")
        ax.grid(True, axis = "y", linestyle = "--", alpha = 0.35)

    axes[0].set_ylabel("Proportion" if normalize else "Count")
    fig.suptitle(f"Zero-run Distribution — {channel}, Q = {quality}")
    fn = outp / f"compare_zerorun_{channel}_Q{quality}.png"
    plt.savefig(fn, dpi=200)
    plt.close(fig)
    print(f"Saved {fn}")
    

def zerorun_grid_all_channels(stats_json_glob,
                              quality = 50,
                              channels=("Y", "Cb", "Cr"),
                              methods=("444", "nearest", "average"),
                              max_run = 63,
                              normalize = True,
                              out_dir = "figs"):
    """
    Plot zero-run histogram grid for all channels and methods.
    """
    # Builds a panel of histograms where rows represent Y/Cb/Cr channels and
    # columns represent subsampling methods. Each histogram summarizes how
    # frequently each zero-run length appears in that configuration, making it
    # easy to compare AC coefficient sparsity patterns across the pipeline.
    
    outp = Path(out_dir)
    outp.mkdir(parents = True, exist_ok = True)

    recs = load_stats(stats_json_glob)
    agg = _agg_zerorun(recs, quality, max_run = max_run, channels = channels,  methods = methods)

    if not agg:
        print(f"No zero-run data for Q = {quality}")
        return

    fig, axes = plt.subplots(len(channels), len(methods), figsize = (12, 10), sharey = "row", constrained_layout = True)

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
                ax.set_ylabel(f"{ch}\n" + ("Proportion" if normalize else "Count"))
            ax.set_xlabel("Zero-run length")
            ax.grid(True, axis = "y", linestyle = "--", alpha = 0.35)

    fig.suptitle(f"Zero-run Distribution — All Channels × Methods, Q = {quality}")
    fn = outp / f"zerorun_grid_all_Q{quality}.png"
    plt.savefig(fn, dpi = 200)
    plt.close(fig)
    print(f"Saved {fn}")
    
def zerorun_mean_vs_quality(stats_json_glob,
                            channels=("Y", "Cb", "Cr"),
                            methods=("444", "nearest", "average"),
                            max_run = 63,
                            out_dir = "figs"):
    """
    Plot mean zero-run length as a function of JPEG quality.
    """
    # For each channel and subsampling method, this function computes the
    # average zero-run length at every available JPEG quality setting. The
    # resulting curves show how quantization strength impacts AC sparsity and
    # how each subsampling method scales with quality.
    
    outp = Path(out_dir)
    outp.mkdir(parents = True, exist_ok = True)

    recs = load_stats(stats_json_glob)
    if not recs:
        print("No stats JSONs found for zerorun_mean_vs_quality")
        return

    qualities = sorted({q for (_, _, q, _, _, _) in recs if q >= 0})
    runs = np.arange(max_run + 1)

    for ch in channels:
        plt.figure(figsize=(6, 4))
        any_plotted = False

        for m in methods:
            means = []
            for Q in qualities:
                counts = np.zeros(max_run + 1, dtype = np.int64)
                for r_ch, r_m, r_q, _, __, zr in recs:
                    if r_ch == ch and r_m == m and r_q == Q:
                        for k, v in zr.items():
                            if 0 <= int(k) <= max_run:
                                counts[int(k)] += int(v)
                total = counts.sum()
                if total == 0:
                    means.append(np.nan)
                else:
                    means.append((runs * counts).sum() / float(total))

            if not all(np.isnan(means)):
                plt.plot(qualities, means, marker = "o", label = m)
                any_plotted = True

        if not any_plotted:
            plt.close()
            print(f"No zero-run data to plot for channel {ch}")
            continue

        plt.xlabel("JPEG Quality")
        plt.ylabel("Mean zero-run length")
        plt.title(f"Mean zero-run length vs Quality — {ch}")
        plt.grid(True, linestyle = "--", alpha = 0.35)
        plt.legend(title = "Subsampling")
        plt.tight_layout()

        fn = outp / f"zerorun_mean_vs_quality_{ch}.png"
        plt.savefig(fn, dpi=200)
        plt.close()
        print(f"Saved {fn}")
        


def zerorun_cdf_for_channel(stats_json_glob, channel = "Cb", quality = 50,
                            methods = ("444", "nearest", "average"),
                            max_run = 63,
                            out_dir = "figs"):
    """
    Plot CDF of zero-run lengths for a single channel and quality.
    """
    # For each subsampling method, the function constructs a cumulative
    # distribution function (CDF) describing the proportion of zero-runs with
    # length <= k. The CDF curves provide a smooth comparison of sparsity
    # characteristics and show which pipelines produce longer zero sequences.
    
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok = True)

    recs = load_stats(stats_json_glob)
    if not recs:
        print("No stats JSONs found for zerorun_cdf_for_channel")
        return

    runs = np.arange(max_run + 1)
    plt.figure(figsize =(6, 4))
    any_plotted = False

    for m in methods:
        counts = np.zeros(max_run + 1, dtype=np.int64)
        for r_ch, r_m, r_q, _, __, zr in recs:
            if r_ch == channel and r_m == m and r_q == quality:
                for k, v in zr.items():
                    if 0 <= int(k) <= max_run:
                        counts[int(k)] += int(v)
        total = counts.sum()
        if total == 0:
            continue
        cdf = np.cumsum(counts) / float(total)
        plt.plot(runs, cdf, label = m)
        any_plotted = True

    if not any_plotted:
        plt.close()
        print(f"No zero-run data for channel = {channel}, Q = {quality}")
        return

    plt.xlabel("Zero-run length")
    plt.ylabel("Cumulative proportion")
    plt.title(f"Zero-run CDF — {channel}, Q = {quality}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Subsampling")
    plt.tight_layout()

    fn = outp / f"zerorun_cdf_{channel}_Q{quality}.png"
    plt.savefig(fn, dpi = 200)
    plt.close()
    print(f"Saved {fn}")



if __name__ == "__main__":
    pass


# python3 -c "import plots; plots.plot_rd_curves('compression_results/compression_results_YYYYMMDD_HHMMSS.csv')"
# python3 -c "import plots; plots.occupancy_grid_all_channels('compression_results/stats/*_stats_*.json', quality=50)"
# python3 -c "import plots; plots.zerorun_grid_all_channels('compression_results/stats/*_stats_*.json', quality=50)"
# python3 -c "import plots; plots.zerorun_mean_vs_quality('compression_results/stats/*_stats_*.json')"
# python3 -c "import plots; plots.zerorun_cdf_for_channel('compression_results/stats/*_stats_*.json', channel='Cb', quality=50)"
