# Author: Leon Liang
# Date: Thursday, Oct 9 - Oct 15, 2025

import numpy as np
from typing import Dict, List, Tuple

# a: minimum, b: maximum, t: quality
def linear_interpolation(a, b, t):
    return a + (b - a) * t / 100

def calc_params(num_levels: int, quality: float) -> Tuple[Dict[Tuple[int, str], float], float, Dict[str, float]]:
    """
    Return:
      deltas: {(level, band) -> delta} for bands in {"LH","HL","HH"}
      delta_LL: delta for LL
      deadzones: {"LL": dz_LL, "other": dz_ow}
    quality in [0,100]: 100 = best quality, 0 = lowest quality
    """
    quality = float(np.clip(quality, 0.0, 100.0))

    # Delta
    global_delta = linear_interpolation(1.8, 0.6, quality)   

    base_delta = 1.0 * global_delta
    delta_LL = 0.7 * global_delta

    first_delta = linear_interpolation(1.8, 1.2, quality)   # low Q -> stronger growth, high Q -> gentler

    deltas: Dict[Tuple[int, str], float] = {}
    for level in range(1, num_levels + 1):
        # level 1 (finest) should get biggest factor
        t = (num_levels - level) / max(1, (num_levels - 1)) * 100 # t = 0 at finest, 100 at coarsest
        level_factor = linear_interpolation(first_delta, 1.0, t)

        for b in ("LH", "HL", "HH"):
            deltas[(level, b)] = base_delta * level_factor
            if b == "HH":
                deltas[(level, b)] = deltas[(level, b)] * 1.3

    # Deadzone
    dz_ow = linear_interpolation(1.8, 1.2, quality)  # low quality: larger deadzone; high quality: smaller
    dz_LL = linear_interpolation(1.3, 1.0, quality)  

    deadzones = {"LL": float(dz_LL), "other": float(dz_ow)}
    return deltas, float(delta_LL), deadzones

# Quantizing a coefficient array (x)
def quantize(x: np.ndarray, delta: float, deadzone: float = 1.5) -> np.ndarray:
    if delta <= 0:
        raise ValueError("delta must be > 0")
    a = np.abs(x)
    q = np.where(a >= deadzone * delta,
                 np.floor((a - (deadzone - 1) * delta) / delta),
                 0.0)
    return (np.sign(x) * q).astype(np.int32)

# Quantizing single component
def quantize_component(comp: Dict, quality: float) -> Dict:
    num_levels = len(comp["levels"])
    deltas, delta_LL, deadzones = calc_params(num_levels = num_levels, quality = quality)

    dz_ow = deadzones.get("other", deadzones.get("other", 1.5))
    dz_LL = deadzones.get("LL", dz_ow)

    out_levels = []
    for level, bands in enumerate(comp["levels"], start = 1):
        out_levels.append({
            "LH": quantize(bands["LH"], deltas.get((level, "LH"), 1.0), dz_ow),
            "HL": quantize(bands["HL"], deltas.get((level, "HL"), 1.0), dz_ow),
            "HH": quantize(bands["HH"], deltas.get((level, "HH"), 1.0), dz_ow),
        })
    out_LL = quantize(comp["LL"], delta_LL, dz_LL)
    return {"levels": out_levels, "LL": out_LL}


# Quantizing all coefficients
def quantize_all(dwt_result: List[Dict], quality: float) -> List[Dict]:
    return [quantize_component(c, quality) for c in dwt_result]

# Testing if the deltas and deadzones work correctly
def test_params(num_levels, quality):

    deltas, delta_LL, deadzones = calc_params(num_levels=num_levels, quality=quality)

    print(f"\n=== Quantization params ===")
    print(f"levels: {num_levels} | quality: {quality}")
    print(f"deadzone_detail (for LH/HL/HH): {deadzones['other']:.3f}")
    print(f"deadzone_LL (for LL):           {deadzones['LL']:.3f}\n")

    # table header
    print("Per-level deltas (Δ) and deadzones")
    print("Level |    Δ(LH)    Δ(HL)    Δ(HH)  |  deadzone (detail)")
    print("------+--------------------------------+-------------------")

    for level in range(1, num_levels + 1):
        lh = deltas[(level, "LH")]
        hl = deltas[(level, "HL")]
        hh = deltas[(level, "HH")]
        print(f"{level:>5} |  {lh:8.4f}  {hl:8.4f}  {hh:8.4f}  |      {deadzones['other']:.3f}")

    # LL row
    print("\nFinal LL band")
    print("----------------------------")
    print(f"Δ(LL): {delta_LL:.4f}")
    print(f"deadzone(LL): {deadzones['LL']:.3f}")

    # also print as a simple nested dict for easy copy/paste
    nested = {
        "deltas": {
            level: {
                "LH": float(deltas[(level, "LH")]),
                "HL": float(deltas[(level, "HL")]),
                "HH": float(deltas[(level, "HH")]),
            }
            for level in range(1, num_levels + 1)
        },
        "LL": {"delta": float(delta_LL)},
        "deadzones": {"other": float(deadzones["other"]), "LL": float(deadzones["LL"])},
    }
    print("\nAs dict:")
    print(nested)

def main():
    quality = 75
    # # Testing if the deltas and deadzones work correctly
    # # Works fine looking at the results
    # test_params(3, quality)

    # Waiting for the early two parts to test teh quantozation part




if __name__ == "__main__":
    main()


