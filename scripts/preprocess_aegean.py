from __future__ import annotations
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import rasterio

GRAPH = Path(__file__).resolve().parents[1] / "configs" / "s1_grd_preprocess.xml"
NODATA = -32768.0

def run_gpt(gpt: str, inp: Path, out_tif: Path) -> None:
    cmd = [gpt, str(GRAPH), f"-Pinput={inp}", f"-Poutput={out_tif}", "-q", "4"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def split_bands(stack: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(stack) as src:
        names = [d or f"band{i + 1}" for i, d in enumerate(src.descriptions)]
        prof = src.profile.copy()
        prof.update(count=1, dtype="float32", nodata=NODATA, compress="deflate", tiled=True,blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")

        for pol in ("VV", "VH"):
            idx = next((i for i, n in enumerate(names) if pol in n.upper()), None)
            assert idx is not None, f"{pol} band not found in {names}"

            with rasterio.open(out_dir / f"{pol}_dB.tif", "w", **prof) as dst:
                for _, win in src.block_windows(1):
                    a = src.read(idx + 1, window=win).astype(np.float32)
                    bad = ~np.isfinite(a) | (a <= -100)

                    if src.nodata is not None:
                        bad |= a == src.nodata

                    a[bad] = NODATA
                    dst.write(a, 1, window=win)
            print(f"wrote {out_dir / f'{pol}_dB.tif'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpt", default="gpt")
    ap.add_argument("--keep-tmp", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="s1pre_"))

    try:
        stack = tmp / "stack.tif"
        run_gpt(a.gpt, Path(a.input), stack)
        split_bands(stack, Path(a.out))
    finally:
        if not a.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()