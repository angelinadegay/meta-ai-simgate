#!/usr/bin/env python3
"""Write a FAKE HiRISE-style DTM GeoTIFF for testing the converter offline.

It mimics the real files' format: Mars equirectangular CRS, ~1.02 m pixels,
heights around -2,550 m, and a "no data" border of -3.4e38. The terrain itself
(rolling plain, a crater, a ridge) is invented. Never present it as real Mars.
"""
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

MARS_EQC = ('PROJCS["Mars (2015) - Sphere / Ocentric / Equirectangular, clon = 0",GEOGCS["Mars (2015) - Sphere / Ocentric",'
            'DATUM["Mars (2015) - Sphere",SPHEROID["Mars (2015) - Sphere",3396190,0]],PRIMEM["Reference Meridian",0],'
            'UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
            'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]')
NODATA = -3.4028226550889045e38


def main(path: str) -> None:
    h, w, mpp = 600, 300, 1.0165533
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:h, 0:w] * mpp
    z = -2550 + 4 * np.sin(x / 37) + 3 * np.cos(y / 53)                          # rolling plain
    d = np.hypot(x - 150, y - 420) / 45
    z += 9 * np.exp(-((d - 1) ** 2) / 0.04) - 14 * np.exp(-(d**2) / 0.5)        # crater
    z += 18 * np.exp(-((x - 0.5 * y - 20) ** 2) / 800)                           # ridge
    z += rng.normal(0, 0.05, z.shape)                                              # sensor-like noise
    z[:, :12] = NODATA                                                            # ragged no-data edges, like real strips
    z[:, -20:] = NODATA
    z[:8, :] = NODATA
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs=MARS_EQC, transform=from_origin(4586874.8, 1099451.6, mpp, mpp), nodata=NODATA) as ds:
        ds.write(z.astype(np.float32), 1)
    print(f"Wrote fake DTM {path} ({h}x{w} px)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fake_hirise_dtm.tif")
