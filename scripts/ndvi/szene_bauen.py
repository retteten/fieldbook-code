"""Baut die Schnappschuss-Ebenen einer Layer-Explorer-Szene um einen Punkt.

Warum eigene Anfragen statt nb.process()
----------------------------------------
nb.process() setzt die bbox fest auf CRS84. Der Layer-Explorer rechnet aber in
EPSG:3857 und teilt die bbox linear auf das Pixelraster, damit das Frontend
Klicks ohne Projektionsbibliothek in Koordinaten umrechnen kann. Ein Quadrat in
CRS84 ist in 3857 KEIN Quadrat — die Kacheln waeren minimal verzerrt und der
Klick liefe daneben. Deshalb hier derselbe Weg wie in relief_ersatz.py: eigene
Anfrage mit EPSG:3857.

Erzeugt drei Ebenen, wie die vorhandenen Szenen sie fuehren:
    s2.jpg      Sentinel-2 Echtfarbe (Saison-Median)
    ndvi.png    NDVI derselben Saison, feste Farbskala
    relief.jpg  Copernicus DEM GLO-30, Schummerung

Pflichtvermerke der Ergebnisse:
    "Contains modified Copernicus Sentinel data <Jahr>"
    "Enthaelt modifizierte Copernicus-Daten <Jahr> (Copernicus DEM GLO-30),
     eigene Schummerung."

Aufruf:
    python szene_bauen.py --lat -23.973037 --lon 25.854650 --name botswana \
                          --kante-m 3400 --jahr 2026
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndvi_batch as nb  # noqa: E402
import punktauswertung as pa  # noqa: E402
import punktbilder as pb  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LAYERS = REPO / "ftp-mirror-geophora" / "layer-explorer" / "layers"

TYP_TIFF = "image/tiff"
TYP_PNG = "image/png"
TYP_JPEG = "image/jpeg"
R_ERDE = 6378137.0


def nach_3857(lat: float, lon: float) -> tuple[float, float]:
    x = R_ERDE * math.radians(lon)
    y = R_ERDE * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def bbox_3857(lat: float, lon: float, kante_m: float) -> list[float]:
    """Quadrat in 3857 um den Punkt.

    ACHTUNG: 3857-Meter sind keine Bodenmeter. Der Massstabsfaktor ist
    1/cos(Breite) — auf 24 Grad Sued sind 1000 3857-Meter rund 914 Bodenmeter.
    kante_m meint hier BODENmeter; die Umrechnung passiert unten.
    """
    x, y = nach_3857(lat, lon)
    halb = kante_m / math.cos(math.radians(lat)) / 2.0
    return [x - halb, y - halb, x + halb, y + halb]


def anfrage(token: str, bbox: list[float], groesse: tuple[int, int],
            evalscript: str, typ: str, datenteil: dict) -> bytes:
    breite, hoehe = groesse
    nutzlast = {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"},
            },
            "data": [datenteil],
        },
        "output": {
            "width": breite, "height": hoehe,
            "responses": [{"identifier": "default", "format": {"type": typ}}],
        },
        "evalscript": evalscript,
    }
    antwort = requests.post(
        nb.PROCESS_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": typ},
        json=nutzlast, timeout=300,
    )
    if antwort.status_code != 200:
        raise RuntimeError(f"HTTP {antwort.status_code}: {antwort.text[:300]}")
    return antwort.content


def s2_datenteil(von: str, bis: str) -> dict:
    return {
        "type": "sentinel-2-l2a",
        "dataFilter": {
            "timeRange": {"from": f"{von}T00:00:00Z", "to": f"{bis}T23:59:59Z"},
            "maxCloudCoverage": 75,
            "mosaickingOrder": "leastCC",
        },
    }


EVAL_DEM = """//VERSION=3
function setup() {
  return { input: [{bands: ["DEM"], units: "METERS"}],
           output: {bands: 1, sampleType: "FLOAT32"} };
}
function evaluatePixel(s) { return [s.DEM]; }"""


def schummerung(hoehen: np.ndarray, meter_pro_pixel: float,
                z_faktor: float, staerke: float) -> Image.Image:
    """Hillshade 315 Grad / 45 Grad, Kontrast ueber die Perzentilspanne.

    Wie in relief_ersatz.py: Der Median wandert auf Neutralgrau, damit die
    Ebene sich als soft-light ueberlagern laesst, ohne das Bild abzudunkeln.
    """
    dy, dx = np.gradient(np.nan_to_num(hoehen) * z_faktor, meter_pro_pixel)
    neigung = np.arctan(np.hypot(dx, dy))
    ausrichtung = np.arctan2(-dx, dy)
    az, hoehe_sonne = math.radians(315.0), math.radians(45.0)
    hs = (np.sin(hoehe_sonne) * np.cos(neigung)
          + np.cos(hoehe_sonne) * np.sin(neigung) * np.cos(az - ausrichtung))
    unten, mitte, oben = np.percentile(hs, [2.0, 50.0, 98.0])
    spanne = max(float(oben - unten), 1e-6)
    grau = 0.5 + (hs - float(mitte)) / spanne * staerke
    return Image.fromarray((np.clip(grau, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--kante-m", type=float, default=3400.0,
                    help="Kantenlaenge in BODENmetern")
    ap.add_argument("--jahr", type=int, default=2026)
    ap.add_argument("--px", type=int, default=1000)
    a = ap.parse_args()

    bbox = bbox_3857(a.lat, a.lon, a.kante_m)
    ziel = LAYERS / a.name
    ziel.mkdir(parents=True, exist_ok=True)
    von, bis = pa.saisonfenster(a.jahr)
    lauf = pa.SchlankerLauf()
    token = lauf.token()

    print(f"Szene {a.name}")
    print(f"  Mittelpunkt  {a.lat:.6f}, {a.lon:.6f}")
    print(f"  Kante        {a.kante_m:.0f} m Boden "
          f"({a.kante_m/math.cos(math.radians(a.lat)):.0f} m in EPSG:3857)")
    print(f"  bbox 3857    {','.join(f'{v:.1f}' for v in bbox)}")
    print(f"  Saison       {von} bis {bis}")

    # --- Echtfarbe --------------------------------------------------------
    roh = anfrage(token, bbox, (a.px, a.px), pb.EVAL_ECHT, TYP_JPEG,
                  s2_datenteil(von.isoformat(), bis.isoformat()))
    Image.open(io.BytesIO(roh)).convert("RGB").save(ziel / "s2.jpg", quality=88)
    print(f"  s2.jpg       {(ziel / 's2.jpg').stat().st_size // 1024} KB")

    # --- NDVI -------------------------------------------------------------
    roh = anfrage(token, bbox, (a.px, a.px), nb.evalscript_ndvi_tiff(), TYP_TIFF,
                  s2_datenteil(von.isoformat(), bis.isoformat()))
    ndvi, n = nb.lies_tiff_bytes(roh, mit_n=True)
    Image.fromarray(pb.ndvi_farbe(ndvi)).save(ziel / "ndvi.png")
    print(f"  ndvi.png     {(ziel / 'ndvi.png').stat().st_size // 1024} KB")

    # --- Relief -----------------------------------------------------------
    roh = anfrage(token, bbox, (a.px, a.px), EVAL_DEM, TYP_TIFF,
                  {"type": "dem", "dataFilter": {"demInstance": "COPERNICUS_30"},
                   "processing": {"upsampling": "BILINEAR",
                                  "downsampling": "BILINEAR"}})
    hoehen, _ = nb.lies_tiff_bytes(roh, mit_n=False), None
    if isinstance(hoehen, tuple):
        hoehen = hoehen[0]
    m_pro_px = a.kante_m / a.px
    # Flaches Gelaende: ohne kraeftige Ueberhoehung bliebe die Ebene leer.
    schummerung(hoehen, m_pro_px, z_faktor=8.0, staerke=0.45).save(
        ziel / "relief.jpg", quality=88)
    print(f"  relief.jpg   {(ziel / 'relief.jpg').stat().st_size // 1024} KB")
    print(f"  Hoehen       {np.nanmin(hoehen):.0f} bis {np.nanmax(hoehen):.0f} m")

    print(f"\n  Szenendefinition fuer erkunden.html:")
    print(f"    bbox:'{','.join(f'{v:.1f}' for v in bbox)}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
