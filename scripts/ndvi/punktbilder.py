"""Anschauungsbilder zu einer Punktauswertung: Echtbild, NDVI, Differenzkarte.

Zahlen ohne Bild sind bei Fernerkundung gefaehrlich — ein Median verrraet nicht,
ob die Flaeche gleichmaessig anders ist oder ob ein einzelnes Objekt den Wert
zieht. Deshalb drei Bilder, alle mit demselben Ausschnitt:

  echtbild   was ein Mensch sehen wuerde
  ndvi       wo Vegetation steht
  differenz  jedes Pixel gegen den Median SEINES Rings — macht sichtbar,
             WO die Abweichung sitzt, statt sie zu einer Zahl zu mitteln

Der eingezeichnete Kreis ist reine Orientierung und keine Grenze im Gelaende.

Aufruf:
    python punktbilder.py --lat -23.973037 --lon 25.854650 --name botswana
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndvi_batch as nb  # noqa: E402
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS, R1)
import punktauswertung as pa  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AUSGABE = REPO / "docs" / "daten" / "punktbilder"

TYP_TIFF = "image/tiff"
TYP_PNG = "image/png"

# Echtfarbe mit derselben Wolkenmaske wie die Analyse. mosaickingOrder
# leastCC waehlt je Pixel die wolkenaermste Szene — ohne das bleiben in
# einer einzelnen Saison Loecher.
EVAL_ECHT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B04", "B03", "B02", "dataMask"]}],
    output: {bands: 3, sampleType: "AUTO"},
    mosaicking: "ORBIT"
  };
}
function median(v) {
  if (!v.length) return 0;
  v.sort(function (a, b) { return a - b; });
  var m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}
function evaluatePixel(samples) {
  var r = [], g = [], b = [];
  for (var i = 0; i < samples.length; i++) {
    if (samples[i].dataMask === 1) {
      r.push(samples[i].B04); g.push(samples[i].B03); b.push(samples[i].B02);
    }
  }
  // 2.5 als Streckung ist die uebliche Sentinel-2-Echtfarbdarstellung.
  return [median(r) * 2.5, median(g) * 2.5, median(b) * 2.5];
}"""


def kreis_einzeichnen(bild: Image.Image, radien_m, bbox_utm, x0: float,
                      y0: float):
    """Kreise um den PROJIZIERTEN Punkt, Radius in Metern -> Pixel.

    Seit R1 rechnet das Bild im metrischen Gitter: Nach dem Gitter-Snap liegt
    der Punkt nicht mehr exakt in der Bildmitte (Versatz <= 1 Zelle), also
    wird seine Pixellage aus der projizierten bbox bestimmt statt geraten."""
    d = ImageDraw.Draw(bild, "RGBA")
    w, s, o, n = bbox_utm
    mx = (x0 - w) / (o - w) * bild.width
    my = (n - y0) / (n - s) * bild.height
    for radius_m, farbe, breite in radien_m:
        r = radius_m / (o - w) * bild.width
        d.ellipse([mx - r, my - r, mx + r, my + r],
                  outline=farbe, width=breite)
    return bild


def ndvi_farbe(ndvi: np.ndarray) -> np.ndarray:
    """Braun -> gelb -> gruen. Feste Skala 0.0 bis 0.7, damit Bilder
    verschiedener Jahre vergleichbar bleiben."""
    x = np.clip((np.nan_to_num(ndvi, nan=0.0) - 0.0) / 0.7, 0, 1)
    stuetzen = np.array([[150, 110, 70], [200, 180, 110],
                         [150, 190, 100], [40, 120, 55]], dtype=float)
    pos = x * (len(stuetzen) - 1)
    unten = np.clip(pos.astype(int), 0, len(stuetzen) - 2)
    t = (pos - unten)[..., None]
    rgb = stuetzen[unten] * (1 - t) + stuetzen[unten + 1] * t
    return rgb.astype(np.uint8)


def differenz_farbe(diff: np.ndarray) -> np.ndarray:
    """Bernstein = weniger Gruen als der Ring, Pinie = mehr, Papier = wie ringsum.

    Dieselbe Rampe wie im Feldbuch (kampagne.css: Papier #F6F4EE, Bernstein
    #E8A13C, Gruen #2C8A6B). Wer zwei Folgen nebeneinander legt, soll dieselbe
    Farbe dasselbe bedeuten sehen — eine eigene Skala je Folge waere eine
    Falle, keine Gestaltung.
    """
    x = np.clip(np.nan_to_num(diff, nan=0.0) / 0.25, -1, 1)
    papier = np.array([246, 244, 238], dtype=float)
    bernstein = np.array([232, 161, 60], dtype=float)
    pinie = np.array([44, 138, 107], dtype=float)
    t = np.abs(x)[..., None]
    ziel = np.where(x[..., None] < 0, bernstein, pinie)
    return np.clip(papier * (1 - t) + ziel * t, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--jahr", type=int, default=date.today().year)
    ap.add_argument("--kern-m", type=float, default=500.0)
    ap.add_argument("--ring-von-m", type=float, default=800.0)
    ap.add_argument("--ring-bis-m", type=float, default=1500.0)
    ap.add_argument("--kante", type=int, default=900, help="Bildkante in Pixeln")
    a = ap.parse_args()

    halbkante = a.ring_bis_m * 1.15
    bbox = pa.bbox_um(a.lat, a.lon, halbkante)
    aoi = {"id": a.name, "bbox": bbox, "aufloesung_m": 10.0}
    # Metrisches Analysegitter (R1), 10 m: alle drei Bilder aus demselben
    # UTM-Raster; die Anzeigegroesse entsteht lokal per Nearest-Resize.
    epsg, bbox_utm, breite, hoehe = nb.aoi_gitter(aoi)
    x0, y0 = ch.punkt_nach(epsg, a.lon, a.lat)
    von, bis = pa.saisonfenster(a.jahr)
    lauf = pa.SchlankerLauf()
    AUSGABE.mkdir(parents=True, exist_ok=True)

    kante_h = round(a.kante * hoehe / breite)   # Seitenverhaeltnis des Gitters
    print(f"Punktbilder {a.name}, Saison {von} bis {bis}")
    print(f"  Ausschnitt {2*halbkante/1000:.1f} km, Gitter {breite} x {hoehe} px"
          f" auf 10 m (EPSG:{epsg}), Ausgabe {a.kante} x {kante_h} px")

    # --- Echtbild ---------------------------------------------------------
    # Natives 10-m-Gitter statt CDSE-seitigem Hochrechnen auf Anzeigepixel:
    # das Hochskalieren hat keine Information, die das Gitter nicht haette,
    # und so bleibt die Georeferenz aller drei Bilder identisch.
    roh = nb.process(lauf, aoi, von, bis, EVAL_ECHT, TYP_PNG,
                     aufloesung_m=10.0, mosaik_reihenfolge="leastCC")
    import io
    echt = Image.open(io.BytesIO(roh)).convert("RGB")
    echt = echt.resize((a.kante, kante_h), Image.NEAREST)
    kreis_einzeichnen(echt, [(a.kern_m, (255, 220, 60, 235), 3),
                             (a.ring_von_m, (255, 255, 255, 120), 1),
                             (a.ring_bis_m, (255, 255, 255, 120), 1)],
                      bbox_utm, x0, y0)
    echt.save(AUSGABE / f"{a.name}-echtbild-{a.jahr}.png")
    print(f"  echtbild-{a.jahr}.png")

    # --- NDVI + Differenz -------------------------------------------------
    roh = nb.process(lauf, aoi, von, bis, nb.evalscript_ndvi_tiff(),
                     TYP_TIFF, aufloesung_m=10.0)
    ndvi, n = nb.lies_tiff_bytes(roh, mit_n=True)
    bild = Image.fromarray(ndvi_farbe(ndvi))
    bild = bild.resize((a.kante, kante_h), Image.NEAREST)
    kreis_einzeichnen(bild, [(a.kern_m, (20, 20, 20, 255), 3)],
                      bbox_utm, x0, y0)
    bild.save(AUSGABE / f"{a.name}-ndvi-{a.jahr}.png")
    print(f"  ndvi-{a.jahr}.png")

    abstand = pa.abstandsraster(bbox_utm, ndvi.shape, x0, y0)
    ring = (abstand >= a.ring_von_m) & (abstand <= a.ring_bis_m)
    ringmedian = float(np.median(ndvi[ring & np.isfinite(ndvi) & (n >= 2)]))
    diff = np.where(np.isfinite(ndvi) & (n >= 2), ndvi - ringmedian, np.nan)
    bild = Image.fromarray(differenz_farbe(diff)).resize((a.kante, kante_h),
                                                         Image.NEAREST)
    kreis_einzeichnen(bild, [(a.kern_m, (20, 20, 20, 255), 3)],
                      bbox_utm, x0, y0)
    bild.save(AUSGABE / f"{a.name}-differenz-{a.jahr}.png")
    print(f"  differenz-{a.jahr}.png  (Ringmedian {ringmedian:.3f})")

    if lauf.pu_je_aoi:
        print(f"  Processing Units: {sum(lauf.pu_je_aoi.values()):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
