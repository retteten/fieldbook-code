# -*- coding: utf-8 -*-
"""Duerrereaktion einer Baumschullandschaft — NDVI und NDMI im selben Raster.

Warum zwei Indizes statt einem
------------------------------
Der NDVI misst, wie gruen eine Flaeche ist. Bei Trockenheit ist das die
falsche Frage — oder genauer: die zu langsame. Ein Gehoelz mit tiefen Wurzeln
bleibt gruen, waehrend sein Gewebe laengst austrocknet; ein Rasen wird braun,
sobald die oberen zwanzig Zentimeter leer sind. Wer allein den NDVI ansieht,
haelt das eine fuer gesund und das andere fuer sterbend, und beides kann
falsch sein.

Der NDMI (Normalized Difference Moisture Index, (B8A-B11)/(B8A+B11)) misst
den Wassergehalt des Blattes selbst: B11 liegt im kurzwelligen Infrarot, wo
fluessiges Wasser absorbiert. Trocknet das Blattgewebe, steigt die Reflexion
in B11 und der Index faellt — Wochen bevor das Blatt seine Farbe verliert.

Die beiden zusammen ergeben vier Faelle, und erst die machen eine Karte
lesbar:

    gruen + feucht   Wasser ist da (bewaessert, Grundwassernaehe, Moorboden)
    gruen + trocken  Gehoelz zehrt aus der Tiefe — Stress, den Gruen verdeckt
    braun + trocken  klassische Duerre, Grasland ueber Sand
    braun + feucht   kein Bestand: abgeerntet, gerodet, offener Boden, Wasser

Die dritte Zeile findet jeder. Die zweite ist der Grund fuer dieses Skript.

Was es tut
----------
Holt je Jahr einen Peak-Season-Median (ISO-Wochen 23-35, wie der Monitor) mit
drei Baendern — NDVI, NDMI, Zahl der wolkenfreien Beobachtungen — auf dem
metrischen Analyse-Gitter von 20 m (R1, 12.08.2026: WGS 84/UTM 32N,
EPSG:32632 — die CDSE-API rendert kein 25832, s. crs_helfer.py;
Zellflaeche exakt 400 m2), und legt ihn in den lokalen Cache.
Ein Jahr, das schon liegt, wird nicht noch einmal geholt.

Die AOI steht bewusst HIER und nicht in ftp-mirror-geophora/tiles/ndvi/
aois.json: Ein Eintrag dort erschiene sofort im oeffentlichen Monitor und im
Layer-Explorer. Das ist Roberts Entscheidung, nicht die eines Rechenskripts.
Fuer den Abruf genuegt process(), und das liest nur bbox und Aufloesung.

Aufruf (venv des NDVI-Batches):
    python scripts/ortstermin/duerre_baumschulen.py --jahre 2018-2025
    python scripts/ortstermin/duerre_baumschulen.py --nur-bestand
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

BASIS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASIS / "scripts" / "ndvi"))

import ndvi_batch as nb  # noqa: E402  (Pfad muss vorher stehen)

CACHE = BASIS / "scripts" / "ndvi" / "cache"

# ---------------------------------------------------------------------------
# Die Flaeche
# ---------------------------------------------------------------------------
# Ammerland zwischen Westerstede, Bad Zwischenahn und Edewecht — die dichteste
# Baumschullandschaft Europas, unmittelbar westlich von Oldenburg. Der
# Ausschnitt fasst das Zwischenahner Meer mit ein: eine Wasserflaeche im Bild
# ist der beste Beleg dafuer, dass die Indizes richtig herum liegen.
AOI = {
    "id": "ammerland",
    "name": "Ammerland — Baumschulland",
    "bbox": [7.87, 53.12, 8.13, 53.30],
    "aufloesung_m": 10,
    "aufloesung_analyse_m": 20,
}


def evalscript_ndvi_ndmi() -> str:
    """Drei Baender: NDVI-Median, NDMI-Median, Zahl der Beobachtungen.

    Beide Indizes aus DENSELBEN Aufnahmen und mit derselben Wolkenmaske —
    sonst vergleicht man am Ende zwei verschiedene Sommer miteinander. B8A
    statt B08 fuer den NDMI, weil B8A und B11 nativ auf 20 m liegen; B08
    (10 m) waere hier nur hochskalierte Genauigkeit, die es nicht gibt.
    """
    return """//VERSION=3
var SCL_VERWORFEN = [3, 8, 9, 10, 11];
function gueltig(p) {
  return p.dataMask === 1 && SCL_VERWORFEN.indexOf(p.SCL) === -1;
}
function median(werte) {
  if (werte.length === 0) return NaN;
  werte.sort(function (a, b) { return a - b; });
  var mitte = Math.floor(werte.length / 2);
  return werte.length % 2 === 1 ? werte[mitte]
                                : 0.5 * (werte[mitte - 1] + werte[mitte]);
}
function setup() {
  return {
    input: [{bands: ["B04", "B08", "B8A", "B11", "SCL", "dataMask"]}],
    output: {bands: 3, sampleType: "FLOAT32"},
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(proben) {
  var ndvi = [], ndmi = [];
  for (var i = 0; i < proben.length; i++) {
    var p = proben[i];
    if (!gueltig(p)) continue;
    var n1 = p.B08 + p.B04;
    if (n1 > 0) ndvi.push((p.B08 - p.B04) / n1);
    var n2 = p.B8A + p.B11;
    if (n2 > 0) ndmi.push((p.B8A - p.B11) / n2);
  }
  return [median(ndvi), median(ndmi), ndvi.length];
}
"""


def pfad(jahr: int, winter: bool = False) -> Path:
    art = "winter" if winter else "duerre"
    return CACHE / AOI["id"] / f"{art}-{jahr}.tif"


def winterfenster(jahr: int):
    """Spaetwinter, 1. Februar bis 20. Maerz des angegebenen Jahres.

    Warum nicht Dezember: Bei 53 Grad Nord steht die Sonne im Dezember rund
    13 Grad ueber dem Horizont. Die Schatten sind dann laenger als die
    Baumreihen, und ein Index, der Schatten misst, misst nicht die Pflanze.
    Mitte Februar bis Mitte Maerz steht sie bei 22 bis 30 Grad — und die
    Laubgehoelze sind noch kahl, der Austrieb kommt hier erst Mitte April.
    Das ist das Fenster, in dem sich immergruen von sommergruen trennt.
    """
    return dt.date(jahr, 2, 1), dt.date(jahr, 3, 20)


def schreibe_dreiband(ziel: Path, ndvi, ndmi, n, bbox_utm, epsg: int) -> None:
    """Dreiband-GeoTIFF im metrischen Ziel-CRS (R1): bbox_utm und epsg
    kommen aus nb.aoi_gitter() — dieselbe Georeferenz wie der Abruf."""
    ziel.parent.mkdir(parents=True, exist_ok=True)
    hoehe, breite = ndvi.shape
    west, sued, ost, nord = bbox_utm
    profil = {
        "driver": "GTiff", "width": breite, "height": hoehe, "count": 3,
        "dtype": "float32", "crs": f"EPSG:{epsg}",
        "transform": from_bounds(west, sued, ost, nord, breite, hoehe),
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "compress": "deflate", "nodata": float("nan"),
    }
    with rasterio.open(ziel, "w", **profil) as q:
        for i, (a, name) in enumerate(
                ((ndvi, "ndvi_median"), (ndmi, "ndmi_median"), (n, "n")), 1):
            q.write(a.astype("float32"), i)
            q.set_band_description(i, name)


def lies_dreiband(jahr: int):
    with rasterio.open(pfad(jahr)) as q:
        return q.read(1), q.read(2), q.read(3)


def bestand() -> list[int]:
    ordner = CACHE / AOI["id"]
    if not ordner.exists():
        return []
    return sorted(int(p.stem.split("-")[1]) for p in ordner.glob("duerre-*.tif"))


def hole(jahre: list[int], erneuern: bool, winter: bool = False) -> None:
    args = argparse.Namespace(ausgabe=str(nb.STANDARD_AUSGABE), aoi=None,
                              jahr=None, force=False, wochen=None,
                              aufloesung=None)
    lauf = nb.Lauf(args)
    # Nur die eigene AOI rechnen — die Konfiguration (Saisonfenster, Farbskala)
    # bleibt die des Monitors, damit die Jahre mit dem Rest vergleichbar sind.
    lauf.aois = [AOI]

    epsg, bbox_utm, breite, hoehe = nb.aoi_gitter(AOI, AOI["aufloesung_analyse_m"])
    print(f"AOI {AOI['id']}: {breite} x {hoehe} Pixel auf 20 m (EPSG:{epsg})")

    for jahr in jahre:
        ziel = pfad(jahr, winter)
        if ziel.exists() and not erneuern:
            print(f"  {jahr}  liegt schon")
            continue
        von, bis = (winterfenster(jahr) if winter
                    else nb.saison_fenster(jahr, lauf.konfig["saison"]))
        print(f"  {jahr}  {von} bis {bis} ...", end="", flush=True)
        try:
            # NICHT nb.TYP_GEOTIFF: Die Processing-API weist die Langform
            # "image/tiff; application=geotiff" als Request-Format mit
            # HTTP 400 ab (gemessen 09.08.2026). Die Konstante im Batch ist
            # der STAC-Medientyp der Assets — dort korrekt, als Request
            # falsch. Der Rueckgabewert ist unveraendert ein GeoTIFF.
            inhalt = nb.process(lauf, AOI, von, bis, evalscript_ndvi_ndmi(),
                                "image/tiff",
                                aufloesung_m=AOI["aufloesung_analyse_m"])
        except Exception as fehler:
            print(f" FEHLER: {fehler}")
            continue
        with nb.MemoryFile(inhalt) as speicher:
            with speicher.open() as q:
                ndvi, ndmi, n = q.read(1), q.read(2), q.read(3)
        # n = 0 heisst: kein wolkenfreier Blick. Der Median ist dann NaN, aber
        # ein Rechenweg, der das uebersieht, macht daraus eine 0 und damit eine
        # kahle Flaeche, die es nie gab.
        ndvi = np.where(n >= 1, ndvi, np.nan)
        ndmi = np.where(n >= 1, ndmi, np.nan)
        schreibe_dreiband(ziel, ndvi, ndmi, n, bbox_utm, epsg)
        gueltig = np.isfinite(ndvi)
        print(f" ok — {gueltig.mean() * 100:.1f} % gueltig, "
              f"NDVI {np.nanmedian(ndvi):.3f}, NDMI {np.nanmedian(ndmi):.3f}, "
              f"n-Median {np.median(n):.0f}")

    summe = sum(lauf.pu_je_aoi.values())
    if summe:
        print(f"\nProcessing Units in diesem Lauf: {summe:.1f}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jahre", default="2018-2025",
                   help="Bereich '2018-2025' oder Liste '2018,2022,2024'")
    p.add_argument("--erneuern", action="store_true",
                   help="vorhandene Jahre neu holen")
    p.add_argument("--nur-bestand", action="store_true",
                   help="nur zeigen, was im Cache liegt")
    p.add_argument("--winter", action="store_true",
                   help="Spaetwinter-Composite (Feb-Maerz) statt Sommer — "
                        "trennt immergruene von sommergruenen Bestaenden")
    a = p.parse_args()

    if a.nur_bestand:
        da = bestand()
        print(f"Cache {AOI['id']}: {len(da)} Jahre" + (f" — {da}" if da else ""))
        return 0

    if "-" in a.jahre:
        von, bis = (int(x) for x in a.jahre.split("-"))
        jahre = list(range(von, bis + 1))
    else:
        jahre = [int(x) for x in a.jahre.split(",")]
    hole(jahre, a.erneuern, a.winter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
