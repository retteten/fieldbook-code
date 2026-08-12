# -*- coding: utf-8 -*-
"""Klimatischer Kontext eines Sommers — DWD-Monatsraster, Gebietsmittel, Rangliste.

Wozu
----
Die Ammerland-Folge nennt 2018 und 2022 Duerresommer. Der Satellit kann das
nicht belegen — er misst die Reaktion der Vegetation, nicht das Wetter. Die
Belege liegen beim DWD: offene Monatsraster (1 km) ueber opendata.dwd.de.

Gerechnet wird die klimatische Wasserbilanz des Sommers (Juni bis August):

    KWB = Niederschlag − potenzielle Verdunstung (AMBAV, ueber Gras)

als Gebietsmittel ueber den Ausschnitt der Folge, fuer jeden Sommer seit 1991.
Dazu die Bodenfeuchte (% nutzbarer Feldkapazitaet unter Gras) als agronomischer
Zweitzeuge. Ergebnis ist eine Rangliste: Wo stehen 2018 und 2022 in 35 Jahren
wirklich? Das ersetzt die Formel "zwei der trockensten Sommer" durch eine Zahl.

Warum genau diese Produkte
--------------------------
* `precipitation` allein waere zu wenig: Ein heisser Sommer trocknet auch bei
  mittlerem Regen. Die Wasserbilanz fasst beides.
* `soil_moist` ist die Groesse, mit der Landwirtschaft tatsaechlich rechnet —
  und sie integriert das Fruehjahr mit, das der Sommer-KWB nicht sieht.
* Der `drought_index` (SPI, ab 1970) beruht nur auf Niederschlag; er bleibt
  hier bewusst aussen vor.

Datenformat (empirisch geprueft am 09.08.2026, Beschreibungs-PDFs beim DWD):
ESRI-ASCII-Grid, gezippt, 1 km, Gauss-Krueger 3 (EPSG:31467). Werte:
precipitation in mm, evapo_p in 1/10 mm, soil_moist in % nFK — das Skript
prueft die Groessenordnung zur Laufzeit und bricht ab, statt still Unsinn zu
mitteln.

Lizenz: DWD CDC OpenData, CC BY 4.0 — Registerzeile in
docs/recht-und-lizenzen.md § 3.1 (geprueft 06.08.2026). Vermerk fuer Bilder:
"Datenbasis: Deutscher Wetterdienst, Rasterdaten bildlich wiedergegeben".

Aufruf (venv des NDVI-Batches):
    python scripts/klima/duerre_kontext.py --json docs/daten/klima-ammerland-1991-2025.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from rasterio.warp import transform

BASIS = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache"

CDC = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/monthly"
KOPFZEUG = {"User-Agent": "geophora.de Rechenwerkstatt (robert@retteten.de)"}

# Standard: Ausschnitt der Ammerland-Folge (duerre_baumschulen.py) — bewusst
# dieselben Zahlen, damit DWD- und Sentinel-Aussagen ueber dieselbe Flaeche
# sprechen. Mit --bbox laesst sich jede andere AOI rechnen; das Fenster wird je
# Rasterkopf neu bestimmt, der Cache der Monatsraster bleibt gemeinsam nutzbar.
BBOX = (7.87, 53.12, 8.13, 53.30)

SOMMER = (6, 7, 8)

# (Produkt, Unterordner je Monat?, Skalenfaktor nach mm bzw. %, Plausibilitaet
#  des GEBIETSMITTELS eines Sommermonats nach Skalierung)
PRODUKTE = {
    "precipitation": ("precipitation", True, 1.0, (5.0, 400.0)),
    "evapo_p": ("evapo_p", False, 0.1, (30.0, 250.0)),
    "soil_moist": ("soil_moist", False, 1.0, (5.0, 200.0)),
}
MONATSORDNER = {6: "06_Jun", 7: "07_Jul", 8: "08_Aug"}


def datei_url(produkt: str, jahr: int, monat: int) -> str:
    name, je_monat, _, _ = PRODUKTE[produkt]
    ordner = f"{name}/{MONATSORDNER[monat]}" if je_monat else name
    return (f"{CDC}/{ordner}/grids_germany_monthly_{name}_"
            f"{jahr}{monat:02d}.asc.gz")


def hole(produkt: str, jahr: int, monat: int) -> Path:
    ziel = CACHE / produkt / f"{jahr}{monat:02d}.asc.gz"
    if ziel.exists() and ziel.stat().st_size > 0:
        return ziel
    ziel.parent.mkdir(parents=True, exist_ok=True)
    url = datei_url(produkt, jahr, monat)
    with urllib.request.urlopen(
            urllib.request.Request(url, headers=KOPFZEUG), timeout=120) as q:
        ziel.write_bytes(q.read())
    return ziel


def lies_asc(pfad: Path):
    """ESRI-ASCII-Grid -> (array, header). Header-Schluessel kleingeschrieben."""
    with gzip.open(pfad, "rt") as f:
        kopf = {}
        while True:
            pos = f.tell()
            teile = f.readline().split()
            if len(teile) == 2 and not teile[0].lstrip("-").replace(".", "").isdigit():
                kopf[teile[0].lower()] = float(teile[1])
            else:
                f.seek(pos)
                break
        daten = np.loadtxt(f)
    nodata = kopf.get("nodata_value", -999.0)
    return np.where(daten == nodata, np.nan, daten), kopf


_fenster_cache: dict[tuple, tuple] = {}


def gebietsmittel(pfad: Path, faktor: float) -> float:
    """Mittel ueber die bbox — Zeilen/Spalten aus dem GK3-Header abgeleitet."""
    daten, kopf = lies_asc(pfad)
    schluessel = (kopf["ncols"], kopf["nrows"], kopf["xllcorner"],
                  kopf["yllcorner"], kopf["cellsize"])
    if schluessel not in _fenster_cache:
        west, sued, ost, nord = BBOX
        (x0, x1), (y0, y1) = transform("EPSG:4326", "EPSG:31467",
                                       [west, ost], [sued, nord])
        zelle = kopf["cellsize"]
        c0 = int((x0 - kopf["xllcorner"]) / zelle)
        c1 = int((x1 - kopf["xllcorner"]) / zelle) + 1
        oben = kopf["yllcorner"] + kopf["nrows"] * zelle
        r0 = int((oben - y1) / zelle)
        r1 = int((oben - y0) / zelle) + 1
        _fenster_cache[schluessel] = (r0, r1, c0, c1)
    r0, r1, c0, c1 = _fenster_cache[schluessel]
    aus = daten[r0:r1, c0:c1] * faktor
    if np.isnan(aus).all():
        raise RuntimeError(f"Nur NoData im Ausschnitt: {pfad.name}")
    return float(np.nanmean(aus))


def sommerwert(produkt: str, jahr: int) -> float:
    """Sommersumme (mm) bzw. Sommermittel (% nFK) als Gebietsmittel."""
    _, _, faktor, (lo, hi) = PRODUKTE[produkt]
    werte = []
    for monat in SOMMER:
        w = gebietsmittel(hole(produkt, jahr, monat), faktor)
        if not (lo <= w <= hi):
            raise RuntimeError(
                f"{produkt} {jahr}-{monat:02d}: Gebietsmittel {w:.1f} liegt "
                f"ausserhalb der Plausibilitaet [{lo}, {hi}] — Skalenfaktor "
                f"pruefen, NICHT weiterrechnen.")
        werte.append(w)
    return float(sum(werte)) if produkt != "soil_moist" else float(np.mean(werte))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--von", type=int, default=1991,
                   help="erstes Jahr (evapo_p/soil_moist beginnen 1991)")
    p.add_argument("--bis", type=int, default=2025,
                   help="letztes VOLLSTAENDIGES Sommerjahr")
    p.add_argument("--bbox", help="W,S,O,N in Dezimalgrad (Default: Ammerland)")
    p.add_argument("--name", default="ammerland",
                   help="Kennung fuer die Ausgabe")
    p.add_argument("--json", type=Path)
    a = p.parse_args()

    if a.bbox:
        global BBOX
        BBOX = tuple(float(x) for x in a.bbox.split(","))
        if len(BBOX) != 4:
            sys.exit("--bbox braucht vier Zahlen: W,S,O,N")

    jahre = list(range(a.von, a.bis + 1))
    print(f"DWD-Monatsraster, Gebietsmittel {BBOX} — Sommer {a.von}-{a.bis}")
    reihen = []
    for jahr in jahre:
        try:
            nieder = sommerwert("precipitation", jahr)
            verdunst = sommerwert("evapo_p", jahr)
            boden = sommerwert("soil_moist", jahr)
        except Exception as fehler:
            print(f"  {jahr}: FEHLER {fehler}")
            return 1
        reihen.append({"jahr": jahr,
                       "niederschlag_mm": round(nieder, 1),
                       "verdunstung_mm": round(verdunst, 1),
                       "kwb_mm": round(nieder - verdunst, 1),
                       "bodenfeuchte_prozent_nfk": round(boden, 1)})
        print(f"  {jahr}  P {nieder:6.1f}  ETp {verdunst:6.1f}  "
              f"KWB {nieder - verdunst:+7.1f}  Boden {boden:5.1f} % nFK")

    rang = sorted(reihen, key=lambda r: r["kwb_mm"])
    print(f"\nRangliste der Sommer-Wasserbilanz ({len(rang)} Jahre, "
          f"trockenster zuerst):")
    for platz, r in enumerate(rang[:8], 1):
        print(f"  {platz}. {r['jahr']}  KWB {r['kwb_mm']:+7.1f} mm  "
              f"Boden {r['bodenfeuchte_prozent_nfk']:.1f} % nFK")
    median_kwb = float(np.median([r["kwb_mm"] for r in reihen]))
    print(f"\n  Median aller Sommer: KWB {median_kwb:+.1f} mm")
    for jahr in (2018, 2022):
        r = next((x for x in reihen if x["jahr"] == jahr), None)
        if r is None:
            continue          # Teillaeufe (--von/--bis) decken das Jahr nicht ab
        platz = rang.index(r) + 1
        print(f"  {jahr}: Platz {platz} von {len(rang)}, "
              f"KWB {r['kwb_mm']:+.1f} mm, Boden "
              f"{r['bodenfeuchte_prozent_nfk']:.1f} % nFK")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "aoi": a.name,
            "quelle": "DWD CDC OpenData, Monatsraster 1 km (precipitation, "
                      "evapo_p, soil_moist), CC BY 4.0",
            "vermerk": "Datenbasis: Deutscher Wetterdienst, Rasterdaten "
                       "bildlich wiedergegeben",
            "abruf": "2026-08-09",
            "bbox": list(BBOX), "monate": list(SOMMER),
            "median_kwb_mm": round(median_kwb, 1),
            "sommer": reihen,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nGeschrieben: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
