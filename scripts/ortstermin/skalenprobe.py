"""Skalenprobe: Ab welcher Objektgröße schlägt das Raster an — und ab welcher nicht?

Beantwortet die Frage, die vor jeder Folge steht: *Lebt diese Frage auf einer
Skala, die meine Quelle erreicht?* Drei unabhängige Wege, damit die Antwort nicht
an einer Annahme hängt:

  1 · RAUSCHBODEN   Wie stark streut die Anomalie dort, wo sich nichts ändert?
                    Darunter ist jede Schwelle Selbstbetrug. Gemessen an den
                    Pixeln der bebauten Nachbarschaft ohne Auffälligkeit.
  2 · VERDÜNNUNG    Ein Objekt kleiner als die Zelle färbt die Zelle nur
                    anteilig ein (lineare Mischung, Näherung erster Ordnung):
                        beobachtet = wahr * Objektfläche / Zellfläche
                    Daraus die kleinste Fläche, die die Schwelle noch reißt —
                    je Zelle und für die geforderte Mindestzahl Zellen.
  3 · VERGRÖBERUNG  Das echte Anomaliebild schrittweise auf gröbere Gitter
                    mitteln und jedes Mal neu zählen. Der ehrlichste Test:
                    er nimmt alles mit, was die Rechnung nicht kennt.

Aufruf (venv des NDVI-Batches):
    python skalenprobe.py --aoi oldenburg --bbox 8.155,53.105,8.285,53.185
Optionen: --delta 0.55 (NDVI-Sprung Kronendach -> offener Boden),
          --schwelle 0.10, --min-zellen 4, --json <pfad>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

BASIS = Path(__file__).resolve().parents[2]
TILES = BASIS / "ftp-mirror-geophora" / "tiles" / "ndvi"

sys.path.insert(0, str(BASIS / "scripts" / "ndvi"))
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS + Kartenpass, R1)


def lade(pfad: Path, band: int = 1) -> np.ndarray:
    with rasterio.open(pfad) as ds:
        return ds.read(band).astype("float64")


def kastenmittel(a: np.ndarray, radius: int) -> np.ndarray:
    """Gleitendes Mittel (Summenbild) — wie in ndvi_fundstellen.py."""
    gueltig = np.isfinite(a)
    s_w = np.pad(np.where(gueltig, a, 0.0), 1, mode="edge").cumsum(0).cumsum(1)
    s_n = np.pad(gueltig.astype("float64"), 1, mode="edge").cumsum(0).cumsum(1)

    def fenster(s):
        h, b = a.shape
        r0 = np.clip(np.arange(h) - radius, 0, h); r1 = np.clip(np.arange(h) + radius + 1, 0, h)
        c0 = np.clip(np.arange(b) - radius, 0, b); c1 = np.clip(np.arange(b) + radius + 1, 0, b)
        return (s[np.ix_(r1, c1)] - s[np.ix_(r0, c1)] - s[np.ix_(r1, c0)] + s[np.ix_(r0, c0)])

    with np.errstate(invalid="ignore", divide="ignore"):
        return fenster(s_w) / np.maximum(fenster(s_n), 1e-9)


def vergroebern(a: np.ndarray, faktor: int) -> np.ndarray:
    """Blockmittel um `faktor` — simuliert einen gröberen Sensor."""
    h = a.shape[0] // faktor * faktor
    b = a.shape[1] // faktor * faktor
    block = a[:h, :b].reshape(h // faktor, faktor, b // faktor, faktor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(block, axis=(1, 3))


def cluster_zaehlen(maske: np.ndarray, min_zellen: int) -> int:
    """Zusammenhängende Flächen (8er-Nachbarschaft) ab min_zellen — iterativ."""
    from collections import deque
    besucht = np.zeros_like(maske, dtype=bool)
    hoehe, breite = maske.shape
    treffer = 0
    for start in np.argwhere(maske):
        r0, c0 = int(start[0]), int(start[1])
        if besucht[r0, c0]:
            continue
        stapel, n = deque([(r0, c0)]), 0
        besucht[r0, c0] = True
        while stapel:
            r, c = stapel.popleft(); n += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < hoehe and 0 <= cc < breite and maske[rr, cc] and not besucht[rr, cc]:
                        besucht[rr, cc] = True
                        stapel.append((rr, cc))
        if n >= min_zellen:
            treffer += 1
    return treffer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", default="oldenburg")
    p.add_argument("--bbox", help="W,S,O,N — Ausschnitt")
    p.add_argument("--delta", type=float, default=0.55,
                   help="NDVI-Sprung des Objekts (Kronendach 0,80 -> offener Boden 0,25)")
    p.add_argument("--schwelle", type=float, default=0.10)
    p.add_argument("--min-zellen", type=int, default=4)
    p.add_argument("--stadt-radius-m", type=float, default=250.0)
    p.add_argument("--stadt-schwelle", type=float, default=0.55)
    p.add_argument("--json", type=Path)
    a = p.parse_args()

    ordner = TILES / a.aoi
    jetzt = lade(ordner / "aktuell-analyse.tif")
    basis = lade(ordner / "baseline-fenster.tif")
    saison = lade(ordner / "baseline.tif")

    with rasterio.open(ordner / "aktuell-analyse.tif") as ds:
        # Metrisches Gitter Pflicht (R1): Kantenlaenge exakt aus dem
        # Transform, keine Grad-Naeherung mehr.
        epsg = ch.pruefe_metrisch(ds, "aktuell-analyse.tif")
        bounds, breite, hoehe = ds.bounds, ds.width, ds.height
        raster_crs = ds.crs
        kante_m = float(ds.transform.a)
    d_x = (bounds.right - bounds.left) / breite
    d_y = (bounds.top - bounds.bottom) / hoehe

    if a.bbox:
        # Ausschnitt kommt als lon/lat und wird ins Raster-CRS projiziert.
        w, s, o, n = ch.bbox_nach(raster_crs,
                                  [float(v) for v in a.bbox.split(",")])
        cc, rr = np.meshgrid(np.arange(breite), np.arange(hoehe))
        x_mitte = bounds.left + (cc + 0.5) * d_x
        y_mitte = bounds.top - (rr + 0.5) * d_y
        gebiet = (x_mitte >= w) & (x_mitte <= o) & (y_mitte >= s) & (y_mitte <= n)
    else:
        gebiet = np.ones((hoehe, breite), bool)

    radius_px = max(1, int(round(a.stadt_radius_m / kante_m)))
    bebaut = kastenmittel(saison, radius_px) < a.stadt_schwelle

    with np.errstate(invalid="ignore"):
        anomalie = jetzt - basis
    gueltig = np.isfinite(anomalie) & gebiet & bebaut

    # ---------- 1 · Rauschboden ----------
    ruhig = gueltig & (np.abs(anomalie) < 3 * a.schwelle)   # Auffälliges raushalten
    werte = anomalie[ruhig]
    # Robuste Streuung: der Median der absoluten Abweichung, auf Sigma normiert
    mad = float(np.median(np.abs(werte - np.median(werte))))
    rauschen = {
        "n_pixel": int(werte.size),
        "median": round(float(np.median(werte)), 4),
        "sigma_robust": round(mad * 1.4826, 4),
        "p05": round(float(np.percentile(werte, 5)), 4),
        "p95": round(float(np.percentile(werte, 95)), 4),
        "schwelle_in_sigma": round(a.schwelle / max(mad * 1.4826, 1e-9), 2),
    }

    # ---------- 2 · Verdünnung ----------
    verduennung = []
    for gitter in (10, 20, 30, 60, 100, 1000):
        zelle = gitter ** 2
        a_zelle = zelle * a.schwelle / a.delta          # eine Zelle reißt die Schwelle
        verduennung.append({
            "gitter_m": gitter,
            "zellflaeche_m2": zelle,
            "min_objekt_1zelle_m2": round(a_zelle),
            "min_objekt_nzellen_m2": round(a_zelle * a.min_zellen),
            "entspricht_kronen": round(a_zelle * a.min_zellen / 78.5, 1),   # Krone 10 m Ø
        })

    # ---------- 3 · Vergröberung ----------
    vergroebert = []
    for faktor in (1, 2, 3, 5, 10):
        g = int(round(kante_m * faktor))
        an = vergroebern(np.where(gueltig, anomalie, np.nan), faktor) if faktor > 1 else \
            np.where(gueltig, anomalie, np.nan)
        gg = np.isfinite(an)
        verlust = gg & (an <= -a.schwelle)
        vergroebert.append({
            "gitter_m": g,
            "zellen_im_gebiet": int(gg.sum()),
            "anteil_verlust": round(float(verlust.sum()) / max(int(gg.sum()), 1), 4),
            "cluster_ab_min": cluster_zaehlen(verlust, a.min_zellen),
            "flaeche_je_cluster_m2": g * g * a.min_zellen,
        })

    ergebnis = {
        "aoi": a.aoi, "gitter_m": round(kante_m, 1),
        "crs": f"EPSG:{epsg}",
        "aufruf": ch.aufruf_protokoll(),
        "annahmen": {"delta_ndvi": a.delta, "schwelle": a.schwelle,
                     "min_zellen": a.min_zellen,
                     "stadtmaske": {"radius_m": a.stadt_radius_m, "schwelle": a.stadt_schwelle}},
        "rauschboden": rauschen,
        "verduennung": verduennung,
        "vergroeberung": vergroebert,
    }
    print(json.dumps(ergebnis, ensure_ascii=False, indent=2))
    if a.json:
        a.json.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\ngeschrieben: {a.json}")


if __name__ == "__main__":
    main()
