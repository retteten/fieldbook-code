# -*- coding: utf-8 -*-
"""Handstichprobe: Baumschulquartier gegen Gruenland, im Luftbild bestimmt.

Warum von Hand
--------------
Die Identitaetsprobe in duerre_auswertung.py zeigt, dass sich Baumschulflaeche
nicht aus dem Spektrum abgrenzen laesst — die Winterverteilung hat einen
Gipfel und kein Tal. Damit faellt jede automatische Klassifikation aus.

Was bleibt, ist die ehrliche Notloesung: hinsehen. Jeder Punkt unten wurde im
DOP20 (20 cm) einzeln angesehen und eingeordnet. Ein Baumschulquartier
erkennt man dort an drei Dingen, die auf 20 m alle verschwinden: den feinen
Reihen im Meterabstand, den rechteckigen Quartieren mit Wendestreifen, und
den Folien- oder Containerflaechen daneben.

Was diese Stichprobe IST und was nicht
--------------------------------------
Sie ist eine Handvoll bestaetigter Flaechen, kein Zufallsstichprobe. Sie kann
zeigen, dass ein vermuteter Unterschied in der Groessenordnung nicht auftaucht.
Sie kann NICHT die Baumschulen des Ammerlandes repraesentieren, und sie kann
aus einem gefundenen Unterschied keine Signifikanz machen. Wer mehr will,
braucht die amtlichen Schlaggeometrien (InVeKoS) — und dafuer zuerst eine
Zeile im Lizenzregister.

Jeder Wert ist das Mittel eines 3x3-Fensters auf dem 20-m-Gitter, also
60 x 60 m um den Punkt.

Aufruf (venv des NDVI-Batches):
    python scripts/ortstermin/duerre_stichprobe.py
    python scripts/ortstermin/duerre_stichprobe.py --bogen probe.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

BASIS = Path(__file__).resolve().parents[2]
CACHE = BASIS / "scripts" / "ndvi" / "cache" / "ammerland"

sys.path.insert(0, str(BASIS / "scripts" / "ndvi"))
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS + Kartenpass, R1)

# Von Hand im DOP20 bestimmt am 09.08.2026. Die Kennung ist die Reihenfolge
# der Sichtung, nicht eine Rangfolge — sie steht nur da, damit ein Punkt in
# den Kontaktboegen wiederfindbar bleibt.
PUNKTE = [
    # Baumschulquartiere: Reihen, Quartiere, Folien-/Containerflaechen sichtbar
    ("A", "baumschule", 53.1535, 7.9889, "Edewecht Nordost, Reihenquartiere"),
    ("C", "baumschule", 53.1554, 7.9730, "Edewecht Nordwest, Quartiere am Weg"),
    ("G", "baumschule", 53.2215, 7.8930, "Ocholt West, Reihen und Wendestreifen"),
    ("H", "baumschule", 53.2260, 7.8975, "Ocholt Nord, Quartiere am Hof"),
    ("O", "baumschule", 53.2480, 7.9350, "Westerstede, Gewaechshaeuser und Container"),
    ("U", "baumschule", 53.2110, 7.8939, "Ocholt Sued, grosse Containerflaeche"),
    ("W", "baumschule", 53.2311, 7.9134, "Block mit farbigen Quartieren, Graben ringsum"),
    ("X", "baumschule", 53.1689, 8.0105, "Folienhaeuser und Quartiere"),
    ("Y", "baumschule", 53.1854, 7.9523, "Reihenquartier neben Acker"),
    # Gruenland und Acker: geschlossene Narbe, Wallhecken, keine Reihen
    ("B", "gruenland", 53.1478, 7.9896, "Weide am Teich"),
    ("F", "gruenland", 53.2430, 7.9210, "Weide am Hof"),
    ("I", "gruenland", 53.2180, 7.8880, "Acker und Weide, Wallhecken"),
    ("M", "gruenland", 53.1460, 7.9800, "lange Schlaege"),
    ("N", "gruenland", 53.1520, 7.9660, "grosser einheitlicher Schlag"),
    ("Q", "gruenland", 53.2300, 7.9150, "Weide mit Hecken"),
    ("S", "gruenland", 53.2050, 7.9450, "Schlag am Graben"),
    ("V", "gruenland", 53.2231, 7.9002, "Gruenland am Weg"),
    ("Z", "gruenland", 53.1577, 7.9784, "Weide am Ortsrand"),
    # Wald als dritter Bezug: tief wurzelnd, nicht bewirtschaftet bewaessert
    ("P", "wald", 53.2380, 7.9420, "geschlossener Laubwald"),
    ("K", "wald", 53.2120, 7.8960, "Waldrand"),
]


def lies(jahr: int):
    with rasterio.open(CACHE / f"duerre-{jahr}.tif") as q:
        # Metrisches Gitter Pflicht (R1): Grad-Altbestaende laut abweisen.
        ch.pruefe_metrisch(q, f"duerre-{jahr}.tif")
        return q.read(1), q.read(2), q.transform, q.crs


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duerre", default="2018,2022")
    p.add_argument("--normal", default="2021,2024,2025")
    p.add_argument("--fenster", type=int, default=1,
                   help="Halbe Fensterbreite in Zellen (1 = 3x3 = 60 m)")
    p.add_argument("--json", type=Path)
    a = p.parse_args()

    duerre = [int(x) for x in a.duerre.split(",")]
    normal = [int(x) for x in a.normal.split(",")]
    ndvi_n = np.nanmedian(np.stack([lies(j)[0] for j in normal]), axis=0)
    ndvi_d = np.nanmedian(np.stack([lies(j)[0] for j in duerre]), axis=0)
    ndmi_n = np.nanmedian(np.stack([lies(j)[1] for j in normal]), axis=0)
    ndmi_d = np.nanmedian(np.stack([lies(j)[1] for j in duerre]), axis=0)
    _, _, T, raster_crs = lies(duerre[0])
    epsg = int(raster_crs.to_epsg() or 0)
    r = a.fenster

    def mittel(feld, lat, lon):
        # Punkt (lon/lat) zuerst ins Raster-CRS projizieren (R1): rowcol
        # direkt mit Grad traefe auf dem UTM-Cache das falsche Pixel.
        x, y = ch.punkt_nach(raster_crs, lon, lat)
        row, col = rasterio.transform.rowcol(T, x, y)
        aus = feld[max(row - r, 0):row + r + 1, max(col - r, 0):col + r + 1]
        return float(np.nanmean(aus))

    zeilen = []
    print(f"Handstichprobe — {len(PUNKTE)} Punkte, je {(2 * r + 1) * 20} x "
          f"{(2 * r + 1) * 20} m")
    print(f"Duerre {duerre} gegen Normal {normal}\n")
    print(f"{'':3} {'Klasse':<11} {'NDVI':>7} {'dNDVI':>8} {'NDMI':>7} "
          f"{'dNDMI':>8}   Beschreibung")
    print("-" * 92)
    for kennung, klasse, lat, lon, was in PUNKTE:
        vn, vd = mittel(ndvi_n, lat, lon), mittel(ndvi_d, lat, lon)
        mn, md = mittel(ndmi_n, lat, lon), mittel(ndmi_d, lat, lon)
        zeilen.append({"kennung": kennung, "klasse": klasse, "lat": lat,
                       "lon": lon, "beschreibung": was,
                       "ndvi_normal": round(vn, 4), "d_ndvi": round(vd - vn, 4),
                       "ndmi_normal": round(mn, 4), "d_ndmi": round(md - mn, 4)})
        print(f"{kennung:<3} {klasse:<11} {vn:>7.3f} {vd - vn:>+8.3f} "
              f"{mn:>7.3f} {md - mn:>+8.3f}   {was}")

    print(f"\n{'Klasse':<12} {'n':>3} {'NDVI norm':>10} {'dNDVI':>9} "
          f"{'NDMI norm':>10} {'dNDMI':>9}")
    print("-" * 58)
    gruppen = {}
    for klasse in ("baumschule", "gruenland", "wald"):
        teil = [z for z in zeilen if z["klasse"] == klasse]
        if not teil:
            continue
        g = {k: round(float(np.median([z[k] for z in teil])), 4)
             for k in ("ndvi_normal", "d_ndvi", "ndmi_normal", "d_ndmi")}
        g["n"] = len(teil)
        gruppen[klasse] = g
        print(f"{klasse:<12} {g['n']:>3} {g['ndvi_normal']:>10.3f} "
              f"{g['d_ndvi']:>+9.3f} {g['ndmi_normal']:>10.3f} "
              f"{g['d_ndmi']:>+9.3f}")

    if "baumschule" in gruppen and "gruenland" in gruppen:
        b, gr = gruppen["baumschule"], gruppen["gruenland"]
        print(f"\nUnterschied Baumschule minus Gruenland:")
        print(f"  im Gruen    {b['d_ndvi'] - gr['d_ndvi']:+.3f} NDVI")
        print(f"  in der Feuchte {b['d_ndmi'] - gr['d_ndmi']:+.3f} NDMI")
        print("\n  Zur Einordnung: Die Streuung zwischen zwei unauffaelligen")
        print("  Sommern liegt bei rund 0,07 NDVI. Ein Unterschied, der")
        print("  darunter bleibt, ist mit dieser Stichprobe nicht zu halten.")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "verfahren": "von Hand im DOP20 (20 cm) bestimmt, 09.08.2026",
            "fenster_m": (2 * r + 1) * 20,
            "duerrejahre": duerre, "normaljahre": normal,
            "crs": f"EPSG:{epsg}",
            "gruppen": gruppen, "punkte": zeilen,
            # Kartenpass (Rezeptur § 8.1): der exakte Aufruf dieses Laufs.
            "aufruf": ch.aufruf_protokoll(),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nGeschrieben: {a.json}")


if __name__ == "__main__":
    main()
