"""Wann ist an diesem Ort ueberhaupt Vegetation zu sehen? Gemessen, nicht geraten.

Warum das noetig ist
--------------------
punktauswertung.py rechnet auf einem festen Fenster (1. Februar bis 15. April)
mit der Begruendung "Regenzeit-Gipfel im suedlichen Afrika". Das ist
Lehrbuchwissen, keine Messung. Liegt der Gipfel an DIESEM Ort spaeter, misst
das Fenster die abklingende Vegetation; liegt er frueher, den Aufwuchs. Beides
verzerrt jeden Jahresvergleich — und zwar systematisch, nicht zufaellig.

Diese Probe holt Monats-Composites ueber ganze Jahreszyklen und zeigt die
Kurve. Danach steht im Bericht eine gemessene Aussage statt einer Annahme.

Der Zyklus laeuft von Juli bis Juni, nicht Januar bis Dezember: Eine Regenzeit
im suedlichen Afrika ueberschreitet den Jahreswechsel, ein Kalenderjahr
zerschneidet sie in der Mitte.

Aufruf (echter Publikationslauf Botswana, 08.08.2026):
    python saisonprobe.py --lat -23.973037 --lon 25.854650 --name botswana \
                          --zyklen 2022 2024 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndvi_batch as nb  # noqa: E402
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS, R1)
import punktauswertung as pa  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AUSGABE = REPO / "docs" / "daten"
CACHE = Path(__file__).resolve().parent / "cache" / "saison"

MONATSNAMEN = ["Jul", "Aug", "Sep", "Okt", "Nov", "Dez",
               "Jan", "Feb", "Mär", "Apr", "Mai", "Jun"]


def monatsfenster(zyklus_ende: int, index: int) -> tuple[date, date]:
    """index 0 = Juli des Vorjahres … index 11 = Juni des Zyklusjahres."""
    monat = 7 + index
    jahr = zyklus_ende - 1
    if monat > 12:
        monat -= 12
        jahr = zyklus_ende
    if monat == 12:
        return date(jahr, 12, 1), date(jahr, 12, 31)
    return date(jahr, monat, 1), date(jahr, monat + 1, 1) - __import__("datetime").timedelta(days=1)


def hole(lauf, aoi, von, bis):
    # Ziel-CRS im Cache-Schluessel (R1): Grad-Gitter-Altbestaende laufen ins
    # Leere und werden metrisch neu geholt statt lagefalsch weiterverwendet.
    kennung = hashlib.sha1(
        f"{aoi['bbox']}|{aoi['aufloesung_m']}|EPSG:{ch.aoi_epsg(aoi)}|{von}|{bis}"
        .encode()).hexdigest()[:16]
    treffer = CACHE / f"{aoi['id']}-{von}-{kennung}.tif"
    if treffer.exists():
        return nb.lies_tiff(treffer, mit_n=True)
    roh = nb.process(lauf, aoi, von, bis, nb.evalscript_ndvi_tiff(),
                     pa.TYP_TIFF, aufloesung_m=aoi["aufloesung_m"])
    treffer.parent.mkdir(parents=True, exist_ok=True)
    treffer.write_bytes(roh)
    return nb.lies_tiff_bytes(roh, mit_n=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--zyklen", type=int, nargs="+", default=[2024, 2026],
                    help="Endjahre der Juli-Juni-Zyklen")
    ap.add_argument("--kern-m", type=float, default=500.0)
    ap.add_argument("--ring-von-m", type=float, default=800.0)
    ap.add_argument("--ring-bis-m", type=float, default=1500.0)
    ap.add_argument("--aufloesung", type=float, default=20.0)
    a = ap.parse_args()

    halbkante = a.ring_bis_m * 1.15
    bbox = pa.bbox_um(a.lat, a.lon, halbkante)
    aoi = {"id": a.name, "bbox": bbox, "aufloesung_m": a.aufloesung}
    # Metrisches Analysegitter (R1) — gleiche Projektion wie punktauswertung.
    epsg, bbox_utm, _, _ = nb.aoi_gitter(aoi)
    x0, y0 = ch.punkt_nach(epsg, a.lon, a.lat)
    lauf = pa.SchlankerLauf()

    print(f"Saisonprobe {a.name} — Zyklen Juli bis Juni: "
          f"{', '.join(str(z) for z in a.zyklen)}")
    print(f"  {'Monat':<7}", end="")
    for z in a.zyklen:
        print(f"{'Kern ' + str(z):>12}{'Ring':>8}{'n':>5}", end="")
    print("   Anteil vom Jahresgipfel (Kern)")
    print("  " + "-" * (7 + 25 * len(a.zyklen) + 32))

    abstand = None
    tabelle = []
    for i in range(12):
        zeile = {"monat": MONATSNAMEN[i], "index": i, "werte": {}}
        print(f"  {MONATSNAMEN[i]:<7}", end="")
        for z in a.zyklen:
            von, bis = monatsfenster(z, i)
            try:
                ndvi, n = hole(lauf, aoi, von, bis)
            except RuntimeError as f:
                print(f"{'—':>12}{'—':>8}{'—':>5}", end="")
                continue
            if abstand is None:
                abstand = pa.abstandsraster(bbox_utm, ndvi.shape, x0, y0)
                kern = abstand <= a.kern_m
                ring = (abstand >= a.ring_von_m) & (abstand <= a.ring_bis_m)
            gilt = np.isfinite(ndvi) & (n >= 1)
            mk = float(np.median(ndvi[kern & gilt])) if (kern & gilt).sum() > 50 else float("nan")
            mr = float(np.median(ndvi[ring & gilt])) if (ring & gilt).sum() > 50 else float("nan")
            mn = float(np.median(n[kern]))
            zeile["werte"][str(z)] = {"kern": round(mk, 4), "ring": round(mr, 4),
                                      "beobachtungen": mn,
                                      "von": von.isoformat(), "bis": bis.isoformat()}
            print(f"{mk:>12.3f}{mr:>8.3f}{mn:>5.0f}", end="")
        tabelle.append(zeile)
        print()

    # Gipfel je Zyklus und Anteil des benutzten Fensters
    print("\n  " + "-" * 60)
    ergebnis = {"erzeugt": nb.jetzt_utc(), "punkt": {"lat": a.lat, "lon": a.lon},
                "aufruf": ch.aufruf_protokoll(),
                "crs": f"EPSG:{epsg}", "aufloesung_m": a.aufloesung,
                "zyklen": a.zyklen, "monate": tabelle,
                "quellenvermerk": f"Contains modified Copernicus Sentinel data {date.today().year}"}
    for z in a.zyklen:
        werte = [(r["monat"], r["werte"].get(str(z), {}).get("kern"))
                 for r in tabelle if r["werte"].get(str(z), {}).get("kern") is not None]
        werte = [(m, v) for m, v in werte if v == v]
        if not werte:
            continue
        gipfel = max(werte, key=lambda x: x[1])
        tal = min(werte, key=lambda x: x[1])
        spanne = gipfel[1] - tal[1]
        # Wie hoch steht das benutzte Fenster (Feb + Mär + halber Apr)?
        fenster = [v for m, v in werte if m in ("Feb", "Mär", "Apr")]
        anteil = (np.mean(fenster) - tal[1]) / spanne if spanne > 0 else float("nan")
        print(f"  Zyklus {z}: Gipfel im {gipfel[0]} ({gipfel[1]:.3f}), "
              f"Tal im {tal[0]} ({tal[1]:.3f}), Spanne {spanne:.3f}")
        print(f"             Fenster Feb-Apr liegt bei {100*anteil:.0f} % "
              f"der Jahresamplitude")
        ergebnis[f"zyklus_{z}"] = {
            "gipfel_monat": gipfel[0], "gipfel_ndvi": round(gipfel[1], 4),
            "tal_monat": tal[0], "tal_ndvi": round(tal[1], 4),
            "amplitude": round(spanne, 4),
            "fenster_feb_apr_anteil": round(float(anteil), 3)}

    AUSGABE.mkdir(parents=True, exist_ok=True)
    ziel = AUSGABE / f"saisonprobe-{a.name}.json"
    ziel.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  geschrieben: {ziel.relative_to(REPO)}")
    if lauf.pu_je_aoi:
        print(f"  Processing Units: {sum(lauf.pu_je_aoi.values()):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
