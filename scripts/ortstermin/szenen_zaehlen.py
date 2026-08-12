# -*- coding: utf-8 -*-
"""Wie viele Sentinel-2-Aufnahmen stecken in den Composites einer Folge?

Zählt je Zeitfenster die Sentinel-2-L2A-Produkte im öffentlichen OData-Katalog
des Copernicus Data Space Ecosystem (keine Anmeldung, keine PU) und dedupliziert
sie auf **Überflüge** (eindeutige Aufnahmezeitpunkte): Ein Überflug kann die
BBox mit zwei Kacheln schneiden und liegt nach dem Collection-1-Reprocessing
teils in mehreren Produktfassungen vor — beides darf nicht doppelt zählen.

Die Fenster kommen aus den Datendateien der Folge (Belegpflicht, Rezeptur § 8):
den Jahresfenstern des Trend-JSON und dem Zweiwochenfenster des
Fundstellen-JSON. Ergebnis-JSON enthält den wörtlichen Aufruf.

Aufruf — Publikationslauf „Vier Pixel" (13.08.2026):
    python scripts/ortstermin/szenen_zaehlen.py --aoi oldenburg \\
        --trend docs/daten/trend-oldenburg-2018-2025.json \\
        --fundstellen docs/daten/fundstellen-oldenburg-2026-W33.json \\
        --json docs/daten/szenen-oldenburg-2018-2026.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import requests

KATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
SENSING = re.compile(r"MSIL2A_(\d{8}T\d{6})_")


def polygon(bbox: list[float]) -> str:
    w, s, o, n = bbox
    return (f"SRID=4326;POLYGON(({w} {s},{o} {s},{o} {n},{w} {n},{w} {s}))")


def zaehle_fenster(bbox: list[float], von: str, bis: str) -> tuple[int, int]:
    """(Überflüge, Produkte) für ein Datumsfenster [von, bis] einschließlich."""
    ende = (dt.date.fromisoformat(bis) + dt.timedelta(days=1)).isoformat()
    f = (
        "Collection/Name eq 'SENTINEL-2' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
        f"OData.CSC.Intersects(area=geography'{polygon(bbox)}') and "
        f"ContentDate/Start ge {von}T00:00:00.000Z and "
        f"ContentDate/Start lt {ende}T00:00:00.000Z"
    )
    namen: list[str] = []
    url = KATALOG
    params = {"$filter": f, "$top": "1000", "$select": "Name"}
    while url:
        antwort = requests.get(url, params=params, timeout=60)
        antwort.raise_for_status()
        daten = antwort.json()
        namen += [p["Name"] for p in daten.get("value", [])]
        url, params = daten.get("@odata.nextLink"), None
    zeitpunkte = {m.group(1) for n in namen if (m := SENSING.search(n))}
    return len(zeitpunkte), len(namen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--aoi", default="oldenburg")
    p.add_argument("--trend", type=Path, required=True,
                   help="Trend-JSON mit fenster.je_jahr")
    p.add_argument("--fundstellen", type=Path,
                   help="Fundstellen-JSON mit dem Zweiwochenfenster")
    p.add_argument("--json", type=Path, required=True)
    a = p.parse_args()

    trend = json.loads(a.trend.read_text(encoding="utf-8"))
    bbox = trend.get("bbox")
    if bbox is None:
        # Der Trend-JSON trägt keine eigene bbox — die Fenster gelten für den
        # Folge-Ausschnitt; er steht im wörtlichen Aufruf des Trend-Laufs.
        m = re.search(r"--bbox (\S+)", trend.get("aufruf", ""))
        if not m:
            sys.exit("Keine bbox gefunden (weder Feld noch --bbox im aufruf).")
        bbox = [float(x) for x in m.group(1).split(",")]

    jahresfenster = {}
    for jahr, (von, bis) in sorted(trend["fenster"]["je_jahr"].items()):
        ueberfluege, produkte = zaehle_fenster(bbox, von, bis)
        jahresfenster[jahr] = {"von": von, "bis": bis,
                               "ueberfluege": ueberfluege, "produkte": produkte}
        print(f"  {jahr}: {von}..{bis}  {ueberfluege} Überflüge "
              f"({produkte} Produkte)")

    ergebnis = {
        "aoi": a.aoi,
        "bbox": bbox,
        "quelle": "CDSE OData-Katalog (öffentlich), Produkttyp S2MSI2A; "
                  "Überflüge = eindeutige Aufnahmezeitpunkte im Produktnamen "
                  "(dedupliziert über Kacheln und Produktfassungen)",
        "abfrage": KATALOG,
        "jahresfenster": jahresfenster,
        "ueberfluege_gesamt": sum(f["ueberfluege"]
                                  for f in jahresfenster.values()),
    }

    if a.fundstellen:
        fund = json.loads(a.fundstellen.read_text(encoding="utf-8"))
        zf = fund.get("zeitfenster") or fund.get("fenster")
        von, bis = zf["von"], zf["bis"]
        ueberfluege, produkte = zaehle_fenster(bbox, von, bis)
        ergebnis["zweiwochenfenster"] = {"von": von, "bis": bis,
                                         "ueberfluege": ueberfluege,
                                         "produkte": produkte}
        print(f"  Zweiwochenfenster {von}..{bis}: {ueberfluege} Überflüge "
              f"({produkte} Produkte)")

    ergebnis["aufruf"] = " ".join(sys.argv[:1] + sys.argv[1:]).replace(
        sys.argv[0], "python scripts/ortstermin/szenen_zaehlen.py")
    a.json.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"geschrieben: {a.json}")


if __name__ == "__main__":
    main()
