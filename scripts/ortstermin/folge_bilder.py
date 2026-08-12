"""Bilder einer Ortstermin-Folge erzeugen (Draufsicht + Nahaufnahme).

Draufsicht  : eigenes wolkenbereinigtes Sentinel-2-Echtbild (CDSE, Median über
              ein Zeitfenster) — Pflichtvermerk „Enthält modifizierte
              Copernicus-Sentinel-Daten [Jahr]". Bewusst in nativer 10-m-
              Auflösung: die Grobheit IST Teil der Erzählung (vier Pixel).
Nahaufnahme : LGLN DOP20 (CC BY 4.0) über den offenen WMS — Vermerk
              „© LGLN, Niedersachsen, CC BY 4.0" gehört in die Bildunterschrift.

Die Bilder werden als WebP in den retteten-Mirror gelegt (Kopie, kein Hotlink —
E8: jede Folge muss auch ohne Geophora Sinn ergeben).

Aufruf (venv des NDVI-Batches):
    python folge_bilder.py --slug vier-pixel --lat 53.1462 --lon 8.2108
Optionen: --breite-km 6.4 (Draufsicht), --nah-m 320 (Nahaufnahme-Kante)
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

# Rechen-Bausteine des NDVI-Batches wiederverwenden (Token, Retry, PU-Konto)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ndvi"))
import ndvi_batch as nb            # noqa: E402
import echtbild_ersatz as eb       # noqa: E402

MIRROR = Path(__file__).resolve().parents[2] / "ftp-mirror-geophora"

DOP_WMS = "https://opendata.lgln.niedersachsen.de/doorman/noauth/dop_wms"
DOP_LAYER = "ni_dop20"

R_ERDE = 6378137.0


def zu_mercator(lon: float, lat: float) -> tuple[float, float]:
    x = math.radians(lon) * R_ERDE
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R_ERDE
    return x, y


def speichere_webp(im: Image.Image, ziel: Path, qualitaet: int = 82) -> int:
    ziel.parent.mkdir(parents=True, exist_ok=True)
    im.save(ziel, "WEBP", quality=qualitaet, method=6)
    kb = ziel.stat().st_size // 1024
    # Budget aus dem Masterprompt: je Bild höchstens ~300 KB
    if kb > 300:
        im.save(ziel, "WEBP", quality=70, method=6)
        kb = ziel.stat().st_size // 1024
    return kb


def draufsicht(lauf, lat: float, lon: float, breite_km: float, ziel: Path) -> None:
    """Sentinel-2-Echtbild in nativer Auflösung (kein Hochrechnen).

    Seit R1 (12.08.2026) auf dem metrischen 10-m-Gitter: Die lon/lat-bbox
    definiert nur noch den Ausschnitt; nb.process() projiziert sie in die
    UTM-Zone des Ortes und snappt aufs Gitter — ein Bildpunkt ist damit exakt
    10 × 10 m, die Pixelmaße ergeben sich aus dem Gitter (≈ breite_km · 100)."""
    hoehe_km = breite_km * 2 / 3
    dlon = breite_km / 111.32 / math.cos(math.radians(lat))
    dlat = hoehe_km / 110.57
    bbox = [lon - dlon / 2, lat - dlat / 2, lon + dlon / 2, lat + dlat / 2]
    bis = dt.date.today()
    von = bis - dt.timedelta(days=45)
    daten = nb.process(lauf, {"id": "folge", "bbox": bbox, "aufloesung_m": 10},
                       von, bis, eb.evalscript_echtbild(), "image/png",
                       aufloesung_m=10)
    with Image.open(io.BytesIO(daten)) as im:
        im = eb.tonwertkurve(im.convert("RGB"))
        kb = speichere_webp(im, ziel)
    print(f"Draufsicht : {im.size[0]}x{im.size[1]} px · {kb} KB · {von} bis {bis} "
          f"· 10 m je Pixel (UTM)")


def nahaufnahme(lat: float, lon: float, kante_m: float, ziel: Path) -> None:
    """LGLN-DOP20-Ausschnitt (20 cm) über den offenen WMS."""
    x, y = zu_mercator(lon, lat)
    # Mercator-Maßstabsfaktor: auf 53° N entspricht 1 m am Boden ~1.665 Kartenmetern
    f = 1 / math.cos(math.radians(lat))
    halb_b = kante_m * f / 2
    halb_h = halb_b * 2 / 3
    bbox = f"{x - halb_b},{y - halb_h},{x + halb_b},{y + halb_h}"
    breite_px = round(kante_m / 0.2)
    hoehe_px = round(breite_px * 2 / 3)
    q = urllib.parse.urlencode({
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": DOP_LAYER, "STYLES": "", "CRS": "EPSG:3857",
        "BBOX": bbox, "WIDTH": breite_px, "HEIGHT": hoehe_px,
        "FORMAT": "image/jpeg",
    })
    req = urllib.request.Request(DOP_WMS + "?" + q,
                                 headers={"User-Agent": "retteten.de Feldbuch (robert@retteten.de)"})
    with urllib.request.urlopen(req, timeout=60) as antwort:
        daten = antwort.read()
    with Image.open(io.BytesIO(daten)) as im:
        kb = speichere_webp(im.convert("RGB"), ziel)
    print(f"Nahaufnahme: {breite_px}x{hoehe_px} px · {kb} KB · LGLN DOP20 (CC BY 4.0)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slug", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--breite-km", type=float, default=6.4)
    p.add_argument("--nah-m", type=float, default=320.0)
    a = p.parse_args()

    # Feldbuch-Umzug E11 (2026-08-12): Folgen leben unter geophora.de/blog/.
    ziel_ordner = MIRROR / "blog" / a.slug / "bilder"

    class Args:  # Lauf erwartet die Argumente des NDVI-Batches
        aoi = None; jahr = None; force = False
        ausgabe = str(nb.STANDARD_AUSGABE); wochen = None; aufloesung = None
    lauf = nb.Lauf(Args())

    draufsicht(lauf, a.lat, a.lon, a.breite_km, ziel_ordner / "draufsicht.webp")
    nahaufnahme(a.lat, a.lon, a.nah_m, ziel_ordner / "nahaufnahme.webp")
    print(f"PU verbraucht: {sum(lauf.pu_je_aoi.values()):.2f}")


if __name__ == "__main__":
    main()
