"""Relief-Ebenen aus dem Copernicus-DEM (Ersatz für die EOX-Terrain-Kacheln).

Warum es dieses Skript gibt (06.08.2026): Der Layer-Explorer lieferte in den
Szenen Deutschland, Hamburg und Alpen eine Relief-Ebene aus, deren Kacheln vom
EOX-Dienst stammten (`tiles.maps.eox.at`, Layer `terrain_3857`). Für diese Layer
steht im EOX-GetCapabilities **keine** Lizenz — nur eine Credit-Zeile. Der eigene
Konnektor-Katalog führt sie deshalb selbst als `license: unbelegt` mit
`redistribution: nein`, während der Mirror die Bilder an jeden Besucher
auslieferte. Das ist Weitergabe von etwas Unbelegtem, und damit genau der Fall,
den `docs/recht-und-lizenzen.md` R3 ausschließt.

Das Copernicus-DEM (GLO-30) fällt dagegen unter die Copernicus-Datenpolitik:
Vervielfältigung, Verbreitung und Bearbeitung sind ausdrücklich erlaubt. Es steht
im Quellenregister § 3.1 als freigegeben.

Ein Hillshade lässt sich nicht im Evalscript rechnen — dort wird pixelweise
ausgewertet, ohne Zugriff auf die Nachbarpixel. Also holen wir die Höhen als
Float-GeoTIFF und schattieren lokal.

Pflichtvermerk der Ergebnisse: „Enthält modifizierte Copernicus-Daten [Jahr]
(Copernicus DEM GLO-30), eigene Schummerung."

Aufruf (venv):
    python relief_ersatz.py                 # alle drei Szenen
    python relief_ersatz.py --nur alpen
    python relief_ersatz.py --trocken       # nur zeigen, was passieren würde
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Token-Handhabung, Retry und PU-Buchhaltung des NDVI-Batches wiederverwenden.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndvi_batch as nb  # noqa: E402

MIRROR = Path(__file__).resolve().parents[2] / "ftp-mirror-geophora"
LX_DIR = MIRROR / "layer-explorer"

# Szenen exakt wie in layer-explorer/erkunden.html (SCENES, bbox in EPSG:3857).
# z_faktor: Überhöhung. Bei 2 km Pixelkante wäre echtes Relief unsichtbar —
# die Schummerung ist ein Anschauungsbild, kein Höhenmodell.
SZENEN = [
    {"id": "de", "bbox3857": [-170100.0, 5929565.0, 2496677.0, 7429626.0],
     "groesse": (1280, 720), "z_faktor": 12.0, "staerke": 0.42},
    {"id": "hamburg", "bbox3857": [1106525.1, 7079451.4, 1116525.1, 7089451.4],
     "groesse": (1000, 1000), "z_faktor": 2.0, "staerke": 0.30},
    {"id": "alpen", "bbox3857": [1422021.0, 6023264.0, 1450021.0, 6051264.0],
     "groesse": (1000, 1000), "z_faktor": 1.2, "staerke": 0.50},
]

# Die Processing-API nimmt hier nur "image/tiff" — die im NDVI-Batch übliche
# Langform "image/tiff; application=geotiff" quittiert sie mit HTTP 400.
TYP_TIFF = "image/tiff"

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["DEM"], units: "METERS" }],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) { return [s.DEM]; }"""


def hole_hoehen(lauf, szene) -> np.ndarray:
    """Ein Float-GeoTIFF je Szene. Das DEM ist zeitlos — kein Zeitfilter, kein
    Mosaikieren, deshalb baut diese Funktion die Anfrage selbst, statt
    nb.process() zu nutzen (das setzt fest auf sentinel-2-l2a)."""
    breite, hoehe = szene["groesse"]
    anfrage = {
        "input": {
            "bounds": {
                "bbox": list(szene["bbox3857"]),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"},
            },
            "data": [{
                "type": "dem",
                "dataFilter": {"demInstance": "COPERNICUS_30"},
                "processing": {"upsampling": "BILINEAR", "downsampling": "BILINEAR"},
            }],
        },
        "output": {
            "width": breite, "height": hoehe,
            "responses": [{"identifier": "default",
                           "format": {"type": TYP_TIFF}}],
        },
        "evalscript": EVALSCRIPT,
    }
    antwort = None
    for versuch in (1, 2):
        kopf = {"Authorization": f"Bearer {lauf.token()}", "Accept": TYP_TIFF}
        antwort = nb.requests.post(nb.PROCESS_URL, headers=kopf, json=anfrage,
                                   timeout=300)
        if antwort.status_code == 200:
            break
        if versuch == 1 and antwort.status_code == 401:
            lauf._token = None
            continue
        if versuch == 1 and (antwort.status_code == 429 or antwort.status_code >= 500):
            nb.time.sleep(15)
            continue
        raise RuntimeError(f"Processing-API HTTP {antwort.status_code}: "
                           f"{antwort.text[:300]}")
    pu = antwort.headers.get("x-processingunits-spent")
    if pu:
        print(f"      PU verbraucht: {pu}")
    with nb.MemoryFile(antwort.content) as mf, mf.open() as ds:
        return ds.read(1).astype("float32")


def schummerung(hoehen: np.ndarray, meter_je_pixel: float,
                z_faktor: float, staerke: float) -> np.ndarray:
    """Standard-Hillshade, Sonne aus Nordwest (315°) bei 45° Höhe.

    Das Ergebnis wird um 128 zentriert, weil die Ebene im Explorer mit
    `mix-blend-mode: soft-light` über dem Echtbild liegt: Grauwert 128 lässt das
    Bild unverändert, dunkler vertieft, heller hebt an. Ein Hillshade von 0..255
    würde die Szene flächig aufhellen statt zu modellieren.
    """
    z = np.nan_to_num(hoehen, nan=0.0, posinf=0.0, neginf=0.0)
    # Meerespegel und Datenlücken liegen bei 0 — nicht als Senke schattieren.
    z = np.maximum(z, 0.0)
    dzdy, dzdx = np.gradient(z, meter_je_pixel)
    dzdx *= z_faktor
    dzdy *= z_faktor
    neigung = np.arctan(np.hypot(dzdx, dzdy))
    ausrichtung = np.arctan2(dzdy, -dzdx)
    zenit = math.radians(90.0 - 45.0)
    azimut = math.radians(360.0 - 315.0 + 90.0)
    hs = (math.cos(zenit) * np.cos(neigung)
          + math.sin(zenit) * np.sin(neigung) * np.cos(azimut - ausrichtung))
    hs = np.clip(hs, 0.0, 1.0)
    # Kontrast über die Perzentilspanne steuern, nicht über einen festen Faktor:
    # In den Alpen sättigt ein roher Hillshade zu Schwarz und Weiß, in Hamburg
    # bliebe er unsichtbar. `staerke` ist deshalb die Zielspanne zwischen dem
    # 2. und 98. Perzentil — der Rest wird geklemmt. Der Median landet auf
    # Neutralgrau, damit die Ebene das Echtbild moduliert statt es aufzuhellen.
    unten, mitte, oben = np.percentile(hs, [2.0, 50.0, 98.0])
    spanne = max(1e-6, float(oben - unten))
    grau = 0.5 + (hs - float(mitte)) / spanne * staerke
    return (np.clip(grau, 0.0, 1.0) * 255.0).astype("uint8")


def meter_je_pixel(szene) -> float:
    """Web-Mercator-Meter sind um 1/cos(Breite) gedehnt — für die Schummerung
    zählt die echte Bodenauflösung, sonst kippt die Überhöhung mit der Breite."""
    w, s, o, n = szene["bbox3857"]
    breite_px = szene["groesse"][0]
    mitte_y = (s + n) / 2.0
    lat = math.degrees(2 * math.atan(math.exp(mitte_y / 6378137.0)) - math.pi / 2)
    return (o - w) / breite_px * math.cos(math.radians(lat))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nur", help="nur diese Szene (de | hamburg | alpen)")
    p.add_argument("--trocken", action="store_true",
                   help="nur zeigen, was passieren würde")
    args = p.parse_args()

    szenen = [s for s in SZENEN if not args.nur or s["id"] == args.nur]
    if not szenen:
        return print(f"Keine Szene '{args.nur}'.") or 2

    if args.trocken:
        for s in szenen:
            print(f"  {s['id']:8s} {s['groesse'][0]}x{s['groesse'][1]}  "
                  f"{meter_je_pixel(s):8.1f} m/px  z={s['z_faktor']}  "
                  f"-> {LX_DIR / 'layers' / s['id'] / 'terrain.jpg'}")
        return 0

    class Args:  # Lauf erwartet die Argumente des NDVI-Batches
        aoi = None; jahr = None; force = False
        ausgabe = str(nb.STANDARD_AUSGABE); wochen = None; aufloesung = None

    lauf = nb.Lauf(Args())
    for s in szenen:
        ziel = LX_DIR / "layers" / s["id"] / "terrain.jpg"
        print(f"  {s['id']}: DEM holen ({s['groesse'][0]}x{s['groesse'][1]}, "
              f"{meter_je_pixel(s):.0f} m/px) …")
        hoehen = hole_hoehen(lauf, s)
        print(f"      Höhen {hoehen.min():.0f}–{hoehen.max():.0f} m")
        grau = schummerung(hoehen, meter_je_pixel(s), s["z_faktor"], s["staerke"])
        bild = Image.fromarray(grau, mode="L").convert("RGB")
        ziel.parent.mkdir(parents=True, exist_ok=True)
        bild.save(ziel, "JPEG", quality=82, optimize=True, progressive=True)
        print(f"      geschrieben: {ziel.relative_to(MIRROR)} "
              f"({ziel.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
