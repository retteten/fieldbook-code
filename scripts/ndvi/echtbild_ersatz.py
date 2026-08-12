"""Echtbild-Composites aus eigenen Copernicus-Daten (Ersatz für EOX-Kacheln).

Warum es dieses Skript gibt (05.08.2026): Die bisher genutzten
Sentinel-2-cloudless-Kacheln von EOX sind ab Jahrgang 2018 **CC BY-NC-SA 4.0**
— am WMTS-GetCapabilities belegt. NC schließt die kommerzielle geophora.de aus,
SA würde abgeleitete Werke binden, und die EOX-Commercial-Lizenz verbietet die
Weitergabe ausdrücklich. Copernicus selbst erlaubt dagegen ausdrücklich
Vervielfältigung, Verbreitung, Bearbeitung und Kombination — also rechnen wir
die Bilder selbst.

Pflichtvermerk der Ergebnisse: „Enthält modifizierte Copernicus-Sentinel-Daten
[Jahr], verarbeitet über das Copernicus Data Space Ecosystem."

Aufruf (venv):
    python echtbild_ersatz.py eudr        # 7 Bilder des EUDR-Tools (Jahr 2020)
    python echtbild_ersatz.py szenen      # 15 Layer-Explorer-Echtbilder (aktuell)
    python echtbild_ersatz.py alles
Optionen: --jahr 2020 --nur <name> --trocken (nur zeigen, was passieren würde)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

from PIL import Image

# Rechen- und Netzwerk-Bausteine des NDVI-Batches wiederverwenden, damit
# Token-Handhabung, Retry und PU-Buchhaltung identisch bleiben.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndvi_batch as nb  # noqa: E402

MIRROR = Path(__file__).resolve().parents[2] / "ftp-mirror-geophora"
EUDR_DIR = MIRROR / "nachhaltigkeitsmanagement" / "assets" / "eudr"
LX_DIR = MIRROR / "layer-explorer"
SZENEN_JSON = Path(__file__).resolve().parent / "echtbild_szenen.json"

# EUDR-Tool: Region je Rohstoff (aus nachhaltigkeitsmanagement.js, coord=[lat,lon]).
# Kantenlänge in Grad so gewählt, dass Struktur erkennbar bleibt (~25 km).
EUDR = [
    ("kaffee", 12.67, 108.05), ("kakao", 5.78, -6.60), ("palmoel", 0.51, 101.45),
    ("soja", -12.55, -55.71), ("kautschuk", 8.90, 99.33), ("rind", -6.64, -51.99),
    ("holz", 2.00, 16.00),
]
EUDR_GRAD = 0.22          # ~24 km Kantenlänge
EUDR_GROESSE = (960, 754)  # exakt die Maße der bisherigen Bilder


def evalscript_echtbild() -> str:
    """Wolkenbereinigtes Echtbild (True Color) als Median über das Zeitfenster.

    Gleiche Maskenkonvention wie im Datenvertrag (SCL 3/8/9/10/11 + dataMask),
    damit die Bilder zur übrigen Kette passen. Ausgabe 8 Bit, weil das Ergebnis
    ein Anschauungsbild ist und kein Messraster.
    """
    return """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "SCL", "dataMask"] }],
    output: { bands: 3, sampleType: "UINT8" },
    mosaicking: "ORBIT"
  };
}
function median(a) {
  if (!a.length) return null;
  a.sort(function (x, y) { return x - y; });
  var m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}
function evaluatePixel(samples) {
  var r = [], g = [], b = [];
  for (var i = 0; i < samples.length; i++) {
    var s = samples[i];
    if (s.dataMask !== 1) continue;
    if (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 || s.SCL === 10 || s.SCL === 11) continue;
    r.push(s.B04); g.push(s.B03); b.push(s.B02);
  }
  var mr = median(r), mg = median(g), mb = median(b);
  if (mr === null) return [0, 0, 0];
  // 2.5x Verstärkung wie bei den False-Color-Bildern, dann auf 0..255 klemmen
  function stretch(v) { return Math.max(0, Math.min(255, Math.round(v * 2.5 * 255))); }
  return [stretch(mr), stretch(mg), stretch(mb)];
}"""


def evalscript_echtbild_einfach() -> str:
    """Übersichtsbilder über sehr große Gebiete (Deutschland/Europa).

    Ein Median über alle Orbits mehrerer Monate ist dort weder nötig noch
    machbar — die Anfrage lief in den Zeitablauf (beobachtet 05.08.2026). Für
    eine Übersichtskarte genügt die Mosaikierung nach geringster Bewölkung:
    CDSE wählt je Pixel die wolkenärmste Szene, wir maskieren nur noch grob.
    """
    return """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "dataMask"] }],
    output: { bands: 3, sampleType: "UINT8" },
    mosaicking: "SIMPLE"
  };
}
function evaluatePixel(s) {
  if (s.dataMask !== 1) return [0, 0, 0];
  function stretch(v) { return Math.max(0, Math.min(255, Math.round(v * 2.5 * 255))); }
  return [stretch(s.B04), stretch(s.B03), stretch(s.B02)];
}"""


def ist_grossflaechig(bbox: list[float]) -> bool:
    """Ab etwa Bundesland-übergreifender Ausdehnung wird der Median zu teuer."""
    return (bbox[2] - bbox[0]) > 5.0 or (bbox[3] - bbox[1]) > 5.0


def bbox_aus_mittelpunkt(lat: float, lon: float, grad: float) -> list[float]:
    """Quadratischer Ausschnitt in Grad — Längengrade werden mit der Breite
    gestaucht, damit das Bild nicht verzerrt."""
    dlon = grad / max(0.2, math.cos(math.radians(lat)))
    return [lon - dlon / 2, lat - grad / 2, lon + dlon / 2, lat + grad / 2]


def mercator_zu_grad(bbox3857: list[float]) -> list[float]:
    """EPSG:3857 → EPSG:4326 (die Szenen-bboxes des Explorers liegen in Mercator)."""
    R = 6378137.0
    w, s, o, n = bbox3857
    grad = lambda x: x / R * 180 / math.pi
    lat = lambda y: math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)
    return [grad(w), lat(s), grad(o), lat(n)]


def hole_bild(lauf, bbox: list[float], groesse: tuple[int, int], von, bis) -> bytes:
    aoi = {"id": "echtbild", "bbox": bbox, "aufloesung_m": 10}
    breite, hoehe = groesse
    return nb.process_roh(lauf, aoi, von, bis, evalscript_echtbild(), "image/png",
                          breite=breite, hoehe=hoehe)


def hole_bild_gekachelt(lauf, bbox: list[float], groesse: tuple[int, int],
                        von, bis, kachel_grad: float = 4.0) -> Image.Image:
    """Große Ausschnitte als Mosaik aus Einzelanfragen zusammensetzen.

    Ein kontinentaler wolkenfreier Composite ist in EINER Anfrage nicht zu
    haben: Der Dienst verarbeitet je Anfrage nur eine begrenzte Zahl an Szenen,
    der Rest bleibt leer (beobachtet 05.08.2026 — schwarze Streifen quer über
    Deutschland). Fertigprodukte wie EOX cloudless existieren genau deshalb —
    sind aber ab Jahrgang 2018 nicht-kommerziell lizenziert. Also rechnen wir
    das Mosaik selbst: Der Ausschnitt wird in Kacheln zerlegt, jede Kachel
    bekommt ihren eigenen wolkenmaskierten Median, danach werden sie
    zusammengesetzt.
    """
    w, s, o, n = bbox
    spalten = max(1, math.ceil((o - w) / kachel_grad))
    zeilen = max(1, math.ceil((n - s) / kachel_grad))
    ziel = Image.new("RGB", groesse)
    for zi in range(zeilen):
        for sp in range(spalten):
            tw = w + (o - w) * sp / spalten
            to = w + (o - w) * (sp + 1) / spalten
            tn = n - (n - s) * zi / zeilen
            ts = n - (n - s) * (zi + 1) / zeilen
            px_w = round(groesse[0] * (sp + 1) / spalten) - round(groesse[0] * sp / spalten)
            px_h = round(groesse[1] * (zi + 1) / zeilen) - round(groesse[1] * zi / zeilen)
            daten = nb.process_roh(
                lauf, {"id": "kachel", "bbox": [tw, ts, to, tn], "aufloesung_m": 10},
                von, bis, evalscript_echtbild(), "image/png",
                breite=max(16, px_w), hoehe=max(16, px_h))
            with Image.open(io.BytesIO(daten)) as kachel:
                ziel.paste(kachel.convert("RGB"),
                           (round(groesse[0] * sp / spalten),
                            round(groesse[1] * zi / zeilen)))
    return ziel


def tonwertkurve(im: Image.Image, gamma: float = 1.55, weiss: float = 0.92) -> Image.Image:
    """Mitteleuropäische Sommerszenen sind in reiner Reflexion dunkel: Dächer,
    Asphalt und Wald liegen bei 5–20 % Reflexion, ein linearer 2,5-facher
    Faktor lässt das Bild abgesoffen wirken. Deshalb eine Tonwertkurve —
    Weißpunkt auf das obere Perzentil, dann Gamma auf die Mitteltöne.
    Das verändert nur die Darstellung, nicht die zugrunde liegenden Messwerte
    (die liegen ohnehin in den GeoTIFFs des NDVI-Monitors).
    """
    import numpy as np
    a = np.asarray(im).astype("float32") / 255.0
    obergrenze = float(np.percentile(a[a > 0.02], 99.0)) if (a > 0.02).any() else 1.0
    a = np.clip(a / max(1e-3, obergrenze * weiss), 0, 1) ** (1.0 / gamma)
    return Image.fromarray((a * 255).astype("uint8"))


def speichere_jpg(ziel: Path, png_bytes: bytes, groesse: tuple[int, int]) -> int:
    """Als JPEG in der bisherigen Qualität ablegen (die Seiten erwarten .jpg)."""
    with Image.open(io.BytesIO(png_bytes)) as im:
        im = tonwertkurve(im.convert("RGB")).resize(groesse, Image.LANCZOS)
        ziel.parent.mkdir(parents=True, exist_ok=True)
        im.save(ziel, "JPEG", quality=78, optimize=True, progressive=True)
    return ziel.stat().st_size


def cmd_eudr(lauf, args):
    von, bis = nb.dt.date(args.jahr, 5, 1), nb.dt.date(args.jahr, 9, 30)
    for name, lat, lon in EUDR:
        if args.nur and args.nur != name:
            continue
        ziel = EUDR_DIR / f"{name}.jpg"
        bbox = bbox_aus_mittelpunkt(lat, lon, EUDR_GRAD)
        if args.trocken:
            print(f"  würde rechnen: {name} {bbox} {von}–{bis}")
            continue
        try:
            groesse = EUDR_GROESSE
            daten = hole_bild(lauf, bbox, groesse, von, bis)
            kb = speichere_jpg(ziel, daten, groesse) // 1024
            lauf.merke("eudr", name, f"ok ({kb} KB)")
        except Exception as fehler:
            lauf.merke("eudr", name, f"FEHLER: {fehler}")


def cmd_szenen(lauf, args):
    szenen = json.loads(SZENEN_JSON.read_text(encoding="utf-8"))
    von, bis = nb.dt.date(args.jahr, 5, 1), nb.dt.date(args.jahr, 9, 30)
    for z in szenen:
        if args.nur and args.nur != z["szene"]:
            continue
        bbox = mercator_zu_grad(z["bbox3857"])
        groesse = (z["w"], z["h"])
        ziel = LX_DIR / z["datei"]
        if args.trocken:
            print(f"  würde rechnen: {z['szene']} {['%.4f' % v for v in bbox]} {groesse}")
            continue
        try:
            if ist_grossflaechig(bbox):
                # Kontinentale Uebersicht: Mosaik aus Kacheln (s. Funktion)
                bild = hole_bild_gekachelt(lauf, bbox, groesse, von, bis)
                ziel.parent.mkdir(parents=True, exist_ok=True)
                tonwertkurve(bild).save(ziel, "JPEG", quality=78,
                                        optimize=True, progressive=True)
                kb = ziel.stat().st_size // 1024
            else:
                daten = hole_bild(lauf, bbox, groesse, von, bis)
                kb = speichere_jpg(ziel, daten, groesse) // 1024
            lauf.merke("szenen", z["szene"], f"ok ({kb} KB)")
        except Exception as fehler:
            lauf.merke("szenen", z["szene"], f"FEHLER: {fehler}")


def main():
    p = argparse.ArgumentParser(description="Echtbild-Composites statt EOX-Kacheln")
    p.add_argument("befehl", choices=["eudr", "szenen", "alles"])
    p.add_argument("--jahr", type=int, default=None,
                   help="Aufnahmejahr (Default: EUDR 2020, Szenen letztes volles Jahr)")
    p.add_argument("--nur", help="nur dieses Bild/diese Szene")
    p.add_argument("--trocken", action="store_true", help="nur anzeigen, nichts abrufen")
    args = p.parse_args()

    class Args:  # Lauf erwartet die Argumente des NDVI-Batches
        aoi = None; jahr = None; force = False
        ausgabe = str(nb.STANDARD_AUSGABE); wochen = None; aufloesung = None
    lauf = nb.Lauf(Args())

    if args.befehl in ("eudr", "alles"):
        args.jahr = args.jahr or 2020   # EUDR-Stichtagsjahr
        cmd_eudr(lauf, args)
    if args.befehl in ("szenen", "alles"):
        args.jahr = args.jahr or (nb.dt.date.today().year - 1)
        cmd_szenen(lauf, args)
    if not args.trocken:
        lauf.drucke_protokoll()
        sys.exit(1 if lauf.hat_fehler else 0)


if __name__ == "__main__":
    main()
