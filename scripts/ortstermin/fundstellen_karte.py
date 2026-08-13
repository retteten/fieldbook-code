"""Bilder für die Folge: Anomaliekarte, Dreiklang, Einzelkacheln.

    --was karte      Anomalie (aktuell − Fensterbaseline) über dem Ausschnitt,
                     Rückgang bernstein, Zugewinn grün, unauffällig Papier.
                     Die gefundenen Stellen werden als Kreise eingezeichnet —
                     bewusst als Kreise und nicht als Kästchen: es sind Hinweise,
                     keine Parzellen.
    --was dreiklang  Eine Fundstelle in drei Kacheln nebeneinander:
                     Luftbild · Referenzjahr · heute.
    --was kacheln    Dieselben drei Bilder EINZELN — für den Wischvergleich auf
                     der Seite, der die beiden NDVI-Kacheln übereinanderlegt.
                     Schreibt <ziel>-dop.webp, <ziel>-<jahr>.webp, <ziel>-jetzt.webp.

WICHTIG: „heute" kommt aus `aktuell-analyse.tif` (20 m), nicht aus `aktuell.tif`
(10 m) — beide NDVI-Kacheln müssen dasselbe Gitter haben, sonst liest sich der
Auflösungssprung wie eine Veränderung.

Einzelpixel-Hinweislayer (Beschluss 12.08.2026, Default an bei --was karte):
Einzelzellen über der Schwelle, die zu keinem Cluster der Belegschwelle
gehören, erscheinen als blasse kleine Punkte (Deckkraft ~0,35). Sie fließen
nie in Zählungen ein. Legenden-Chip-Text (Vorgabe für die Folge): „Blasse
Punkte: Hinweise unterhalb der Belegschwelle (Einzelzellen — können Rauschen
oder Registrierungsversatz sein)".

Ausgabe als WebP (Budget ~300 KB je Bild, wie folge_bilder.py).

Aufruf (venv des NDVI-Batches) — echter Publikationslauf der Trendkarte
„Vier Pixel" (Legende −0,05 … +0,05 je Jahr, Kreise für alle Top-20 im
Ausschnitt — es zeichnen sich nur die, die wirklich im Bild liegen —,
960 px breit, Ausschnitt = Draufsicht aus folge_bilder.py,
s. docs/methodik-ortstermin-fundstellen.md § 4.4a):
    python scripts/ortstermin/fundstellen_karte.py --was karte --quelle trend \\
        --json docs/daten/trend-oldenburg-2020-2025.json \\
        --ausschnitt 8.162872,53.126906,8.258728,53.165494 \\
        --spanne 0.05 --markiere 20 --breite-px 960 \\
        --ziel ftp-mirror-geophora/blog/vier-pixel/bilder/trend.webp
Weitere Modi:
    python fundstellen_karte.py --was dreiklang --json fundstellen.json --nr 1 \\
        --ziel .../bilder/dreiklang.webp
    python fundstellen_karte.py --was kacheln --json fundstellen.json --nr 15 \\
        --ziel .../bilder/fliegerhorst.webp
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

BASIS = Path(__file__).resolve().parents[2]
TILES = BASIS / "ftp-mirror-geophora" / "tiles" / "ndvi"

sys.path.insert(0, str(BASIS / "scripts" / "ndvi"))
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS, R1)
# Theil-Sen ist Hausstandard (R3'); dieselbe Steigungsquelle wie ndvi_trend.py,
# damit Karte und Zahlen der Folge nie auseinanderlaufen koennen.
from ndvi_trend import theil_sen  # noqa: E402

DOP_WMS = "https://opendata.lgln.niedersachsen.de/doorman/noauth/dop_wms"
DOP_LAYER = "ni_dop20"
KOPFZEUG = {"User-Agent": "retteten.de Feldbuch (robert@retteten.de)"}
R_ERDE = 6378137.0

PAPIER = (246, 244, 238)
PINIE = (22, 48, 42)
BERNSTEIN = (232, 161, 60)
GRUEN = (44, 138, 107)

# NDVI-Farbskala des Monitors (aois.json) für die Dreiklang-Kacheln
SKALA = [(-0.2, (148, 114, 84)), (0.15, (212, 193, 139)), (0.35, (206, 199, 126)),
         (0.55, (150, 178, 88)), (0.72, (60, 138, 58)), (0.9, (4, 62, 28))]


def speichere(im: Image.Image, ziel: Path, budget_kb: int = 300) -> None:
    ziel.parent.mkdir(parents=True, exist_ok=True)
    for q in (86, 78, 68, 58):
        im.save(ziel, "WEBP", quality=q, method=6)
        if ziel.stat().st_size // 1024 <= budget_kb:
            break
    print(f"{ziel}  {im.size[0]}x{im.size[1]} px · {ziel.stat().st_size // 1024} KB")


def ndvi_farben(a: np.ndarray) -> np.ndarray:
    stuetz = np.array([s[0] for s in SKALA])
    farben = np.array([s[1] for s in SKALA], dtype="float64")
    flach = np.clip(np.nan_to_num(a, nan=-0.2), -0.2, 0.9)
    return np.stack([np.interp(flach, stuetz, farben[:, k]) for k in range(3)],
                    axis=-1).astype("uint8")


RAUSCHKERN = 0.10  # Anteil der Spanne um die Null, der Papier bleibt
FARBGAMMA = 1.2    # >1: der Verlauf setzt nach dem Kern sanft statt sofort ein


def anomalie_farben(d: np.ndarray, spanne: float) -> np.ndarray:
    """Divergierend in den Hausfarben: Rückgang bernstein, Zugewinn grün.

    Seit 13.08.2026 mit Rauschkern (Roberts Anmerkung: winzige Steigungen
    wirkten schon leicht grün, wo das keinen Sinn ergibt): Werte unter
    RAUSCHKERN·Spanne bleiben ungefärbt — bei der Trendkarte (--spanne 0.05)
    also ±0,005/Jahr —, danach steigt der Verlauf leicht progressiv an.
    Der Kern gehört als Satz in die Legende der Folge (Rezeptur § 3)."""
    t = np.clip(np.nan_to_num(d, nan=0.0) / spanne, -1, 1)
    staerke = np.clip((np.abs(t) - RAUSCHKERN) / (1 - RAUSCHKERN), 0, 1) ** FARBGAMMA
    aus = np.zeros(d.shape + (3,), dtype="float64")
    for k in range(3):
        aus[..., k] = np.where(
            t < 0,
            PAPIER[k] + (BERNSTEIN[k] - PAPIER[k]) * staerke,
            PAPIER[k] + (GRUEN[k] - PAPIER[k]) * staerke)
    return aus.astype("uint8")


def lies(pfad: Path, fenster=None) -> np.ndarray:
    with rasterio.open(pfad) as ds:
        return ds.read(1, window=fenster, boundless=fenster is not None,
                       fill_value=float("nan")).astype("float64")


def fenster_um(pfad: Path, lat: float, lon: float, kante_m: float):
    """Lesefenster um einen Punkt — metrisch im Raster-CRS (R1), keine
    Grad-Naeherung mehr."""
    with rasterio.open(pfad) as ds:
        ch.pruefe_metrisch(ds, pfad.name)
        x0, y0 = ch.punkt_nach(ds.crs, lon, lat)
        halb = kante_m / 2
        return rasterio.windows.from_bounds(x0 - halb, y0 - halb,
                                            x0 + halb, y0 + halb, ds.transform)


def dop_kachel(lat: float, lon: float, kante_m: float, px: int) -> Image.Image:
    x = math.radians(lon) * R_ERDE
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R_ERDE
    halb = kante_m / math.cos(math.radians(lat)) / 2
    q = urllib.parse.urlencode({
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": DOP_LAYER, "STYLES": "", "CRS": "EPSG:3857",
        "BBOX": f"{x - halb},{y - halb},{x + halb},{y + halb}",
        "WIDTH": px, "HEIGHT": px, "FORMAT": "image/jpeg",
    })
    with urllib.request.urlopen(
            urllib.request.Request(f"{DOP_WMS}?{q}", headers=KOPFZEUG), timeout=60) as antwort:
        return Image.open(io.BytesIO(antwort.read())).convert("RGB")



CACHE = Path(__file__).resolve().parents[2] / "scripts" / "ndvi" / "cache"


def feld(a, fenster=None) -> np.ndarray:
    """Das darzustellende Feld — je nach --quelle.

    `anomalie` : aktuell − Fensterbaseline (die Wetterfrage, zwei Wochen)
    `trend`    : Theil-Sen-Steigung je Pixel über die Peak-Season-Composites
                 der Jahre (die Jahrzehntfrage; Hausstandard R3′ — identisch
                 mit ndvi_trend.py, das die Fundstellen-JSON liefert).
                 Einheit: NDVI je Jahr.
    """
    ordner = TILES / a.aoi
    if a.quelle == "anomalie":
        return (lies(ordner / "aktuell-analyse.tif", fenster)
                - lies(ordner / "baseline-fenster.tif", fenster))
    jahre = [int(j) for j in a.jahre.split(",")]
    stapel = np.stack([lies(CACHE / a.aoi / f"jahr-{j}.tif", fenster) for j in jahre])
    gueltig = np.isfinite(stapel).all(axis=0)
    return np.where(gueltig, theil_sen(stapel, jahre), np.nan)


def einzelzellen(maske: np.ndarray, min_zellen: int) -> np.ndarray:
    """Zellen, deren zusammenhaengender Fleck (8er-Nachbarschaft) KLEINER als
    min_zellen ist — der Stoff des Einzelpixel-Hinweislayers (12.08.2026)."""
    besucht = np.zeros_like(maske, dtype=bool)
    klein = np.zeros_like(maske, dtype=bool)
    hoehe, breite = maske.shape
    for start in np.argwhere(maske):
        r0, c0 = int(start[0]), int(start[1])
        if besucht[r0, c0]:
            continue
        stapel, zellen = deque([(r0, c0)]), []
        besucht[r0, c0] = True
        while stapel:
            r, c = stapel.popleft(); zellen.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < hoehe and 0 <= cc < breite
                            and maske[rr, cc] and not besucht[rr, cc]):
                        besucht[rr, cc] = True
                        stapel.append((rr, cc))
        if len(zellen) < min_zellen:
            for r, c in zellen:
                klein[r, c] = True
    return klein


def karte(a, daten: dict) -> Image.Image:
    ordner = TILES / a.aoi
    with rasterio.open(ordner / "aktuell-analyse.tif") as ds:
        ch.pruefe_metrisch(ds, "aktuell-analyse.tif")
        raster_crs, transform = ds.crs, ds.transform
    # Der Ausschnitt kommt als lon/lat (Notizbuch-Schreibweise) und wird ins
    # metrische Raster-CRS projiziert; gezeichnet und gemessen wird in Metern.
    w, s, o, n = ch.bbox_nach(raster_crs, [float(v) for v in a.ausschnitt.split(",")])
    fenster = rasterio.windows.from_bounds(w, s, o, n, transform)

    roh = feld(a, fenster)
    bild = Image.fromarray(anomalie_farben(roh, a.spanne))

    massstab = a.breite_px / bild.size[0]
    bild = bild.resize((a.breite_px, int(round(bild.size[1] * massstab))), Image.NEAREST)
    zeichner = ImageDraw.Draw(bild, "RGBA")
    m_pro_px = (o - w) / bild.size[0]   # Meter je Anzeigepixel — exakt (R1)

    # --- Einzelpixel-Hinweislayer (Beschluss 12.08.2026) -------------------
    # Zellen ueber der Schwelle, die zu KEINEM Cluster der Belegschwelle
    # gehoeren: blasse kleine Punkte, Deckkraft ~0,35. Nie in Zaehlungen.
    # Legenden-Chip-Text (Vorgabe fuer die Folge): "Blasse Punkte: Hinweise
    # unterhalb der Belegschwelle (Einzelzellen — koennen Rauschen oder
    # Registrierungsversatz sein)".
    if a.hinweise:
        schwellen = daten.get("schwellen", {})
        if a.quelle == "trend":
            schwelle = float(schwellen.get("fallend_je_jahr", -0.02))
        else:
            schwelle = -abs(float(schwellen.get("anomalie", -0.10)))
        min_zellen = int(schwellen.get("min_zellen",
                                       schwellen.get("min_flaeche_px", 4)))
        with np.errstate(invalid="ignore"):
            einzel = einzelzellen(np.isfinite(roh) & (roh <= schwelle),
                                  min_zellen)
        zellen = np.argwhere(einzel)
        for r_z, c_z in zellen:
            x = (c_z + 0.5) / roh.shape[1] * bild.size[0]
            y = (r_z + 0.5) / roh.shape[0] * bild.size[1]
            zeichner.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5],
                             fill=PINIE + (90,))
        print(f"Hinweislayer: {len(zellen)} Einzelzellen unter der "
              f"Belegschwelle (< {min_zellen} Zellen, Schwelle {schwelle:+.2f})"
              f" als blasse Punkte — zaehlen nirgends mit")

    gezeichnet = 0
    for f in daten["fundstellen"][:a.markiere]:
        # Fundstellen außerhalb des Ausschnitts überspringen — ein halber Kreis
        # am Bildrand behauptet etwas, das nicht im Bild ist.
        fx, fy = ch.punkt_nach(raster_crs, f["lon"], f["lat"])
        if not (w <= fx <= o and s <= fy <= n):
            continue
        px = (fx - w) / (o - w) * bild.size[0]
        py = (n - fy) / (n - s) * bild.size[1]
        # Kreisradius aus der echten Fundstellenflaeche, in Metern je Pixel —
        # geklemmt, damit kleine Funde sichtbar und grosse nicht theatralisch
        # werden.
        r = max(9.0, min(math.sqrt(f["flaeche_m2"] / math.pi) / m_pro_px, 26.0))
        zeichner.ellipse([px - r, py - r, px + r, py + r],
                         outline=PINIE + (235,), width=2)
        gezeichnet += 1
    print(f"Kreise gezeichnet: {gezeichnet} von {min(a.markiere, len(daten['fundstellen']))} "
          f"(Rest liegt ausserhalb des Ausschnitts)")
    return bild


def drei_kacheln(a, f: dict, px: int) -> list[Image.Image]:
    """Luftbild · Referenzjahr · heute — beide NDVI-Kacheln auf dem Analysegitter."""
    ordner = TILES / a.aoi
    aus = [dop_kachel(f["lat"], f["lon"], a.kante_m, px)]
    for datei in (f"composite-{a.referenzjahr}.tif", "aktuell-analyse.tif"):
        pfad = ordner / datei
        # Nicht jedes Jahr liegt als Composite im Mirror; der Rechen-Cache hat
        # dieselben Peak-Season-Composites als jahr-<jahr>.tif.
        if not pfad.exists():
            pfad = CACHE / a.aoi / f"jahr-{a.referenzjahr}.tif"
        aus.append(Image.fromarray(ndvi_farben(
            lies(pfad, fenster_um(pfad, f["lat"], f["lon"], a.kante_m))
        )).resize((px, px), Image.NEAREST))
    # ASCII in der Konsole: die Windows-Codepage kann keine Pfeile.
    # Die Zeile vertraegt beide JSON-Formate: Anomalie (ndvi_referenz/ndvi_jetzt/
    # anomalie/persistent) und Trend (ndvi_start/ndvi_ende/steigung_je_jahr) —
    # seit der Neuberechnung (12.08.2026) speist das Trend-JSON den Dreiklang.
    ref = f.get("ndvi_referenz", f.get("ndvi_start", float("nan")))
    jetzt = f.get("ndvi_jetzt", f.get("ndvi_ende", float("nan")))
    zusatz = (f"Anomalie {f['anomalie']:+.2f} | persistent {f['persistent']:.0%}"
              if "anomalie" in f
              else f"Steigung {f.get('steigung_je_jahr', float('nan')):+.3f}/Jahr")
    print(f"  {f['lat']:.5f}, {f['lon']:.5f} | {f['flaeche_m2'] / 10000:.2f} ha | "
          f"NDVI {ref:.2f} -> {jetzt:.2f} | {zusatz}")
    return aus


def dreiklang(a, daten: dict) -> Image.Image:
    """Eine Zeile je Fundstelle: Luftbild · Referenzjahr · heute."""
    nummern = [int(v) for v in str(a.nr).split(",")]
    px = a.breite_px // 3 - 8
    bild = Image.new("RGB", (3 * px + 16, len(nummern) * (px + 8) - 8), PAPIER)
    for zeile, nr in enumerate(nummern):
        print(f"Fundstelle {nr}:")
        for i, k in enumerate(drei_kacheln(a, daten["fundstellen"][nr - 1], px)):
            bild.paste(k, (i * (px + 8), zeile * (px + 8)))
    return bild


def kacheln(a, daten: dict) -> None:
    """Die drei Kacheln einzeln — Bausteine eines Wischvergleichs auf der Seite."""
    nr = int(str(a.nr).split(",")[0])
    print(f"Fundstelle {nr}:")
    bilder = drei_kacheln(a, daten["fundstellen"][nr - 1], a.breite_px)
    for teil, bild in zip(("dop", str(a.referenzjahr), "jetzt"), bilder):
        speichere(bild, a.ziel.with_name(f"{a.ziel.stem}-{teil}{a.ziel.suffix}"))


def raster_reihe(a) -> Image.Image:
    """Dieselbe Anomalie auf mehreren Gittern — der Beleg für die Skalenprobe.

    Zeigt, was ein gröberer Sensor aus denselben Daten macht: Kleinteiliges
    verschwindet, weil es weggemittelt wird. Die Reihe ist das Bild zu der
    Tabelle in docs/format-ortstermin-serie.md § 3.3.
    """
    ordner = TILES / a.aoi
    with rasterio.open(ordner / "aktuell-analyse.tif") as ds:
        ch.pruefe_metrisch(ds, "aktuell-analyse.tif")
        raster_crs, transform = ds.crs, ds.transform
    w, s, o, n = ch.bbox_nach(raster_crs, [float(v) for v in a.ausschnitt.split(",")])
    fenster = rasterio.windows.from_bounds(w, s, o, n, transform)
    roh = feld(a, fenster)

    faktoren = [int(v) for v in str(a.faktoren).split(",")]
    kachel_b = (a.breite_px - 8 * (len(faktoren) - 1)) // len(faktoren)
    teile = []
    for f in faktoren:
        grob = roh if f == 1 else vergroebern(roh, f)
        bild = Image.fromarray(anomalie_farben(grob, a.spanne))
        hoehe = int(round(bild.size[1] * kachel_b / bild.size[0]))
        teile.append((f, bild.resize((kachel_b, hoehe), Image.NEAREST)))
        print(f"  Gitter {f * 20:4d} m: {grob.shape[1]}x{grob.shape[0]} Zellen")

    hoehe = max(t.size[1] for _, t in teile)
    reihe = Image.new("RGB", (len(teile) * kachel_b + 8 * (len(teile) - 1), hoehe), PAPIER)
    for i, (_, t) in enumerate(teile):
        reihe.paste(t, (i * (kachel_b + 8), 0))
    return reihe


def vergroebern(a: np.ndarray, faktor: int) -> np.ndarray:
    """Blockmittel — simuliert einen gröberen Sensor (wie in skalenprobe.py)."""
    h = a.shape[0] // faktor * faktor
    b = a.shape[1] // faktor * faktor
    block = a[:h, :b].reshape(h // faktor, faktor, b // faktor, faktor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(block, axis=(1, 3))


def ndvi_paar(a) -> None:
    """Der ganze Ausschnitt zweimal als NDVI: Referenzjahr und heute.

    Für den Wischvergleich über der Stadt. Beide aus dem Analysegitter (20 m),
    beide mit derselben Farbskala und demselben Ausschnitt — sonst vergleicht
    der Wischer zwei verschiedene Dinge.
    """
    ordner = TILES / a.aoi
    for teil, datei in ((str(a.referenzjahr), f"composite-{a.referenzjahr}.tif"),
                        ("jetzt", "aktuell-analyse.tif")):
        pfad = ordner / datei
        with rasterio.open(pfad) as ds:
            ch.pruefe_metrisch(ds, datei)
            w, s, o, n = ch.bbox_nach(ds.crs,
                                      [float(v) for v in a.ausschnitt.split(",")])
            fenster = rasterio.windows.from_bounds(w, s, o, n, ds.transform)
        bild = Image.fromarray(ndvi_farben(lies(pfad, fenster)))
        hoehe = int(round(bild.size[1] * a.breite_px / bild.size[0]))
        bild = bild.resize((a.breite_px, hoehe), Image.NEAREST)
        speichere(bild, a.ziel.with_name(f"{a.ziel.stem}-{teil}{a.ziel.suffix}"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--was", choices=["karte", "dreiklang", "kacheln", "ndvi-paar",
                                     "raster-reihe"], required=True)
    p.add_argument("--json", type=Path)   # bei --was ndvi-paar nicht nötig
    p.add_argument("--ziel", type=Path, required=True)
    p.add_argument("--aoi", default="oldenburg")
    p.add_argument("--referenzjahr", type=int, default=2021)
    p.add_argument("--ausschnitt", default="8.155,53.105,8.285,53.185", help="W,S,O,N")
    p.add_argument("--quelle", choices=["anomalie", "trend"], default="anomalie",
                   help="anomalie = zwei Wochen gegen Normalwert; trend = Steigung über Jahre")
    p.add_argument("--jahre", default="2020,2021,2022,2023,2024,2025",
                   help="Jahre des Trends (nur bei --quelle trend)")
    p.add_argument("--spanne", type=float, default=0.30, help="Spanne der Farbskala")
    p.add_argument("--markiere", type=int, default=12)
    p.add_argument("--breite-px", type=int, default=1000)
    p.add_argument("--faktoren", default="1,3,10",
                   help="Blockfaktoren der Rasterreihe (1 = 20 m, 3 = 60 m, 10 = 200 m)")
    p.add_argument("--nr", default="1",
                   help="Fundstelle(n) für den Dreiklang, komma-getrennt (je eine Zeile)")
    p.add_argument("--kante-m", type=float, default=160.0)
    # Einzelpixel-Hinweislayer: Default AN für Karten mit Clusterregel
    # (Beschluss 12.08.2026); Legenden-Chip-Text s. Modul-Docstring.
    p.add_argument("--hinweise", dest="hinweise", action="store_true",
                   default=True,
                   help="Einzelzellen unter der Belegschwelle als blasse "
                        "Punkte zeigen (Default an, nur --was karte)")
    p.add_argument("--ohne-hinweise", dest="hinweise", action="store_false",
                   help="Hinweislayer abschalten")
    a = p.parse_args()

    if a.was == "ndvi-paar":
        ndvi_paar(a)
        return
    if a.was == "raster-reihe":
        speichere(raster_reihe(a), a.ziel)
        return
    daten = json.loads(a.json.read_text(encoding="utf-8"))
    if a.was == "kacheln":
        kacheln(a, daten)
    else:
        speichere(karte(a, daten) if a.was == "karte" else dreiklang(a, daten), a.ziel)


if __name__ == "__main__":
    main()
