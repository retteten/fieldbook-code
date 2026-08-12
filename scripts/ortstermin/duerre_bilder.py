# -*- coding: utf-8 -*-
"""Die Bilder der Folge „Was gruen bleibt" (Ammerland).

Drei Bilder, drei Aufgaben:

  quadranten.webp   Die Karte: wo faellt im Duerresommer die Feuchte, wo das
                    Gruen, wo beides. Vier Farben, keine Deutung im Bild.
  achtsommer.webp   Der Kern in einer Grafik: acht Sommer, zwei Kurven. Man
                    sieht auf einen Blick, dass die Feuchte staerker
                    ausschlaegt als das Gruen — und dass 2022 im Gruen fast
                    nicht auffaellt.
  identitaet.webp   Das Gatter: vier Luftbildkacheln, zwei Baumschulen, zwei
                    Weiden — mit ihren Messwerten. Zeigt, was auf 20 cm
                    trivial und auf 20 m unmoeglich ist.

Alle Bilder ohne Text im Bild, wo es geht: Beschriftung gehoert ins Markup
(Legende, Bildunterschrift), damit sie uebersetzbar und vorlesbar bleibt.
Ausnahme sind die Achsenzahlen der Grafik — die sind Daten, kein Text.

Aufruf (venv des NDVI-Batches):
    python scripts/ortstermin/duerre_bilder.py --alle
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

BASIS = Path(__file__).resolve().parents[2]
CACHE = BASIS / "scripts" / "ndvi" / "cache" / "ammerland"

sys.path.insert(0, str(BASIS / "scripts" / "ndvi"))
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS, R1)
# Feldbuch-Umzug E11 (2026-08-12): Folgen leben unter geophora.de/blog/.
ZIEL = BASIS / "ftp-mirror-geophora" / "blog" / "duerre-ammerland" / "bilder"
DATEN = BASIS / "docs" / "daten" / "duerre-ammerland-2026-W33.json"
STICHPROBE = BASIS / "docs" / "daten" / "duerre-ammerland-stichprobe.json"

PAPIER = (246, 244, 238)
PINIE = (22, 48, 42)
BERNSTEIN = (232, 161, 60)
GRUEN = (44, 138, 107)

# Vier Quadrantenfarben. Bewusst nicht rot-gruen, sondern Helligkeit UND
# Farbton verschieden — so bleibt die Karte auch in Graustufen lesbar und
# fuer Rotgruenblinde brauchbar.
FARBEN = {
    "unauffaellig": (226, 224, 214),   # blass, tritt zurueck
    "nur_feuchte":  (196, 132, 44),    # bernstein: Feuchte faellt, Gruen nicht
    "beides":       (120, 58, 22),     # dunkelbraun: beides faellt
    "nur_gruen":    (108, 146, 176),   # blau: Gruen faellt, Feuchte nicht
}


def schrift(groesse: int):
    """Eine echte Schrift, keine Bitmap-Notloesung.

    Der erste Lauf zeichnete die Achsenzahlen mit ImageFont.load_default() —
    rund acht Pixel hoch und im WebP nicht mehr zu entziffern. Eine Grafik,
    deren Achse man nicht lesen kann, ist ein Bild von einer Grafik."""
    from PIL import ImageFont
    for kandidat in ("C:/Windows/Fonts/segoeui.ttf",
                     "C:/Windows/Fonts/arial.ttf",
                     "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(kandidat, groesse)
        except OSError:
            continue
    return ImageFont.load_default()


def lies(jahr: int, band: int = 1):
    with rasterio.open(CACHE / f"duerre-{jahr}.tif") as q:
        # Metrisches Gitter Pflicht (R1): Grad-Altbestaende laut abweisen.
        ch.pruefe_metrisch(q, f"duerre-{jahr}.tif")
        return q.read(band), q.transform, q.bounds, q.crs


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


def quadrantenkarte(daten: dict, breite: int = 960, hinweise: bool = True,
                    hinweis_min_zellen: int = 4) -> Image.Image:
    normal, duerre = daten["normaljahre"], daten["duerrejahre"]
    nn = np.nanmedian(np.stack([lies(j, 1)[0] for j in normal]), axis=0)
    nd = np.nanmedian(np.stack([lies(j, 1)[0] for j in duerre]), axis=0)
    mn = np.nanmedian(np.stack([lies(j, 2)[0] for j in normal]), axis=0)
    md = np.nanmedian(np.stack([lies(j, 2)[0] for j in duerre]), axis=0)
    d_ndvi, d_ndmi = nd - nn, md - mn
    land = np.isfinite(d_ndvi) & np.isfinite(d_ndmi) & (nn > 0.25)

    gf = land & (d_ndvi <= -daten["schwelle_ndvi"])
    ff = land & (d_ndmi <= -daten["schwelle_ndmi"])
    rgb = np.zeros(d_ndvi.shape + (3,), dtype="uint8")
    rgb[:] = np.array((236, 234, 228), dtype="uint8")     # ausserhalb: Papier
    masken = {"unauffaellig": land & ~gf & ~ff,
              "nur_feuchte": land & ~gf & ff,
              "beides": land & gf & ff,
              "nur_gruen": land & gf & ~ff}

    # Einzelpixel-Hinweislayer (Beschluss 12.08.2026): auffaellige Zellen,
    # die zu KEINEM Fleck von mindestens hinweis_min_zellen gehoeren, sind
    # Hinweise unterhalb der Belegschwelle — sie bekommen keine volle
    # Flaechenfarbe, sondern werden nach dem Resize als blasse Punkte
    # (Deckkraft ~0,35) gezeichnet. Sie fliessen NIE in Zaehlungen ein
    # (die Zaehlung wohnt in duerre_auswertung.py, nicht in diesem Bild).
    # Legenden-Chip-Text (Vorgabe fuer die Folge): "Blasse Punkte: Hinweise
    # unterhalb der Belegschwelle (Einzelzellen — koennen Rauschen oder
    # Registrierungsversatz sein)".
    hinweis_zellen = {}
    for schluessel in ("nur_feuchte", "beides", "nur_gruen"):
        if hinweise:
            klein = einzelzellen(masken[schluessel], hinweis_min_zellen)
            hinweis_zellen[schluessel] = np.argwhere(klein)
            masken[schluessel] = masken[schluessel] & ~klein
            masken["unauffaellig"] = masken["unauffaellig"] | klein

    for schluessel in ("unauffaellig", "nur_feuchte", "beides", "nur_gruen"):
        rgb[masken[schluessel]] = np.array(FARBEN[schluessel], dtype="uint8")

    orig_h, orig_b = d_ndvi.shape
    bild = Image.fromarray(rgb)
    hoehe = round(breite * bild.height / bild.width)
    bild = bild.resize((breite, hoehe), Image.NEAREST)

    if hinweise:
        deck = ImageDraw.Draw(bild, "RGBA")
        gesamt = 0
        for schluessel, zellen in hinweis_zellen.items():
            farbe = FARBEN[schluessel] + (90,)   # ~0,35 Deckkraft
            for r, c in zellen:
                x = (c + 0.5) / orig_b * breite
                y = (r + 0.5) / orig_h * hoehe
                deck.ellipse([x - 2, y - 2, x + 2, y + 2], fill=farbe)
            gesamt += len(zellen)
        print(f"  Hinweislayer: {gesamt} Einzelzellen unter der Belegschwelle "
              f"(< {hinweis_min_zellen} Zellen) als blasse Punkte")

    # Die neun von Hand bestaetigten Baumschulquartiere einzeichnen. Ohne sie
    # ist die Karte eine Duerrekarte fuer einen Landkreis; mit ihnen ist
    # sichtbar, dass die Baumschulen gerade NICHT dort liegen, wo die Karte
    # ausschlaegt. Das ist der Befund, nicht die Dekoration.
    if STICHPROBE.exists():
        probe = json.loads(STICHPROBE.read_text(encoding="utf-8"))
        _, _, grenzen, raster_crs = lies(duerre[0])
        z = ImageDraw.Draw(bild)
        for p in probe["punkte"]:
            if p["klasse"] != "baumschule":
                continue
            # Stichprobenpunkte liegen in lon/lat — ins metrische Raster-CRS
            # projizieren statt linear in Grad zu teilen (R1).
            px_x, px_y = ch.punkt_nach(raster_crs, p["lon"], p["lat"])
            x = (px_x - grenzen.left) / (grenzen.right - grenzen.left) * breite
            y = (grenzen.top - px_y) / (grenzen.top - grenzen.bottom) * hoehe
            # Zwei Ringe: erst hell, dann dunkel. Ein einzelner dunkler Ring
            # verschwindet ueber den braunen Parzellen — gemessen am fertigen
            # Bild, nicht vermutet.
            z.ellipse([x - 12, y - 12, x + 12, y + 12], outline=PAPIER, width=5)
            z.ellipse([x - 12, y - 12, x + 12, y + 12], outline=PINIE, width=3)
    return bild


def achtsommer(daten: dict, breite: int = 900, hoehe: int = 420) -> Image.Image:
    """Zwei Kurven ueber acht Sommer, gemeinsame Achse in Indexpunkten.

    Beide Indizes laufen von -1 bis 1 und sind damit direkt vergleichbar —
    genau das ist die Aussage der Grafik: Der eine faellt tiefer als der
    andere, und zwar im selben Massstab.
    """
    jahre = sorted(int(j) for j in daten["jahresmedian"])
    ndvi = [daten["jahresmedian"][str(j)]["ndvi"] for j in jahre]
    ndmi = [daten["jahresmedian"][str(j)]["ndmi"] for j in jahre]

    bild = Image.new("RGB", (breite, hoehe), PAPIER)
    z = ImageDraw.Draw(bild)
    f_achse = schrift(19)
    links, rechts, oben, unten = 74, breite - 18, 22, hoehe - 46
    y_min, y_max = 0.15, 0.82

    def x(i): return links + i * (rechts - links) / (len(jahre) - 1)
    def y(w): return unten - (w - y_min) / (y_max - y_min) * (unten - oben)

    # Duerresommer ZUERST hinterlegen. Andersherum malte das Feld die
    # Achsenzahlen zu — aus „0,8" wurde „0,", und zwar nur beim ersten und
    # mittleren Wert, was beim fluechtigen Blick auf das fertige WebP wie ein
    # Schriftfehler aussah statt wie eine Ueberdeckung.
    for j in daten["duerrejahre"]:
        i = jahre.index(j)
        halb = (rechts - links) / (len(jahre) - 1) / 2.6
        z.rectangle([max(x(i) - halb, links), oben,
                     min(x(i) + halb, rechts), unten], fill=(238, 231, 216))

    for wert in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        z.line([(links, y(wert)), (rechts, y(wert))], fill=(222, 219, 210))
        z.text((16, y(wert) - 11), f"{wert:.1f}".replace(".", ","),
               fill=(105, 113, 107), font=f_achse)

    for reihe, farbe, dick in ((ndvi, GRUEN, 4), (ndmi, BERNSTEIN, 4)):
        punkte = [(x(i), y(w)) for i, w in enumerate(reihe)]
        z.line(punkte, fill=farbe, width=dick, joint="curve")
        for px, py in punkte:
            z.ellipse([px - 5, py - 5, px + 5, py + 5], fill=farbe,
                      outline=PAPIER, width=2)

    for i, j in enumerate(jahre):
        z.text((x(i) - 21, unten + 12), str(j), fill=PINIE, font=f_achse)
    return bild


def wasserbilanz(breite: int = 900, hoehe: int = 430) -> Image.Image:
    """35 Sommer klimatische Wasserbilanz (DWD) als Balkenreihe.

    Der Bodenbeleg zur Satellitengrafik: Die Achtsommer-Kurven zeigen, wie die
    Vegetation reagiert; diese Balken zeigen, was das Wetter davor getan hat.
    Duerresommer in Bernstein, alle anderen zurueckhaltend — die Grafik soll
    eine Rangfrage beantworten (wie tief liegen 2018 und 2022?), keine
    Zeitreihenaesthetik pflegen.
    """
    daten = json.loads((BASIS / "docs" / "daten" /
                        "klima-ammerland-1991-2025.json").read_text(encoding="utf-8"))
    reihen = daten["sommer"]
    median = daten["median_kwb_mm"]

    bild = Image.new("RGB", (breite, hoehe), PAPIER)
    z = ImageDraw.Draw(bild)
    f_achse = schrift(19)
    links, rechts, oben, unten = 74, breite - 18, 24, hoehe - 46
    # Skala aus den Daten, auf 50er gerundet — eine feste Annahme (+40) hat im
    # ersten Lauf die nassen Sommer oben abgeschnitten.
    werte = [r["kwb_mm"] for r in reihen]
    y_min = math.floor(min(werte) / 50) * 50 - 10
    y_max = math.ceil(max(werte) / 50) * 50 + 10

    def x(i): return links + (i + 0.5) * (rechts - links) / len(reihen)
    def y(w): return unten - (w - y_min) / (y_max - y_min) * (unten - oben)

    for wert in range(int(y_min) // 50 * 50, int(y_max) + 1, 50):
        z.line([(links, y(wert)), (rechts, y(wert))],
               fill=(210, 207, 197) if wert == 0 else (226, 223, 214))
        z.text((10, y(wert) - 11), f"{wert:d}", fill=(105, 113, 107), font=f_achse)

    balken = (rechts - links) / len(reihen) * 0.62
    for i, r in enumerate(reihen):
        duerre = r["jahr"] in (2018, 2022)
        farbe = BERNSTEIN if duerre else (152, 165, 158)
        y0, y1 = sorted((y(0), y(r["kwb_mm"])))
        z.rectangle([x(i) - balken / 2, y0, x(i) + balken / 2, y1], fill=farbe)

    # Median als gestrichelte Linie — der Bezug, gegen den die Balken sprechen.
    for px in range(int(links), int(rechts), 14):
        z.line([(px, y(median)), (px + 7, y(median))], fill=PINIE, width=2)

    for i, r in enumerate(reihen):
        # 2020 faellt raus: es steht zwischen den beiden markierten Jahren,
        # und drei Zahlen auf vier Balkenbreiten sind keine Beschriftung mehr.
        if (r["jahr"] % 5 == 0 and r["jahr"] != 2020) or r["jahr"] in (2018, 2022):
            fett = r["jahr"] in (2018, 2022)
            z.text((x(i) - 21, unten + 12), str(r["jahr"]),
                   fill=PINIE if fett else (105, 113, 107), font=f_achse)
    return bild


def identitaetsbild(px: int = 300) -> Image.Image:
    """Vier Luftbildkacheln mit ihren Messwerten — zwei Baumschulen, zwei Weiden."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from duerre_kontaktbogen import dop_kachel

    probe = json.loads(STICHPROBE.read_text(encoding="utf-8"))
    nach = {p["kennung"]: p for p in probe["punkte"]}
    wahl = ["W", "X", "N", "Q"]           # 2 Baumschulen, 2 Gruenland
    rand, kopf = 8, 26
    bild = Image.new("RGB", (4 * (px + rand) + rand, px + kopf + 2 * rand),
                     PAPIER)
    z = ImageDraw.Draw(bild)
    for i, k in enumerate(wahl):
        p = nach[k]
        x = rand + i * (px + rand)
        bild.paste(dop_kachel(p["lat"], p["lon"], 260, px), (x, rand + kopf))
        z.text((x, rand + 6),
               f"NDVI {p['ndvi_normal']:.2f}   NDMI {p['ndmi_normal']:.2f}",
               fill=PINIE)
    return bild


def speichern(bild: Image.Image, name: str) -> None:
    ZIEL.mkdir(parents=True, exist_ok=True)
    pfad = ZIEL / name
    bild.save(pfad, "WEBP", quality=88, method=6)
    print(f"  {name}  {bild.width} x {bild.height}  "
          f"{pfad.stat().st_size / 1024:.0f} kB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alle", action="store_true")
    p.add_argument("--karte", action="store_true")
    p.add_argument("--grafik", action="store_true")
    p.add_argument("--identitaet", action="store_true")
    p.add_argument("--wasserbilanz", action="store_true")
    # Einzelpixel-Hinweislayer: Default AN (Beschluss 12.08.2026) — die Karte
    # traegt eine Belegschwelle, also gehoeren Einzelzellen als blasse
    # Hinweise gekennzeichnet statt als vollfarbige Flaechen behauptet.
    p.add_argument("--hinweise", dest="hinweise", action="store_true",
                   default=True,
                   help="Einzelzellen unter der Belegschwelle als blasse "
                        "Punkte zeigen (Default an)")
    p.add_argument("--ohne-hinweise", dest="hinweise", action="store_false",
                   help="Hinweislayer abschalten")
    a = p.parse_args()
    if not any((a.alle, a.karte, a.grafik, a.identitaet, a.wasserbilanz)):
        a.alle = True

    daten = json.loads(DATEN.read_text(encoding="utf-8"))
    print(f"Bilder nach {ZIEL.relative_to(BASIS)}")
    if a.alle or a.karte:
        speichern(quadrantenkarte(daten, hinweise=a.hinweise),
                  "quadranten.webp")
    if a.alle or a.grafik:
        speichern(achtsommer(daten), "achtsommer.webp")
    if a.alle or a.identitaet:
        speichern(identitaetsbild(), "identitaet.webp")
    if a.alle or a.wasserbilanz:
        speichern(wasserbilanz(), "wasserbilanz.webp")


if __name__ == "__main__":
    main()
