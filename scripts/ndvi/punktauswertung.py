"""Ergebnisoffene Auswertung eines Umkreises — Kern gegen eigenes Umland.

Die Frage, die dieses Skript beantwortet
----------------------------------------
"Was ist an diesem Punkt anders als drumherum, und seit wann?"

Es beantwortet NICHT "ist die Flaeche gruener geworden". Das waere die falsche
Frage: In einem Trockengebiet schwankt Vegetation mit dem Regen um ein
Vielfaches dessen, was Bewirtschaftung ausmacht. Wer nur die Zeitreihe einer
Flaeche ansieht, misst das Wetter und nennt es Wirkung.

Deshalb der raeumliche Vergleich: Kernkreis gegen einen Kontrollring
ringsherum. Beide bekommen denselben Regen, denselben Boden(grosstyp),
dieselbe Wolkenlage, dieselbe Sensorkalibrierung. Was uebrig bleibt, wenn man
das Umland abzieht, ist der Unterschied AN DIESEM ORT — und nur der ist
interpretierbar. Das ist eine Differenz von Differenzen im Raum, und sie
braucht keine Niederschlagsdaten, um den Regen herauszurechnen.

Was das Skript ausdruecklich nicht kann
---------------------------------------
- Es sagt nicht, WARUM ein Unterschied besteht. Zaun, Bohrloch, Brand,
  Bodenwechsel und ein Feldweg sehen aus der Umlaufbahn zunaechst gleich aus.
- Es findet nichts unterhalb der Pixelgroesse. Bei 20 m ist ein einzelner
  Baum unsichtbar.
- Ein Bruch in der Zeitreihe ist ein Datum, kein Ereignis.

Quellenvermerk der Ergebnisse (Pflicht):
    "Contains modified Copernicus Sentinel data <Jahr>"

Aufruf (venv):
    python punktauswertung.py --lat -23.973037 --lon 25.854650 --name botswana
    python punktauswertung.py ... --trocken     # nur zeigen, was passieren wuerde
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndvi_batch as nb  # noqa: E402
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS, R1)

REPO = Path(__file__).resolve().parents[2]
AUSGABE = REPO / "docs" / "daten"

# Die Processing-API nimmt nur die Kurzform (siehe hole_jahr).
TYP_TIFF = "image/tiff"

# Zwischenspeicher der Rohkacheln — lokal, nie deployen.
CACHE = Path(__file__).resolve().parent / "cache" / "punkt"


class SchlankerLauf:
    """Nur das, was nb.process() wirklich anfasst: Token und PU-Buchung."""

    def __init__(self):
        self._token_wert = None
        self._token_frist = 0.0
        self.pu_je_aoi: dict[str, float] = {}

    def token(self):
        import time
        if self._token_wert is None or time.monotonic() >= self._token_frist:
            cid, secret = nb.lade_zugangsdaten()
            self._token_wert, lebensdauer = nb.hole_token(cid, secret)
            self._token_frist = time.monotonic() + max(lebensdauer - 60, 60)
        return self._token_wert

    # nb.process() setzt bei 401 lauf._token = None
    @property
    def _token(self):
        return self._token_wert

    @_token.setter
    def _token(self, wert):
        self._token_wert = wert


def bbox_um(lat: float, lon: float, halbkante_m: float) -> list[float]:
    """Quadrat um einen Punkt in Grad (CRS84) — nur die GEBIETSDEFINITION.

    Die Grad-Naeherung ist hier unschaedlich: Sie beschreibt bloss, welcher
    Ausschnitt gerechnet werden soll. Gemessen wird seit R1 (12.08.2026) im
    metrischen UTM-Gitter — nb.aoi_gitter() projiziert diese bbox dorthin."""
    d_lat = halbkante_m / 110574.0
    d_lon = halbkante_m / (111320.0 * math.cos(math.radians(lat)))
    return [lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat]


def abstandsraster(bbox_utm: tuple[float, float, float, float],
                   form: tuple[int, int],
                   x0: float, y0: float) -> np.ndarray:
    """Entfernung jedes Pixelmittelpunkts zu (x0, y0), in Metern.

    Rechnet seit R1 direkt im metrischen Gitter: bbox_utm und der projizierte
    Mittelpunkt kommen aus nb.aoi_gitter() bzw. ch.punkt_nach() — echte
    euklidische Distanz, keine Breitengrad-Naeherung mehr."""
    hoehe, breite = form
    west, sued, ost, nord = bbox_utm
    xs = west + (np.arange(breite) + 0.5) * (ost - west) / breite
    ys = nord - (np.arange(hoehe) + 0.5) * (nord - sued) / hoehe
    gx, gy = np.meshgrid(xs, ys)
    return np.hypot(gx - x0, gy - y0)


# Standardfenster. GEMESSEN, nicht angenommen: saisonprobe.py zeigt fuer
# diesen Ort den Gruenhoehepunkt im JANUAR (Zyklen 2022 und 2026), nicht im
# Februar/Maerz, wie das Lehrbuch fuer das suedliche Afrika nahelegt. Das
# frueher benutzte Fenster 01.02.-15.04. sass auf dem absteigenden Ast und
# traf je nach Jahr nur 69 bis 88 Prozent der Jahresamplitude — eine
# schwankende Verzerrung, keine feste.
# Das Fenster ueberschreitet den Jahreswechsel; das Jahr benennt die Regenzeit,
# die im Dezember davor beginnt.
FENSTER_VON = (12, 15)   # 15. Dezember des Vorjahres
FENSTER_BIS = (3, 15)    # 15. Maerz


def saisonfenster(jahr: int) -> tuple[date, date]:
    """Fenster um den gemessenen Gruenhoehepunkt dieses Ortes."""
    m1, t1 = FENSTER_VON
    m2, t2 = FENSTER_BIS
    beginn = date(jahr - 1, m1, t1) if m1 > m2 else date(jahr, m1, t1)
    return beginn, date(jahr, m2, t2)


def fenster_text() -> str:
    """Das Analysefenster als Text — AUS DEN KONSTANTEN erzeugt (W1-Fix).

    Der Vorgaenger hatte hier einen hartkodierten String ("01.02.-15.04.")
    stehen, der nach der Fensterverschiebung auf 15.12.-15.03. still falsch
    weiterlief — in stdout UND in der veroeffentlichten JSON. Nie wieder:
    Es gibt genau eine Quelle, FENSTER_VON/FENSTER_BIS."""
    m1, t1 = FENSTER_VON
    m2, t2 = FENSTER_BIS
    zusatz = " des Vorjahres" if m1 > m2 else ""
    return (f"{t1:02d}.{m1:02d}.{zusatz}-{t2:02d}.{m2:02d}. "
            f"(gemessener Gruenhoehepunkt dieses Ortes, s. saisonprobe)")


def hole_jahr(lauf, aoi, jahr: int, trocken: bool, cache_dir: Path | None = None):
    von, bis = saisonfenster(jahr)
    if trocken:
        _, _, breite, hoehe = nb.aoi_gitter(aoi)
        return None, (von, bis, breite, hoehe)

    # Zwischenspeicher: Ein zweiter Lauf soll nichts kosten. Der Schluessel
    # traegt alles, was das Ergebnis bestimmt — aendert sich bbox, Aufloesung
    # oder das Ziel-CRS (seit R1 im Schluessel!), entsteht ein neuer Eintrag
    # statt eines falschen Treffers. Grad-Gitter-Altbestaende laufen dadurch
    # automatisch ins Leere und werden metrisch neu geholt.
    treffer = None
    if cache_dir is not None:
        import hashlib
        epsg = ch.aoi_epsg(aoi)
        kennung = hashlib.sha1(
            f"{aoi['bbox']}|{aoi['aufloesung_m']}|EPSG:{epsg}|{von}|{bis}".encode()
        ).hexdigest()[:16]
        treffer = cache_dir / f"{aoi['id']}-{jahr}-{kennung}.tif"
        if treffer.exists():
            ndvi, n = nb.lies_tiff(treffer, mit_n=True)
            return (ndvi, n), (von, bis, ndvi.shape[1], ndvi.shape[0])
    # NICHT nb.TYP_GEOTIFF: die Langform "image/tiff; application=geotiff"
    # quittiert die Processing-API seit einer Aenderung mit HTTP 400
    # (moegliche Werte laut Fehlermeldung: image/jpeg, image/png,
    # image/tiff, application/json). relief_ersatz.py weiss das seit dem
    # 06.08.2026; ndvi_batch.py ist an dieser Stelle noch nicht nachgezogen.
    roh = nb.process(lauf, aoi, von, bis, nb.evalscript_ndvi_tiff(),
                     TYP_TIFF, aufloesung_m=aoi["aufloesung_m"])
    if treffer is not None:
        treffer.parent.mkdir(parents=True, exist_ok=True)
        treffer.write_bytes(roh)
    ndvi, n = nb.lies_tiff_bytes(roh, mit_n=True)
    return (ndvi, n), (von, bis, ndvi.shape[1], ndvi.shape[0])


def vorzeichentest(werte: list[float]) -> dict:
    """Exakter Vorzeichentest: Wie unwahrscheinlich ist dieses Muster, wenn
    Kern und Ring in Wahrheit gleich waeren?

    Der richtige Test laeuft ueber die JAHRE, nicht ueber die Pixel. Ein Test
    auf Pixelebene wuerde jede Winzigkeit hoch signifikant machen, weil
    Nachbarpixel nicht unabhaengig sind (raeumliche Autokorrelation) — bei
    2000 Kernpixeln bekommt man p-Werte, die nichts bedeuten. Zehn Jahre sind
    dagegen zehn halbwegs unabhaengige Beobachtungen desselben Ortes.

    Der Vorzeichentest verlangt keine Normalverteilung und keine gleiche
    Streuung — er zaehlt nur, wie oft das Vorzeichen in dieselbe Richtung
    zeigt. Genau die Sparsamkeit macht ihn hier richtig.
    """
    from math import comb
    n = len([w for w in werte if w != 0])
    k = len([w for w in werte if w < 0])
    if n == 0:
        return {"n": 0, "negativ": 0, "p": None}
    extrem = max(k, n - k)
    p = 2.0 * sum(comb(n, i) for i in range(extrem, n + 1)) / (2 ** n)
    return {"n": n, "negativ": k, "p": round(min(p, 1.0), 5)}


def trend_test(jahre: list[int], werte: list[float], runden: int = 20000) -> dict:
    """Steigung der Differenz plus Permutationstest.

    Eine Steigung allein ist keine Aussage — bei zehn Punkten und dieser
    Streuung entsteht fast immer irgendeine. Deshalb wird die Jahreszuordnung
    vielfach zufaellig vertauscht: Wie oft kommt dabei eine mindestens so
    steile Gerade heraus? Das ist der p-Wert, ohne Verteilungsannahme.
    """
    rng = np.random.default_rng(20260808)   # fest, damit der Lauf reproduzierbar ist
    j = np.array(jahre, dtype=float)
    w = np.array(werte, dtype=float)
    jz = j - j.mean()
    nenner = float((jz ** 2).sum())
    if nenner == 0 or len(j) < 4:
        return {"steigung": None, "p": None}
    echt = float((jz * (w - w.mean())).sum() / nenner)
    zufall = np.array([
        float((jz * (rng.permutation(w) - w.mean())).sum() / nenner)
        for _ in range(runden)
    ])
    p = float((np.abs(zufall) >= abs(echt)).mean())
    return {"steigung": round(echt, 5), "p": round(p, 4), "runden": runden}


def median_in(maske: np.ndarray, ndvi: np.ndarray, n: np.ndarray,
              min_beobachtungen: int = 2):
    """Median der gueltigen Pixel. Pixel mit zu wenigen wolkenfreien
    Beobachtungen fliegen raus — ein Median aus einer einzigen Aufnahme ist
    ein Zufallstreffer zwischen zwei Wolken, kein Messwert."""
    gilt = maske & np.isfinite(ndvi) & (n >= min_beobachtungen)
    werte = ndvi[gilt]
    if werte.size < 5:
        return None, int(werte.size)
    return float(np.median(werte)), int(werte.size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--name", required=True, help="Kennung fuer die Ausgabedatei")
    ap.add_argument("--kern-m", type=float, default=500.0,
                    help="Radius des Kernkreises (Standard 500 m)")
    ap.add_argument("--ring-von-m", type=float, default=800.0)
    ap.add_argument("--ring-bis-m", type=float, default=1500.0)
    ap.add_argument("--von-jahr", type=int, default=2017)
    ap.add_argument("--bis-jahr", type=int, default=date.today().year)
    ap.add_argument("--aufloesung", type=float, default=20.0,
                    help="Analyseaufloesung in Metern (20 m spart PU deutlich)")
    ap.add_argument("--trocken", action="store_true")
    a = ap.parse_args()

    halbkante = a.ring_bis_m * 1.15   # etwas Luft, damit der Ring vollstaendig ist
    bbox = bbox_um(a.lat, a.lon, halbkante)
    aoi = {"id": a.name, "bbox": bbox, "aufloesung_m": a.aufloesung}
    # Metrisches Analysegitter (R1): UTM-Zone des Ortes, bbox gesnappt,
    # Zellflaeche exakt aufloesung**2.
    epsg, bbox_utm, breite, hoehe = nb.aoi_gitter(aoi)
    x0, y0 = ch.punkt_nach(epsg, a.lon, a.lat)

    print(f"Punktauswertung {a.name}")
    print(f"  Mittelpunkt   {a.lat:.6f}, {a.lon:.6f}")
    print(f"  Kern          Kreis mit {a.kern_m:.0f} m Radius "
          f"({math.pi * a.kern_m**2 / 10_000:.1f} ha)")
    print(f"  Kontrollring  {a.ring_von_m:.0f}-{a.ring_bis_m:.0f} m "
          f"({math.pi * (a.ring_bis_m**2 - a.ring_von_m**2) / 10_000:.1f} ha)")
    print(f"  Ausschnitt    {2*halbkante/1000:.1f} x {2*halbkante/1000:.1f} km, "
          f"{breite} x {hoehe} Pixel bei {a.aufloesung:.0f} m (EPSG:{epsg})")
    print(f"  Jahre         {a.von_jahr}-{a.bis_jahr}, "
          f"Fenster {fenster_text()}")
    if a.trocken:
        print(f"\n  TROCKENLAUF — es wuerden {a.bis_jahr - a.von_jahr + 1} "
              f"Processing-Anfragen gestellt.")
        return 0

    lauf = SchlankerLauf()
    abstand = None
    reihen = []
    n_lagen = []   # n-Baender aller Jahre fuer die Beobachtungsdichte (§ 8.3)

    for jahr in range(a.von_jahr, a.bis_jahr + 1):
        try:
            daten, (von, bis, b, h) = hole_jahr(lauf, aoi, jahr, False,
                                                cache_dir=CACHE)
        except RuntimeError as fehler:
            print(f"  {jahr}  FEHLER: {fehler}")
            continue
        ndvi, n = daten
        n_lagen.append(n)
        if abstand is None:
            abstand = abstandsraster(bbox_utm, ndvi.shape, x0, y0)
            kern = abstand <= a.kern_m
            ring = (abstand >= a.ring_von_m) & (abstand <= a.ring_bis_m)
            print(f"\n  Maske: {int(kern.sum())} Kernpixel, "
                  f"{int(ring.sum())} Ringpixel\n")
            print(f"  {'Jahr':<6}{'Kern':>8}{'Ring':>8}{'Differenz':>11}"
                  f"{'Pixel K/R':>12}  Bemerkung")
            print("  " + "-" * 62)

        m_kern, nk = median_in(kern, ndvi, n)
        m_ring, nr = median_in(ring, ndvi, n)
        anteil_gueltig = float((np.isfinite(ndvi) & (n >= 2)).mean())

        if m_kern is None or m_ring is None:
            print(f"  {jahr:<6}{'—':>8}{'—':>8}{'—':>11}{f'{nk}/{nr}':>12}  "
                  f"zu wenig wolkenfreie Sicht")
            reihen.append({"jahr": jahr, "kern": None, "ring": None,
                           "differenz": None, "pixel_kern": nk,
                           "pixel_ring": nr, "anteil_gueltig": anteil_gueltig})
            continue

        diff = m_kern - m_ring
        bem = "" if anteil_gueltig > 0.6 else "duenne Datenlage"
        print(f"  {jahr:<6}{m_kern:>8.3f}{m_ring:>8.3f}{diff:>+11.3f}"
              f"{f'{nk}/{nr}':>12}  {bem}")
        reihen.append({"jahr": jahr, "kern": round(m_kern, 4),
                       "ring": round(m_ring, 4), "differenz": round(diff, 4),
                       "pixel_kern": nk, "pixel_ring": nr,
                       "anteil_gueltig": round(anteil_gueltig, 3)})

    gute = [r for r in reihen if r["differenz"] is not None]
    # Beobachtungsdichte ueber alle Jahre und alle Pixel des Ausschnitts
    # (Rezeptur § 8.3); das Minimum ist ehrlicherweise oft 0 (Wolkenluecken).
    beobachtungen = None
    if n_lagen:
        alle_n = np.concatenate([np.asarray(lage).ravel() for lage in n_lagen])
        beobachtungen = {"min": int(alle_n.min()),
                         "median": round(float(np.median(alle_n)), 1),
                         "max": int(alle_n.max())}
    ergebnis = {
        "erzeugt": nb.jetzt_utc(),
        "aufruf": ch.aufruf_protokoll(),
        "punkt": {"lat": a.lat, "lon": a.lon},
        "kern_radius_m": a.kern_m,
        "ring_m": [a.ring_von_m, a.ring_bis_m],
        "bbox_crs84": bbox,
        "crs": f"EPSG:{epsg}",
        "bbox_utm": [round(v, 1) for v in bbox_utm],
        "aufloesung_m": a.aufloesung,
        "fenster": fenster_text(),
        "beobachtungen": beobachtungen,
        "quelle": "Sentinel-2 L2A ueber Copernicus Data Space Ecosystem",
        "quellenvermerk": f"Contains modified Copernicus Sentinel data {date.today().year}",
        "methode": ("Median-NDVI im Kernkreis abzueglich Median-NDVI im "
                    "Kontrollring. Der Ring traegt denselben Regen wie der "
                    "Kern; die Differenz ist deshalb vom Wetter weitgehend "
                    "unabhaengig. Sie sagt nichts ueber die Ursache."),
        "reihe": reihen,
    }
    if len(gute) >= 4:
        d = [r["differenz"] for r in gute]
        j = [r["jahr"] for r in gute]
        md = statistics.fmean(d)
        vz = vorzeichentest(d)
        tr = trend_test(j, d)
        ergebnis["differenz_mittel"] = round(md, 4)
        ergebnis["differenz_spanne"] = [round(min(d), 4), round(max(d), 4)]
        ergebnis["differenz_streuung"] = round(statistics.pstdev(d), 4)
        ergebnis["vorzeichentest"] = vz
        ergebnis["trendtest"] = tr

        print("\n  " + "-" * 62)
        print(f"  Differenz im Mittel {md:+.3f}  "
              f"(Spanne {min(d):+.3f} bis {max(d):+.3f}, "
              f"Streuung {statistics.pstdev(d):.3f})")
        richtung = "dunkler" if vz["negativ"] > vz["n"] / 2 else "heller"
        print(f"  Vorzeichentest: in {max(vz['negativ'], vz['n']-vz['negativ'])} "
              f"von {vz['n']} Jahren ist der Kern {richtung} als der Ring, "
              f"p = {vz['p']}")
        if vz["p"] is not None:
            print("    -> " + ("der Unterschied traegt" if vz["p"] < 0.05
                               else "das reicht nicht fuer eine Aussage"))
        if tr["steigung"] is not None:
            print(f"  Trend der Differenz {tr['steigung']:+.4f} NDVI pro Jahr, "
                  f"p = {tr['p']} ({tr['runden']} Permutationen)")
            print("    -> " + ("die Luecke schliesst sich nachweisbar"
                               if tr["p"] < 0.05 else
                               "kein nachweisbarer Trend — das ist Rauschen"))

    AUSGABE.mkdir(parents=True, exist_ok=True)
    ziel = AUSGABE / f"punktauswertung-{a.name}.json"
    ziel.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n  geschrieben: {ziel.relative_to(REPO)}")
    if lauf.pu_je_aoi:
        print(f"  Processing Units: {sum(lauf.pu_je_aoi.values()):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
