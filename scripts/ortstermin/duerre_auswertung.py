# -*- coding: utf-8 -*-
"""Vier Quadranten: was Gruen und Feuchte GEMEINSAM ueber einen Duerresommer sagen.

Die Frage
---------
Eine Baumschullandschaft im Duerresommer: Bleibt sie gruen, weil ihr nichts
fehlt — oder weil jemand giesst? Und wo bleibt sie gruen, obwohl ihr etwas
fehlt? Der NDVI allein kann das nicht trennen. Er misst die Farbe, und die
Farbe ist das langsamste Signal, das eine Pflanze sendet.

Der Weg
-------
1. Je Jahr ein Peak-Season-Median von NDVI und NDMI (duerre_baumschulen.py).
2. Aus zwei NORMALEN Jahren den Rauschboden messen. Das ist der Kern der
   Ehrlichkeit hier: Die Streuung zwischen zwei unauffaelligen Sommern ist
   alles, was Orbitlage, Wolkenreste, Schnittzeitpunkte und Sensordrift
   zusammen anrichten. Wer die Duerreschwelle darunter legt, findet Duerre
   im Rauschen.
3. Duerrejahre gegen Normaljahre, pixelweise, in beiden Indizes.
4. Klassifizieren:

       gruen haelt + feucht haelt    Wasser ist da
       gruen haelt + feucht faellt   Stress unter Gruen   <- der Fund
       gruen faellt + feucht faellt  klassische Duerre
       gruen faellt + feucht haelt   kein Bestand (gerodet, geerntet, offen)

5. Zusammenhaengende Flecken der interessanten Klassen als Fundstellen, mit
   Koordinate, Flaeche und beiden Differenzen — pruefbar im Luftbild.

Was das Verfahren NICHT kann
----------------------------
Es sagt nicht, WER giesst. Ein feuchter Fleck kann eine Beregnungsanlage
sein, eine Senke, ein Moorboden oder ein Graben. Es sagt auch nicht, ob
Stress schadet: Ein Gehoelz, das im August Wasser aus der Tiefe zieht, ist
gestresst und ueberlebt trotzdem. Beides gehoert in den Text, nicht in die
Karte.

Aufruf — echter Publikationslauf „Was gruen bleibt" (2026-W33):
    python scripts/ortstermin/duerre_auswertung.py \\
        --duerre 2018,2022 --normal 2021,2024,2025 --rauschpaar 2021,2025 \\
        --json docs/daten/duerre-ammerland-2026-W33.json
    (2023 flog aus den Normaljahren: selbst leicht auffaellig; das Rauschpaar
    2021/2025 sind die zwei unauffaelligsten Sommer der Reihe.)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rasterio

BASIS = Path(__file__).resolve().parents[2]
CACHE = BASIS / "scripts" / "ndvi" / "cache" / "ammerland"

sys.path.insert(0, str(BASIS / "scripts" / "ndvi"))
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS + Kartenpass, R1)

# Vier Klassen. Die Namen sind bewusst BESCHREIBEND und nicht deutend.
#
# Der erste Entwurf hiess hier "Wasser ist da", "Stress unter Gruen",
# "klassische Duerre", "kein Bestand". Das war falsch benannt, und zwar an
# der teuersten Stelle: Die grosse Klasse heisst nicht "Wasser ist da",
# sondern "kein Rueckgang, der die Schwelle reisst". Zwischen beidem liegt
# der Unterschied zwischen einer Messung und einer Behauptung — 78 % der
# Flaeche als "bewaessert" auszugeben, waere aus einer Nicht-Aussage eine
# Aussage gemacht. Die Deutung gehoert in den Text, wo man ihr widersprechen
# kann, nicht in die Legende, wo sie wie ein Messwert aussieht.
KLASSEN = {
    "unauffaellig":  "kein Rueckgang ueber der Schwelle",
    "nur_feuchte":   "Feuchte faellt, Gruen nicht",
    "beides":        "Gruen und Feuchte fallen",
    "nur_gruen":     "Gruen faellt, Feuchte nicht",
}


def lies(jahr: int):
    with rasterio.open(CACHE / f"duerre-{jahr}.tif") as q:
        # Metrisches Gitter Pflicht (R1): Grad-Altbestaende laut abweisen,
        # sonst rechnete die Flaechen- und Koordinatenlogik lagefalsch weiter.
        ch.pruefe_metrisch(q, f"duerre-{jahr}.tif")
        return q.read(1), q.read(2), q.read(3), q.bounds


def georeferenz(jahr: int):
    """(bounds, crs, kante_m) des Cache-Rasters — einmal je Lauf gelesen."""
    with rasterio.open(CACHE / f"duerre-{jahr}.tif") as q:
        ch.pruefe_metrisch(q, f"duerre-{jahr}.tif")
        return q.bounds, q.crs, float(q.transform.a)


def robuste_streuung(a: np.ndarray) -> float:
    """Sigma aus dem Median der absoluten Abweichung — unempfindlich gegen
    die Ausreisser, um die es hier gerade geht."""
    gut = a[np.isfinite(a)]
    if gut.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(gut - np.median(gut))))


def identitaetsprobe(sommer: np.ndarray, winter_jahre: dict, land: np.ndarray,
                     soll_hektar: float) -> dict:
    """Kann man Baumschulflaechen ueberhaupt vom Rest trennen?

    Vor der Skalenfrage („ist das Objekt gross genug?") steht bei diesem Thema
    eine andere: Der Satellit sieht Flaechen, keine Betriebe. Eine Aussage
    ueber BAUMSCHULEN setzt voraus, dass sich Baumschulflaeche von Gruenland
    trennen laesst. Diese Funktion prueft das, statt es anzunehmen.

    Zwei Pruefungen, beide koennen scheitern:

    1. Gibt es ueberhaupt eine Grenze? Immergruene Gehoelze muessten sich im
       Spaetwinter vom kahlen und vom abgeernteten Rest abheben — dann waere
       die Verteilung der Winterwerte zweigipflig. Ist sie eingipflig, gibt es
       keine Grenze zu ziehen; jede Schwelle waere dann gesetzt, nicht gemessen.

    2. Passt die Groessenordnung? Die amtliche Baumschulerhebung nennt fuer
       den Landkreis eine Hektarzahl. Eine Klasse, die ein Vielfaches davon
       umfasst, ist nicht „die Baumschulen" — egal wie plausibel die Karte
       aussieht. Das ist der billigste und schaerfste Test, den es hier gibt.
    """
    ergebnis = {"soll_hektar": soll_hektar}
    for jahr, w in sorted(winter_jahre.items()):
        werte = w[np.isfinite(w) & land]
        kanten = np.arange(-0.1, 1.0, 0.02)
        anzahl, _ = np.histogram(werte, bins=kanten)
        # Dreifach geglaettet, damit einzelne Zacken keine Gipfel vortaeuschen.
        glatt = anzahl.astype(float)
        for _ in range(3):
            glatt = np.convolve(glatt, np.ones(3) / 3, mode="same")
        gipfel = [i for i in range(1, len(glatt) - 1)
                  if glatt[i] > glatt[i - 1] and glatt[i] >= glatt[i + 1]
                  and glatt[i] > 0.05 * glatt.max()]
        ergebnis[str(jahr)] = {
            "median": round(float(np.median(werte)), 4),
            "p10": round(float(np.percentile(werte, 10)), 4),
            "p90": round(float(np.percentile(werte, 90)), 4),
            "gipfel": [round(float(kanten[i] + 0.01), 3) for i in gipfel],
            "gipfel_anzahl": len(gipfel),
        }
        print(f"  Winter {jahr}: Median {np.median(werte):.3f}, "
              f"P10 {np.percentile(werte, 10):.3f}, "
              f"P90 {np.percentile(werte, 90):.3f}, "
              f"{len(gipfel)} Gipfel bei "
              f"{[round(float(kanten[i] + 0.01), 2) for i in gipfel]}")

    # Groessenprobe: Welcher Anteil bliebe uebrig, wenn man die Schwelle so
    # legt, dass gerade die amtliche Hektarzahl herauskommt?
    letztes = max(winter_jahre)
    w = winter_jahre[letztes]
    werte = w[np.isfinite(w) & land]
    anteil_soll = soll_hektar * 1e4 / (land.sum() * 400)
    if 0 < anteil_soll < 1:
        schwelle = float(np.percentile(werte, 100 * (1 - anteil_soll)))
        ergebnis["noetige_winterschwelle"] = round(schwelle, 4)
        ergebnis["anteil_soll_prozent"] = round(anteil_soll * 100, 2)
        p90 = float(np.percentile(werte, 90))
        print(f"\n  Groessenprobe: {soll_hektar:,.0f} ha amtliche "
              f"Baumschulflaeche im ganzen Landkreis = hoechstens "
              f"{anteil_soll * 100:.1f} % der Auswertungsflaeche")
        print(f"  (hoechstens, weil der Ausschnitt nur den Kern des "
              f"Landkreises abdeckt — der Rest liegt ausserhalb).")
        print(f"  Eine Winterschwelle, die genau so viel Flaeche uebrig "
              f"laesst, laege bei NDVI {schwelle:.3f}.")
        print(f"  Das ist die obere Flanke des einen Gipfels "
              f"(P90 = {p90:.3f}) — kein Tal, keine Grenze.")
        print(f"  Eine Schwelle dort waere gesetzt, nicht gemessen.")
    return ergebnis


def cluster(maske: np.ndarray, min_zellen: int) -> list[np.ndarray]:
    besucht = np.zeros_like(maske, dtype=bool)
    hoehe, breite = maske.shape
    aus = []
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
        if len(zellen) >= min_zellen:
            aus.append(np.array(zellen))
    return aus


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duerre", default="2018,2022")
    p.add_argument("--normal", default="2021,2023,2024")
    p.add_argument("--rauschpaar", default="2021,2023",
                   help="zwei unauffaellige Jahre fuer den Rauschboden")
    p.add_argument("--sigma", type=float, default=2.0,
                   help="Schwelle in Vielfachen des Rauschbodens")
    p.add_argument("--min-zellen", type=int, default=12,
                   help="kleinster Fund in Zellen (20 m): 12 = 0,48 ha")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--soll-hektar", type=float, default=2252.0,
                   help="amtliche Baumschulflaeche des Landkreises "
                        "(Baumschulerhebung 2025: 2.252 ha)")
    p.add_argument("--json", type=Path)
    a = p.parse_args()

    duerre = [int(x) for x in a.duerre.split(",")]
    normal = [int(x) for x in a.normal.split(",")]
    alle = sorted({int(p.stem.split("-")[1]) for p in CACHE.glob("duerre-*.tif")})

    print(f"Ammerland — {len(alle)} Sommer im Cache: {alle}\n")
    print(f"{'Jahr':<6}{'gueltig':>9}{'NDVI':>9}{'NDMI':>9}{'n':>6}   Wasserflaeche")
    print("-" * 62)
    felder = {}
    n_felder = {}   # Band 3 je Jahr — Beobachtungsdichte fuer die JSON (§ 8.3)
    for j in alle:
        ndvi, ndmi, n, grenzen = lies(j)
        felder[j] = (ndvi, ndmi)
        n_felder[j] = n
        gut = np.isfinite(ndvi)
        # NDVI < 0 ist offenes Wasser. Das Zwischenahner Meer liegt im
        # Ausschnitt und ist die beste Probe darauf, dass die Achsen richtig
        # herum zeigen: Wasser muss im NDMI ganz oben stehen.
        wasser = gut & (ndvi < 0)
        print(f"{j:<6}{gut.mean() * 100:>8.1f}%{np.nanmedian(ndvi):>9.3f}"
              f"{np.nanmedian(ndmi):>9.3f}{np.median(n):>6.0f}"
              f"   {wasser.sum():>6} Zellen, NDMI {np.nanmedian(ndmi[wasser]):>6.3f}")

    fehlend = [j for j in duerre + normal if j not in felder]
    if fehlend:
        raise SystemExit(f"\nFehlende Jahre im Cache: {fehlend}")

    # Georeferenz einmal ziehen: metrisches Gitter (R1), Zellflaeche exakt
    # kante_m**2 — fuer das 20-m-Gitter exakt 400 m2, keine Naeherung mehr.
    grenzen, raster_crs, kante_m = georeferenz(duerre[0])

    # --- Rauschboden aus zwei unauffaelligen Sommern -----------------------
    r1, r2 = (int(x) for x in a.rauschpaar.split(","))
    rausch_ndvi = felder[r1][0] - felder[r2][0]
    rausch_ndmi = felder[r1][1] - felder[r2][1]
    s_ndvi = robuste_streuung(rausch_ndvi)
    s_ndmi = robuste_streuung(rausch_ndmi)
    schwelle_ndvi = a.sigma * s_ndvi
    schwelle_ndmi = a.sigma * s_ndmi
    print(f"\nRauschboden aus {r1} gegen {r2} (beide unauffaellig):")
    print(f"  NDVI  sigma {s_ndvi:.4f}  ->  Schwelle {schwelle_ndvi:.4f}")
    print(f"  NDMI  sigma {s_ndmi:.4f}  ->  Schwelle {schwelle_ndmi:.4f}")

    # --- Duerre gegen Normal ----------------------------------------------
    d_ndvi = (np.nanmedian(np.stack([felder[j][0] for j in duerre]), axis=0)
              - np.nanmedian(np.stack([felder[j][0] for j in normal]), axis=0))
    d_ndmi = (np.nanmedian(np.stack([felder[j][1] for j in duerre]), axis=0)
              - np.nanmedian(np.stack([felder[j][1] for j in normal]), axis=0))

    bezug = np.nanmedian(np.stack([felder[j][0] for j in normal]), axis=0)
    # Wasser und Siedlungskern raus: Auf offenem Wasser und auf Dach/Asphalt
    # ist "Duerrestress" eine sinnlose Aussage.
    land = np.isfinite(d_ndvi) & np.isfinite(d_ndmi) & (bezug > 0.25)

    print(f"\nAuswertungsflaeche: {land.sum():,} Zellen a 400 m2 "
          f"= {land.sum() * 400 / 1e6:.1f} km2 "
          f"({land.mean() * 100:.1f} % des Ausschnitts)")
    print(f"  Median  dNDVI {np.nanmedian(d_ndvi[land]):+.4f}   "
          f"dNDMI {np.nanmedian(d_ndmi[land]):+.4f}")

    # --- Identitaetsprobe: laesst sich Baumschule ueberhaupt abgrenzen? ----
    winter = {}
    for p in sorted(CACHE.glob("winter-*.tif")):
        with rasterio.open(p) as q:
            winter[int(p.stem.split("-")[1])] = q.read(1)
    probe = {}
    if winter:
        print("\nIdentitaetsprobe — Spaetwinter (Feb bis 20. Maerz), "
              "immergruen gegen kahl:")
        probe = identitaetsprobe(bezug, winter, land, a.soll_hektar)

    gruen_faellt = land & (d_ndvi <= -schwelle_ndvi)
    feucht_faellt = land & (d_ndmi <= -schwelle_ndmi)
    masken = {
        "unauffaellig": land & ~gruen_faellt & ~feucht_faellt,
        "nur_feuchte":  land & ~gruen_faellt & feucht_faellt,
        "beides":       land & gruen_faellt & feucht_faellt,
        "nur_gruen":    land & gruen_faellt & ~feucht_faellt,
    }
    print("\nVier Quadranten:")
    for schluessel, name in KLASSEN.items():
        m = masken[schluessel]
        print(f"  {m.sum() / land.sum() * 100:>6.2f} %  {m.sum():>7} Zellen  "
              f"{m.sum() * 400 / 1e4:>8.0f} ha   {name}")

    # --- Fundstellen der beiden erklaerungsbeduerftigen Klassen ------------
    hoehe, breite = d_ndvi.shape

    def koord(zellen):
        # Zellschwerpunkt im metrischen Raster-CRS, dann zurueck nach lon/lat
        # (WGS 84) fuer Kontaktbogen, Luftbild und Ortstermin.
        r = zellen[:, 0].mean(); c = zellen[:, 1].mean()
        x = grenzen.left + (c + 0.5) / breite * (grenzen.right - grenzen.left)
        y = grenzen.top - (r + 0.5) / hoehe * (grenzen.top - grenzen.bottom)
        lon, lat = ch.nach_lonlat(raster_crs, x, y)
        return round(lat, 4), round(lon, 4)

    fundstellen = {}
    for schluessel in ("nur_feuchte", "beides"):
        gefunden = []
        for zellen in cluster(masken[schluessel], a.min_zellen):
            r, c = zellen[:, 0], zellen[:, 1]
            lat, lon = koord(zellen)
            gefunden.append({
                "klasse": schluessel,
                "lat": lat, "lon": lon,
                "zellen": int(len(zellen)),
                "hektar": round(len(zellen) * 400 / 1e4, 2),
                "d_ndvi": round(float(np.nanmean(d_ndvi[r, c])), 4),
                "d_ndmi": round(float(np.nanmean(d_ndmi[r, c])), 4),
                "ndvi_normal": round(float(np.nanmean(bezug[r, c])), 4),
            })
        gefunden.sort(key=lambda f: f["d_ndmi"] if schluessel == "nur_feuchte"
                      else f["d_ndvi"])
        fundstellen[schluessel] = gefunden
        print(f"\nFundstellen '{schluessel}': {len(gefunden)} "
              f"(mindestens {a.min_zellen} Zellen = "
              f"{a.min_zellen * 400 / 1e4:.2f} ha)")
        print(f"  {'#':>3} {'lat':>9} {'lon':>9} {'ha':>7} {'dNDVI':>8} "
              f"{'dNDMI':>8} {'NDVI norm':>10}")
        for i, f in enumerate(gefunden[:a.top], 1):
            print(f"  {i:>3} {f['lat']:>9.4f} {f['lon']:>9.4f} {f['hektar']:>7.2f} "
                  f"{f['d_ndvi']:>+8.3f} {f['d_ndmi']:>+8.3f} "
                  f"{f['ndvi_normal']:>10.3f}")

    if a.json:
        # Beobachtungsdichte (§ 8.3) ueber die verwendeten Jahre und alle
        # Pixel; das Minimum ist ehrlicherweise oft 0 (Wolkenluecken).
        alle_n = np.concatenate([np.asarray(n_felder[j]).ravel()
                                 for j in duerre + normal])
        # bbox zusaetzlich in lon/lat, damit Leser ohne Projektionswerkzeug
        # den Ausschnitt einordnen koennen; gemessen wird in bbox_utm.
        bbox_lonlat = list(ch.bbox_nach_lonlat(raster_crs, grenzen))
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "aoi": "ammerland",
            "aufruf": ch.aufruf_protokoll(),
            "crs": f"EPSG:{raster_crs.to_epsg()}",
            "bbox_utm": [grenzen.left, grenzen.bottom, grenzen.right, grenzen.top],
            "bbox": [round(v, 6) for v in bbox_lonlat],
            "gitter_m": round(kante_m, 1),
            "beobachtungen": {"min": int(alle_n.min()),
                              "median": round(float(np.median(alle_n)), 1),
                              "max": int(alle_n.max())},
            "duerrejahre": duerre,
            "normaljahre": normal,
            "rauschpaar": [r1, r2],
            "sigma_ndvi": round(s_ndvi, 5),
            "sigma_ndmi": round(s_ndmi, 5),
            "schwelle_sigma": a.sigma,
            "schwelle_ndvi": round(schwelle_ndvi, 5),
            "schwelle_ndmi": round(schwelle_ndmi, 5),
            "flaeche_zellen": int(land.sum()),
            "identitaetsprobe": probe,
            "jahresmedian": {str(j): {
                "ndvi": round(float(np.nanmedian(felder[j][0])), 4),
                "ndmi": round(float(np.nanmedian(felder[j][1])), 4)} for j in alle},
            "quadranten": {k: {
                "name": KLASSEN[k],
                "zellen": int(masken[k].sum()),
                "anteil_prozent": round(float(masken[k].sum() / land.sum() * 100), 2),
                "hektar": round(float(masken[k].sum() * 400 / 1e4), 1)}
                for k in KLASSEN},
            "fundstellen": fundstellen,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nGeschrieben: {a.json}")


if __name__ == "__main__":
    main()
