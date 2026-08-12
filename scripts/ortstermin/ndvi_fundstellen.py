"""Fundstellen suchen: wo im Stadtgebiet ist Grün verschwunden — und wie viel?

Rechnet ausschließlich auf den Rastern, die der NDVI-Monitor ohnehin erzeugt
(ftp-mirror-geophora/tiles/ndvi/<aoi>/) — kein neuer Satellitenabruf, kein PU.
Seit R1 (12.08.2026) auf dem metrischen Analysegitter (UTM, Zellfläche exakt
aus dem Raster-Transform); Grad-Altbestände weist das Skript laut ab.

Zwei Maße, bewusst getrennt:

  Anomalie   = aktuell − Fensterbaseline (tagesgenau gematchte Klimatologie
               2020–2025 im selben Kalenderfenster).  Beantwortet: „ungewöhnlich
               für diese Jahreszeit?" — trennt Ernte und Mahd sauber ab, weil die
               jedes Jahr zur selben Zeit passieren.
  Mehrjahres = aktuell − Peak-Season-Composite eines Referenzjahres.
               Beantwortet: „ist hier über die Jahre etwas verschwunden?"

Erst beides zusammen mit der Persistenzklasse (4 = Vegetationsverlust über
mindestens 4 von 6 Monaten) ergibt einen Kandidaten, den anzusehen sich lohnt.

WICHTIG, und das gehört in jede Veröffentlichung: Das Analysegitter ist 20 m.
Ein Einzelbaum ist darin unsichtbar. Gefunden werden Flächen, nicht Bäume, und
der NDVI nennt keine Ursache — Fällung, Baustelle, Trockenstress und Umbruch
sehen gleich aus. Die Fundstellen sind Hinweise für den Ortstermin, keine
Belege.

Aufruf (venv des NDVI-Batches):
    python ndvi_fundstellen.py --aoi oldenburg --referenzjahr 2021
Optionen: --bbox W,S,O,N (Ausschnitt „Stadtgebiet"), --min-flaeche 4 (Pixel),
          --top 20, --json <pfad>
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
TILES = BASIS / "ftp-mirror-geophora" / "tiles" / "ndvi"

sys.path.insert(0, str(BASIS / "scripts" / "ndvi"))
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS + Kartenpass, R1)

# Schwellen — bewusst dieselben wie in aois.json (Klassifikation), damit die
# Fundstellen zur Karte des Monitors passen.
S_ANOMALIE = -0.10        # NDVI-Punkte unter der Klimatologie
S_MEHRJAHRES = -0.15      # NDVI-Punkte unter dem Referenzjahr
S_WAR_GRUEN = 0.55        # so grün war die Fläche im Referenzjahr mindestens


def lade(pfad: Path, band: int = 1) -> np.ndarray:
    with rasterio.open(pfad) as ds:
        # Metrisches Gitter Pflicht (R1) — in JEDER Quelle: schon ein einzelnes
        # Grad-Altraster (etwa ein altes composite-<jahr>) läge sonst still
        # neben dem UTM-Gitter und würde pixelverschoben verrechnet.
        ch.pruefe_metrisch(ds, pfad.name)
        return ds.read(band).astype("float64")


def gitter(pfad: Path):
    """Georeferenz des Analysegitters: bounds, Maße, CRS, EPSG, Kantenlänge (m)
    und Band 2 (Zahl wolkenfreier Beobachtungen, falls vorhanden)."""
    with rasterio.open(pfad) as ds:
        epsg = ch.pruefe_metrisch(ds, pfad.name)
        n_beob = ds.read(2) if ds.count >= 2 else None
        return (ds.bounds, ds.width, ds.height, ds.crs, epsg,
                float(ds.transform.a), n_beob)


def cluster(maske: np.ndarray, min_flaeche: int) -> list[np.ndarray]:
    """Zusammenhängende Flächen (8er-Nachbarschaft) als Index-Listen."""
    besucht = np.zeros_like(maske, dtype=bool)
    hoehe, breite = maske.shape
    gefunden: list[np.ndarray] = []
    for start in np.argwhere(maske):
        r0, c0 = int(start[0]), int(start[1])
        if besucht[r0, c0]:
            continue
        stapel = deque([(r0, c0)])
        besucht[r0, c0] = True
        zellen = []
        while stapel:
            r, c = stapel.popleft()
            zellen.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < hoehe and 0 <= cc < breite \
                            and maske[rr, cc] and not besucht[rr, cc]:
                        besucht[rr, cc] = True
                        stapel.append((rr, cc))
        if len(zellen) >= min_flaeche:
            gefunden.append(np.array(zellen))
    return gefunden


def anteil(maske: np.ndarray, gebiet: np.ndarray) -> float:
    n = int(gebiet.sum())
    return float((maske & gebiet).sum()) / n if n else 0.0


def kastenmittel(a: np.ndarray, radius: int) -> np.ndarray:
    """Gleitendes Mittel über ein Quadrat (Summenbild — ohne scipy)."""
    gueltig = np.isfinite(a)
    werte = np.where(gueltig, a, 0.0)
    s_w = np.pad(werte, 1, mode="edge").cumsum(0).cumsum(1)
    s_n = np.pad(gueltig.astype("float64"), 1, mode="edge").cumsum(0).cumsum(1)

    def fenster(s):
        h, b = a.shape
        r0 = np.clip(np.arange(h) - radius, 0, h)
        r1 = np.clip(np.arange(h) + radius + 1, 0, h)
        c0 = np.clip(np.arange(b) - radius, 0, b)
        c1 = np.clip(np.arange(b) + radius + 1, 0, b)
        return (s[np.ix_(r1, c1)] - s[np.ix_(r0, c1)]
                - s[np.ix_(r1, c0)] + s[np.ix_(r0, c0)])

    with np.errstate(invalid="ignore", divide="ignore"):
        return fenster(s_w) / np.maximum(fenster(s_n), 1e-9)


def stadtmaske(basis_saison: np.ndarray, radius_px: int, schwelle: float) -> np.ndarray:
    """„Die Umgebung ist bebaut“ — grober, aber belastbarer Ersatz für einen
    Versiegelungslayer: im Nachbarschaftsmittel des Saison-Normals liegen
    Dächer und Straßen dauerhaft niedrig, Acker und Grünland im Hochsommer
    dagegen hoch. Bewusst auf der Baseline gerechnet (2020–2025), damit das
    aktuelle Jahr die Maske nicht verschiebt."""
    return kastenmittel(basis_saison, radius_px) < schwelle


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", default="oldenburg")
    p.add_argument("--referenzjahr", type=int, default=2021)
    p.add_argument("--bbox", help='W,S,O,N — Ausschnitt „Stadtgebiet“ (Default: ganze AOI)')
    p.add_argument("--min-flaeche", type=int, default=4, help="Mindestzahl Pixel je Fundstelle")
    p.add_argument("--max-flaeche", type=int, default=0,
                   help="Höchstzahl Pixel je Fundstelle (0 = unbegrenzt); trennt "
                        "Baumgruppen von Schlägen")
    p.add_argument("--stadt-radius-m", type=float, default=250.0,
                   help="Fensterradius der Stadtmaske (0 = keine Maske)")
    p.add_argument("--stadt-schwelle", type=float, default=0.55,
                   help="Nachbarschaftsmittel des Saison-Normals darunter = bebaut")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--json", type=Path)
    a = p.parse_args()

    ordner = TILES / a.aoi
    jetzt = lade(ordner / "aktuell-analyse.tif")
    basis_fenster = lade(ordner / "baseline-fenster.tif")
    basis_saison = lade(ordner / "baseline.tif")
    referenz = lade(ordner / f"composite-{a.referenzjahr}.tif")
    persistenz = lade(ordner / "persistenz.tif")
    meta = json.loads((ordner / "meta.json").read_text(encoding="utf-8"))

    (bounds, breite, hoehe, crs, epsg,
     kante_m, n_beob) = gitter(ordner / "aktuell-analyse.tif")
    d_x = (bounds.right - bounds.left) / breite
    d_y = (bounds.top - bounds.bottom) / hoehe
    # Zellfläche exakt aus dem Transform (R1) — keine Grad-Näherung mehr.
    px_m2 = kante_m ** 2

    # Pixelmittelpunkte im Raster-CRS (Meter); der lon/lat-Ausschnitt wird
    # dorthin projiziert statt umgekehrt jede Zelle nach lon/lat zu rechnen
    # (Muster ndvi_trend.py).
    cc, rr = np.meshgrid(np.arange(breite), np.arange(hoehe))
    x_mitte = bounds.left + (cc + 0.5) * d_x
    y_mitte = bounds.top - (rr + 0.5) * d_y

    if a.bbox:
        w, s, o, n = ch.bbox_nach(crs, [float(v) for v in a.bbox.split(",")])
        gebiet = (x_mitte >= w) & (x_mitte <= o) & (y_mitte >= s) & (y_mitte <= n)
    else:
        gebiet = np.ones((hoehe, breite), bool)

    gueltig = np.isfinite(jetzt) & np.isfinite(basis_fenster) & np.isfinite(referenz)
    with np.errstate(invalid="ignore"):
        anomalie = jetzt - basis_fenster
        mehrjahres = jetzt - referenz

    # ---------- Kontextzahlen: wie viel verändert sich überhaupt? ----------
    def bilanz(g: np.ndarray, titel: str) -> dict:
        gg = g & gueltig
        verlust = anomalie <= S_ANOMALIE
        zugewinn = anomalie >= -S_ANOMALIE
        return {
            "gebiet": titel,
            "flaeche_km2": round(float(gg.sum()) * px_m2 / 1e6, 2),
            "anomalie_median": round(float(np.median(anomalie[gg])), 4),
            "anteil_verlust": round(anteil(verlust, gg), 4),
            "anteil_zugewinn": round(anteil(zugewinn, gg), 4),
            "anteil_persistenter_verlust": round(anteil(persistenz == 4, gg), 4),
            "mehrjahres_median": round(float(np.median(mehrjahres[gg])), 4),
        }

    # ---------- Stadtmaske ----------
    if a.stadt_radius_m > 0:
        radius_px = max(1, int(round(a.stadt_radius_m / kante_m)))
        bebaut = stadtmaske(basis_saison, radius_px, a.stadt_schwelle)
    else:
        radius_px = 0
        bebaut = np.ones((hoehe, breite), bool)

    kontext = [bilanz(np.ones_like(gebiet), f"gesamte AOI „{a.aoi}“")]
    if a.bbox:
        kontext.append(bilanz(gebiet, "Ausschnitt (Stadtgebiet)"))
    if radius_px:
        kontext.append(bilanz(gebiet & bebaut, "bebaute Nachbarschaft"))
        kontext.append(bilanz(gebiet & ~bebaut, "offene Landschaft"))

    # ---------- Fundstellen ----------
    kandidat = (gebiet & bebaut & gueltig
                & (referenz >= S_WAR_GRUEN)
                & (basis_saison >= S_WAR_GRUEN)
                & (anomalie <= S_ANOMALIE)
                & (mehrjahres <= S_MEHRJAHRES))

    fundstellen = []
    for zellen in cluster(kandidat, a.min_flaeche):
        if a.max_flaeche and len(zellen) > a.max_flaeche:
            continue
        r, c = zellen[:, 0], zellen[:, 1]
        # Zentrum und Kasten in Metern gerechnet, für die Ausgabe zurück nach
        # lon/lat projiziert (UTM → WGS 84, R1) — die JSON bleibt lesbar.
        lon_f, lat_f = ch.nach_lonlat(crs, float(x_mitte[r, c].mean()),
                                      float(y_mitte[r, c].mean()))
        kasten = ch.bbox_nach_lonlat(crs, (
            bounds.left + c.min() * d_x, bounds.top - (r.max() + 1) * d_y,
            bounds.left + (c.max() + 1) * d_x, bounds.top - r.min() * d_y))
        fundstellen.append({
            "lat": round(lat_f, 5),
            "lon": round(lon_f, 5),
            "pixel": int(len(zellen)),
            "flaeche_m2": int(round(len(zellen) * px_m2)),
            "ndvi_referenz": round(float(np.mean(referenz[r, c])), 3),
            "ndvi_jetzt": round(float(np.mean(jetzt[r, c])), 3),
            "anomalie": round(float(np.mean(anomalie[r, c])), 3),
            "mehrjahres": round(float(np.mean(mehrjahres[r, c])), 3),
            "persistent": round(float(np.mean(persistenz[r, c] == 4)), 2),
            "bbox": [round(float(k), 5) for k in kasten],
        })
    # Rang: Fläche mal Stärke des Rückgangs — große, deutliche Stellen zuerst.
    fundstellen.sort(key=lambda f: f["pixel"] * -f["mehrjahres"], reverse=True)

    # Beobachtungsdichte (§ 8.3, analog ndvi_trend.py): Band 2 des aktuellen
    # Analyse-Composites — das Minimum ist ehrlicherweise oft 0 (Wolkenrand).
    beobachtungen = None
    if n_beob is not None:
        beobachtungen = {"min": int(n_beob.min()),
                         "median": round(float(np.median(n_beob)), 1),
                         "max": int(n_beob.max())}

    ergebnis = {
        "aoi": a.aoi,
        "stand": meta.get("aktualisiert"),
        "zeitfenster": meta.get("zeitfenster"),
        "referenzjahr": a.referenzjahr,
        "gitter_m": round(kante_m, 1),
        "crs": f"EPSG:{epsg}",
        "beobachtungen": beobachtungen,
        "schwellen": {"anomalie": S_ANOMALIE, "mehrjahres": S_MEHRJAHRES,
                      "war_gruen": S_WAR_GRUEN, "min_flaeche_px": a.min_flaeche,
                      "max_flaeche_px": a.max_flaeche or None,
                      "stadtmaske": (None if not radius_px else
                                     {"radius_px": radius_px,
                                      "radius_m": a.stadt_radius_m,
                                      "schwelle": a.stadt_schwelle})},
        "kontext": kontext,
        "fundstellen_gesamt": len(fundstellen),
        "fundstellen": fundstellen[:a.top],
        # Kartenpass (Rezeptur § 8.1): der exakte Aufruf dieses Laufs.
        "aufruf": ch.aufruf_protokoll(),
    }

    print(json.dumps(ergebnis, ensure_ascii=False, indent=2))
    if a.json:
        a.json.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\ngeschrieben: {a.json}")


if __name__ == "__main__":
    main()
