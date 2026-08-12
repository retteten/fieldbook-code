"""Mehrjahrestrend statt Zweiwochen-Anomalie — die andere Frage ans selbe Raster.

Die Anomaliekarte fragt: *Ist dieser Bildpunkt gerade ungewöhnlich für die
Jahreszeit?* Das ist eine Wetterfrage. Wer wissen will, ob eine Stadt über Jahre
Grün verliert, muss anders fragen: *Verliert dieser Bildpunkt Jahr für Jahr
Grünmasse?*

(Bis zum 09.08.2026 stand hier „wird dunkler". Das war in beide Leserichtungen
falsch: Im Echtbild ist dichte Vegetation das Dunkelste überhaupt, und auf der
NDVI-Farbskala dieses Projekts ist 0,9 Tiefgrün und 0,1 heller Sand. „Dunkler"
hieße also mehr Grün, nicht weniger. Der NDVI misst Blattmasse über das
Nahinfrarot — nicht Helligkeit.)

Verfahren (Hausstandard R3′, 12.08.2026): je Pixel die **Theil-Sen-Steigung**
durch die Peak-Season-Composites der Jahre — der Median aller paarweisen
Steigungen, robust gegen ein einzelnes Ausreißerjahr (Dürresommer!). Dazu je
Pixel der **Mann-Kendall-Test** (verteilungsfrei, Standardtest für
Umweltzeitreihen). Quellen: Sen 1968, Mann 1945, Kendall 1975; praxisnah
Helsel u. a. 2020 (USGS TM 4-A3, Kap. 12).

Startjahr-Regel (seit 12.08.2026 abends): Die Reihe beginnt 2018 — volle
Zwillingskonstellation (5-Tage-Wiederkehr), Collection-1 homogen reprozessiert.
Ehrlich gesagt werden muss: Auch bei n = 8 bleibt der Test streng — zweiseitig
p < 0,05 verlangt |S| ≥ 18 von maximal 28 (bei n = 6 wären es 13 von 15, fast
perfekte Monotonie; genau deshalb wurde der unbegründete Start 2020 der
Erstrechnung aufgegeben). Schwache echte Trends bleiben unerkannt; die
Belegkraft trägt deshalb weiterhin die 4-Zellen-Clusterregel mit, und genau
das gehört in den Methodenabschnitt der Folge. Die Kleinste-Quadrate-Gerade
der Läufe vor 08/2026 ist die benannte historische Fassung. Wichtig: Die
Trendreihe ist von der Anomalie-Klimatologie (baseline_jahre in aois.json,
2020–2025) bewusst entkoppelt — die Klimatologie soll „junges Normal" sein,
der Trend möglichst lang.

Gerechnet wird auf dem metrischen Analysegitter (R1: UTM, Zellfläche exakt
400 m²) aus dem Cache des NDVI-Batches — kein neuer Satellitenabruf.
Ausgegeben werden Verteilung, Fundstellen und — das ist der eigentliche
Zweck — die Überschneidung mit den Fundstellen der Anomaliekarte (steht seit
V3 auch in der JSON, nicht mehr nur auf stdout).

Aufruf (venv des NDVI-Batches) — echter Publikationslauf „Vier Pixel"
(verlängerte Reihe, 12.08.2026 abends):
    python scripts/ortstermin/ndvi_trend.py --aoi oldenburg \\
        --bbox 8.155,53.105,8.285,53.185 \\
        --jahre 2018,2019,2020,2021,2022,2023,2024,2025 \\
        --vergleich docs/daten/fundstellen-oldenburg-2026-W33.json \\
        --json docs/daten/trend-oldenburg-2018-2025.json
Echter Publikationslauf Harz (09.08.2026):
    python scripts/ortstermin/ndvi_trend.py --aoi harz \\
        --maske-arm "dunkle Nachbarschaft (Kahlflaeche/Ort)" \\
        --maske-reich "geschlossene Vegetation" \\
        --json docs/daten/harz-trend-2026-08-09.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rasterio

BASIS = Path(__file__).resolve().parents[2]
CACHE = BASIS / "scripts" / "ndvi" / "cache"
TILES = BASIS / "ftp-mirror-geophora" / "tiles" / "ndvi"

sys.path.insert(0, str(BASIS / "scripts" / "ndvi"))
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS + Kartenpass, R1)


def theil_sen(stapel: np.ndarray, jahre: list[int]) -> np.ndarray:
    """Theil-Sen-Steigung je Bildpunkt: Median aller Paarsteigungen (R3′).

    stapel: (jahre, hoehe, breite). Bei n = 6 Jahren sind das 15 Paare —
    vektorisiert über die Paarliste (15 Array-Operationen), kein Loop über
    Pixel. Paare mit NaN ignoriert nanmedian; Bildpunkte, die nicht in allen
    Jahren gültig sind, maskiert der Aufrufer ohnehin (Konsistenz mit dem
    Mann-Kendall-S, das dort NaN würde).
    """
    n = len(jahre)
    paare = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    steigungen = np.stack([(stapel[j] - stapel[i]) / float(jahre[j] - jahre[i])
                           for i, j in paare])
    with np.errstate(invalid="ignore"):
        return np.nanmedian(steigungen, axis=0)


def mann_kendall(stapel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mann-Kendall-S und zweiseitiger p-Wert je Bildpunkt.

    Normalapproximation mit Kontinuitätskorrektur (S∓1). Zum Vergleich die
    EXAKTEN kritischen Werte bei n = 6 (S läuft von −15 bis +15, Verteilung
    über die 720 Permutationen): P(|S| ≥ 15) = 0,0028 · P(|S| ≥ 13) = 0,0167 ·
    P(|S| ≥ 11) = 0,0556 · P(|S| ≥ 9) = 0,1361. Zweiseitig p < 0,05 verlangt
    also |S| ≥ 13 — fast perfekte Monotonie; die Approximation ist bei n = 6
    nur eine Näherung und die Testpower gering. Deshalb bleibt die
    4-Zellen-Clusterregel der eigentliche Beleg-Träger (§ 6.3 der Rezeptur).

    Bindungskorrektur entfällt bewusst: NDVI ist stetig, echte Bindungen sind
    bei float32-Medianen praktisch ausgeschlossen.
    """
    n = stapel.shape[0]
    S = np.zeros(stapel.shape[1:], dtype="float64")
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += np.sign(stapel[j] - stapel[i])
    wurzel = math.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
    with np.errstate(invalid="ignore"):
        z = np.where(S > 0, (S - 1) / wurzel,
                     np.where(S < 0, (S + 1) / wurzel, 0.0))
        p = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(
            np.abs(z) / math.sqrt(2))))
    return S, np.clip(p, 0.0, 1.0)


def kastenmittel(a: np.ndarray, radius: int) -> np.ndarray:
    """Gleitendes Mittel (Summenbild) — wie in ndvi_fundstellen.py."""
    gueltig = np.isfinite(a)
    s_w = np.pad(np.where(gueltig, a, 0.0), 1, mode="edge").cumsum(0).cumsum(1)
    s_n = np.pad(gueltig.astype("float64"), 1, mode="edge").cumsum(0).cumsum(1)

    def fenster(s):
        h, b = a.shape
        r0 = np.clip(np.arange(h) - radius, 0, h); r1 = np.clip(np.arange(h) + radius + 1, 0, h)
        c0 = np.clip(np.arange(b) - radius, 0, b); c1 = np.clip(np.arange(b) + radius + 1, 0, b)
        return (s[np.ix_(r1, c1)] - s[np.ix_(r0, c1)] - s[np.ix_(r1, c0)] + s[np.ix_(r0, c0)])

    with np.errstate(invalid="ignore", divide="ignore"):
        return fenster(s_w) / np.maximum(fenster(s_n), 1e-9)


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
                    if 0 <= rr < hoehe and 0 <= cc < breite and maske[rr, cc] and not besucht[rr, cc]:
                        besucht[rr, cc] = True; stapel.append((rr, cc))
        if len(zellen) >= min_zellen:
            aus.append(np.array(zellen))
    return aus


def saison_fenster_metadaten(jahre: list[int]) -> dict | None:
    """Die ECHTEN Rechenfenster der Jahres-Composites, aus aois.json abgeleitet.

    Hartkodierte Fensterangaben in Metadaten waren der Botswana-Fehler (W1) —
    hier kommt das Fenster aus derselben Quelle, aus der auch der Batch die
    Composites rechnet (saison.start_woche/end_woche, ISO-Wochen)."""
    pfad = TILES / "aois.json"
    if not pfad.exists():
        return None
    saison = json.loads(pfad.read_text(encoding="utf-8"))["saison"]
    je_jahr = {}
    for j in jahre:
        von = dt.date.fromisocalendar(j, saison["start_woche"], 1)
        bis = dt.date.fromisocalendar(j, saison["end_woche"], 7)
        je_jahr[str(j)] = [von.isoformat(), bis.isoformat()]
    return {"iso_wochen": [saison["start_woche"], saison["end_woche"]],
            "je_jahr": je_jahr}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", default="oldenburg")
    p.add_argument("--bbox", help="W,S,O,N (lon/lat — wird ins Raster-CRS projiziert)")
    p.add_argument("--jahre", default="2018,2019,2020,2021,2022,2023,2024,2025",
                   help="Startjahr-Regel 2018 (volle Zwillingskonstellation); "
                        "bewusst länger als die Anomalie-Klimatologie")
    p.add_argument("--schwelle", type=float, default=-0.02,
                   help="NDVI-Verlust je Jahr, ab dem eine Zelle als fallend gilt")
    p.add_argument("--min-zellen", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Signifikanzniveau des Mann-Kendall-Tests (zweiseitig)")
    p.add_argument("--stadt-radius-m", type=float, default=250.0)
    p.add_argument("--stadt-schwelle", type=float, default=0.55)
    # NICHT "dunkel"/"hell": Der NDVI misst Blattmasse ueber das Nahinfrarot,
    # keine Helligkeit — und beides laeuft auseinander. Ein dichter Fichten-
    # bestand ist im Echtbild fast schwarz und hat NDVI 0,85; ein Sandweg ist
    # weiss und hat 0,15. Auf unserer eigenen Farbskala ist 0,9 sogar Tiefgruen.
    # "Wird dunkler" hiesse dort also MEHR Vegetation — das Gegenteil dessen,
    # wofuer der Begriff hier jahrelang stand.
    p.add_argument("--maske-arm", default="bebaute Nachbarschaft",
                   help="Name der Klasse UNTER der Schwelle, also blattarm "
                        "(Oldenburg: Bebauung, Oberharz: Kahlflaeche und Ort)")
    p.add_argument("--maske-reich", default="offene Landschaft",
                   help="Name der Klasse UEBER der Schwelle, also blattreich")
    p.add_argument("--vergleich", type=Path, help="Fundstellen-JSON der Anomaliekarte")
    p.add_argument("--json", type=Path, help="Ergebnis als JSON schreiben")
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args()

    jahre = [int(j) for j in a.jahre.split(",")]
    stapel, n_lagen = [], []
    for j in jahre:
        pfad = CACHE / a.aoi / f"jahr-{j}.tif"
        if not pfad.exists():
            raise SystemExit(f"fehlt: {pfad}")
        with rasterio.open(pfad) as ds:
            # Metrisches Gitter Pflicht (R1): Grad-Altbestaende laut abweisen.
            epsg = ch.pruefe_metrisch(ds, pfad.name)
            stapel.append(ds.read(1).astype("float64"))
            # Band 2 = wolkenfreie Beobachtungen (Beobachtungsdichte, § 8.3).
            n_lagen.append(ds.read(2) if ds.count >= 2 else None)
            bounds, breite, hoehe = ds.bounds, ds.width, ds.height
            crs = ds.crs
            kante_m = float(ds.transform.a)   # exakt, keine Naeherung mehr
    stapel = np.stack(stapel)

    with rasterio.open(TILES / a.aoi / "baseline.tif") as ds:
        ch.pruefe_metrisch(ds, "baseline.tif")
        saison = ds.read(1).astype("float64")

    d_x = (bounds.right - bounds.left) / breite
    d_y = (bounds.top - bounds.bottom) / hoehe

    # Pixelmittelpunkte im Raster-CRS (Meter); der lon/lat-Ausschnitt wird
    # dorthin projiziert statt umgekehrt jede Zelle nach lon/lat zu rechnen.
    cc, rr = np.meshgrid(np.arange(breite), np.arange(hoehe))
    x_mitte = bounds.left + (cc + 0.5) * d_x
    y_mitte = bounds.top - (rr + 0.5) * d_y
    if a.bbox:
        w, s, o, n = ch.bbox_nach(crs, [float(v) for v in a.bbox.split(",")])
        gebiet = (x_mitte >= w) & (x_mitte <= o) & (y_mitte >= s) & (y_mitte <= n)
    else:
        gebiet = np.ones((hoehe, breite), bool)

    radius_px = max(1, int(round(a.stadt_radius_m / kante_m)))
    bebaut = kastenmittel(saison, radius_px) < a.stadt_schwelle

    # --- Steigung + Signifikanz je Pixel: Theil-Sen + Mann-Kendall (R3′) ---
    gueltig = np.isfinite(stapel).all(axis=0)
    steigung = np.where(gueltig, theil_sen(stapel, jahre), np.nan)
    S, p_wert = mann_kendall(stapel)   # NaN, wo ein Jahr fehlt — passt zu gueltig

    gesammelt = []

    def bilanz(maske, titel):
        m = maske & gueltig & np.isfinite(steigung)
        n = int(m.sum())
        if not n:
            print(f"  {titel}: keine Zellen"); return
        s = steigung[m]
        sig = np.isfinite(p_wert[m]) & (p_wert[m] < a.alpha)
        gesammelt.append({
            "gebiet": titel,
            "flaeche_km2": round(n * kante_m ** 2 / 1e6, 2),
            "median_je_jahr": round(float(np.median(s)), 4),
            "anteil_fallend": round(float((s <= a.schwelle).mean()), 4),
            "anteil_steigend": round(float((s >= -a.schwelle).mean()), 4),
            # Aggregat berichten, auch wenn es klein ausfaellt (§ 6.4):
            # Anteil der Zellen, deren Mann-Kendall-p unter alpha liegt.
            "anteil_mk_signifikant": round(float(sig.mean()), 4),
        })
        print(f"  {titel:28s} {n * kante_m ** 2 / 1e6:6.2f} km² | "
              f"Median {np.median(s):+.4f}/Jahr | "
              f"fallend <={a.schwelle:+.2f}: {(s <= a.schwelle).mean():5.2%} | "
              f"steigend >={-a.schwelle:+.2f}: {(s >= -a.schwelle).mean():5.2%} | "
              f"MK p<{a.alpha:g}: {sig.mean():5.2%}")

    print(f"Trend {jahre[0]}–{jahre[-1]}, Gitter {kante_m:.0f} m (EPSG:{epsg}), "
          f"Theil-Sen + Mann-Kendall, Peak-Season-Composites\n")
    # Die Maske ist generisch (Nachbarschaftsmittel des Saison-Normals unter
    # der Schwelle) — nur ihr NAME war es nicht. In Oldenburg trifft sie
    # Bebauung, im Oberharz Kahlflaechen und Ortschaften zugleich. Ein Label,
    # das anderswo falsch ist, wandert sonst als scheinbarer Messwert in einen
    # Text. Deshalb benennbar, mit der Stadt als Vorgabe.
    bilanz(gebiet, "Ausschnitt gesamt")
    bilanz(gebiet & bebaut, a.maske_arm)
    bilanz(gebiet & ~bebaut, a.maske_reich)

    # --- Fundstellen des Trends ---
    # Belegschwelle unveraendert: Steigung unter der Schwelle UND mindestens
    # vier zusammenhaengende Zellen. Der MK-p wird je Fundstelle BERICHTET,
    # ist aber kein Filter — bei n = 6 waere er strenger als die Datenlage
    # rechtfertigt (s. Docstring von mann_kendall).
    kandidat = (gebiet & bebaut & gueltig & (steigung <= a.schwelle)
                & (saison >= a.stadt_schwelle))
    stellen = sorted(cluster(kandidat, a.min_zellen), key=len, reverse=True)
    print(f"\nTrend-Fundstellen (bebaut, vorher gruen, >={a.min_zellen} Zellen): {len(stellen)}")
    fund = []
    for z in stellen:
        r, c = z[:, 0], z[:, 1]
        lon_f, lat_f = ch.nach_lonlat(crs, float(x_mitte[r, c].mean()),
                                      float(y_mitte[r, c].mean()))
        fund.append({
            "lat": round(lat_f, 5),
            "lon": round(lon_f, 5),
            "pixel": int(len(z)),
            "flaeche_m2": int(round(len(z) * kante_m ** 2)),
            "steigung_je_jahr": round(float(np.median(steigung[r, c])), 4),
            "mk_s_median": round(float(np.median(S[r, c])), 1),
            "mk_p_median": round(float(np.median(p_wert[r, c])), 4),
            "ndvi_start": round(float(stapel[0][r, c].mean()), 3),
            "ndvi_ende": round(float(stapel[-1][r, c].mean()), 3),
        })
    for i, f in enumerate(fund[:8], 1):
        print(f"  {i}. {f['lat']:.5f},{f['lon']:.5f} | {f['flaeche_m2']/1e4:5.2f} ha | "
              f"{f['steigung_je_jahr']:+.3f} NDVI/Jahr | MK p {f['mk_p_median']:.3f} | "
              f"{f['ndvi_start']:.2f} ({jahre[0]}) -> {f['ndvi_ende']:.2f} ({jahre[-1]})")

    # --- Vergroeberung: derselbe Trend auf groeberen Gittern ---
    def vergroebern(x2, f):
        h = x2.shape[0] // f * f; b = x2.shape[1] // f * f
        with np.errstate(invalid="ignore"):
            return np.nanmean(x2[:h, :b].reshape(h // f, f, b // f, f), axis=(1, 3))

    grob = []
    print("\nVergroeberung (Trend, bebaut + offen im Ausschnitt):")
    for f in (1, 3, 10):
        s2 = vergroebern(np.where(gebiet & gueltig, steigung, np.nan), f) if f > 1 \
            else np.where(gebiet & gueltig, steigung, np.nan)
        gg = np.isfinite(s2)
        fallend = gg & (s2 <= a.schwelle)
        n_cl = len(cluster(fallend, a.min_zellen))
        g_m = int(round(kante_m * f))
        grob.append({"gitter_m": g_m, "zellen": int(gg.sum()),
                     "anteil_fallend": round(float(fallend.sum()) / max(int(gg.sum()), 1), 4),
                     "cluster": n_cl})
        print(f"  {g_m:4d} m | {int(gg.sum()):7d} Zellen | fallend {fallend.sum()/max(gg.sum(),1):6.2%} | "
              f"{n_cl:4d} Fundstellen")

    # --- Überschneidung mit der Anomaliekarte ---
    # Seit V3 auch als JSON-Feld: „sieben von zwanzig" stand vorher nur auf
    # stdout — eine tragende Zahl ohne Datendatei verletzt die Belegpflicht.
    ueberschneidung = None
    if a.vergleich and a.vergleich.exists():
        d = json.loads(a.vergleich.read_text(encoding="utf-8"))
        treffer = 0
        for f in d["fundstellen"]:
            fx, fy = ch.punkt_nach(crs, f["lon"], f["lat"])
            r = int((bounds.top - fy) / d_y)
            c = int((fx - bounds.left) / d_x)
            if 0 <= r < hoehe and 0 <= c < breite and np.isfinite(steigung[r, c]) \
                    and steigung[r, c] <= a.schwelle:
                treffer += 1
        n = len(d["fundstellen"])
        ueberschneidung = {"treffer": treffer, "von": n,
                           "quelle": a.vergleich.name}
        print(f"\nVon {n} Fundstellen der Anomaliekarte liegen {treffer} auch im fallenden Trend "
              f"({treffer / n:.0%}).")
        print("  Wenig Überschneidung heißt: die beiden Karten beantworten verschiedene Fragen.")

    if a.json:
        # Beobachtungsdichte (§ 8.3) ueber alle Jahre und alle Pixel; das
        # Minimum ist ehrlicherweise oft 0 (Wolkenluecken am Rand).
        beobachtungen = None
        n_da = [lage for lage in n_lagen if lage is not None]
        if n_da:
            alle_n = np.concatenate([lage.ravel() for lage in n_da])
            beobachtungen = {"min": int(alle_n.min()),
                             "median": round(float(np.median(alle_n)), 1),
                             "max": int(alle_n.max())}
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "aoi": a.aoi, "jahre": jahre, "gitter_m": round(kante_m, 1),
            "crs": f"EPSG:{epsg}",
            "verfahren": "theil_sen_mann_kendall",
            "alpha": a.alpha,
            "fenster": saison_fenster_metadaten(jahre),
            "beobachtungen": beobachtungen,
            "schwellen": {"fallend_je_jahr": a.schwelle, "min_zellen": a.min_zellen,
                          "war_gruen": a.stadt_schwelle,
                          "stadtmaske": {"radius_m": a.stadt_radius_m,
                                         "schwelle": a.stadt_schwelle}},
            "kontext": gesammelt,
            "vergroeberung": grob,
            "ueberschneidung_anomalie": ueberschneidung,
            "fundstellen_gesamt": len(fund),
            "fundstellen": fund[:a.top],
            "aufruf": ch.aufruf_protokoll(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\ngeschrieben: {a.json}")


if __name__ == "__main__":
    main()
