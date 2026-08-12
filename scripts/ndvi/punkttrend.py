"""Veraenderung statt Unterschied: Theil-Sen-Steigung je Bildpunkt.

Warum nicht die Differenz zweier Mediane
----------------------------------------
punktauswertung.py beantwortet "ist der Kern anders als sein Umland" — und die
Antwort war ja, seit zehn Jahren. Das ist ein Zustand, keine Veraenderung. Wer
wissen will, ob sich etwas TUT, braucht eine Steigung, und zwar je Bildpunkt:
Ein Mittelwert ueber 2.000 Punkte verwischt genau den Fall, um den es geht —
eine Teilflaeche, die sich erholt, waehrend der Rest gleich bleibt.

Die Methode
-----------
1. Je Jahr die ABWEICHUNG jedes Bildpunkts vom Median des Kontrollrings.
   Damit faellt das Regenjahr heraus: 2024 war fuer alle schlecht, 2025 fuer
   alle gut. Ohne diesen Schritt misst die Steigung die Niederschlagsreihe.
2. Theil-Sen-Steigung ueber die zehn Jahresabweichungen. Median aller
   paarweisen Steigungen — bricht nicht ein, wenn ein Jahr ausreisst
   (Wolkenrest, Brand, verspaeteter Regen). Kleinste Quadrate taeten das.
3. Mann-Kendall auf dieselbe Reihe: Ist die Richtung mehr als Zufall? Der Test
   verlangt weder Normalverteilung noch gleiche Abstaende und ist der
   Standardtest fuer Umweltzeitreihen.
4. Erst danach zusammenfassen — und getrennt fuer Flaechen, die sich
   unterscheiden, statt ueber sie hinweg zu mitteln.

Was auch das nicht kann
-----------------------
Zehn Werte sind zehn Werte. Mann-Kendall braucht bei n=10 mindestens |S|>=21
fuer p<0,05 — das heisst, ein Bildpunkt muss ueber die Jahre ziemlich
konsequent in eine Richtung laufen. Schwache echte Trends bleiben unerkannt.
Und eine Steigung sagt weiterhin nicht, WARUM.

Aufruf:
    python punkttrend.py --lat -23.973037 --lon 25.854650 --name botswana
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndvi_batch as nb  # noqa: E402
import crs_helfer as ch  # noqa: E402  (metrisches Analyse-CRS, R1)
import punktauswertung as pa  # noqa: E402
import punktbilder as pb  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AUSGABE = REPO / "docs" / "daten"
BILDER = AUSGABE / "punktbilder"


def theil_sen(stapel: np.ndarray, jahre: np.ndarray) -> np.ndarray:
    """Median aller paarweisen Steigungen, je Bildpunkt.

    stapel: (jahre, hoehe, breite). Paare mit NaN fallen heraus, weil nanmedian
    sie ignoriert — ein Bildpunkt mit zwei brauchbaren Jahren bekommt also eine
    Steigung, aber Mann-Kendall wird ihn gleich darauf verwerfen.
    """
    n = len(jahre)
    steigungen = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dt = float(jahre[j] - jahre[i])
            steigungen.append((stapel[j] - stapel[i]) / dt)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(np.stack(steigungen), axis=0)


def mann_kendall(stapel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S-Statistik und zweiseitiger p-Wert je Bildpunkt (Normalapproximation).

    Bindungskorrektur bleibt weg: NDVI ist stetig, echte Bindungen sind bei
    float32 praktisch ausgeschlossen.
    """
    n = stapel.shape[0]
    S = np.zeros(stapel.shape[1:], dtype=np.float64)
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += np.sign(stapel[j] - stapel[i])
    var = n * (n - 1) * (2 * n + 5) / 18.0
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(S > 0, (S - 1) / math.sqrt(var),
                     np.where(S < 0, (S + 1) / math.sqrt(var), 0.0))
        # zweiseitig, ueber die Fehlerfunktion statt scipy
        p = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(z) / math.sqrt(2))))
    return S, np.clip(p, 0.0, 1.0)


def trendfarbe(steigung: np.ndarray, signifikant: np.ndarray) -> np.ndarray:
    """Bernstein = Rueckgang, Pinie = Zunahme, Papier = kein Trend.

    Nicht signifikante Bildpunkte werden BEWUSST blass gezeichnet statt
    weggelassen: Eine Karte, die nur Signifikantes zeigt, sieht nach mehr
    Befund aus, als die Daten hergeben.
    """
    x = np.clip(np.nan_to_num(steigung, nan=0.0) / 0.02, -1, 1)  # ±0,02/Jahr = Vollton
    papier = np.array([246, 244, 238], dtype=float)
    bernstein = np.array([232, 161, 60], dtype=float)
    pinie = np.array([44, 138, 107], dtype=float)
    t = np.abs(x)[..., None]
    ziel = np.where(x[..., None] < 0, bernstein, pinie)
    rgb = papier * (1 - t) + ziel * t
    # Nicht signifikant: auf ein Viertel der Saettigung zurueckziehen.
    blass = papier + (rgb - papier) * 0.25
    return np.clip(np.where(signifikant[..., None], rgb, blass), 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--kern-m", type=float, default=500.0)
    ap.add_argument("--ring-von-m", type=float, default=800.0)
    ap.add_argument("--ring-bis-m", type=float, default=1500.0)
    ap.add_argument("--von-jahr", type=int, default=2017)
    ap.add_argument("--bis-jahr", type=int, default=date.today().year)
    ap.add_argument("--aufloesung", type=float, default=20.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    halbkante = a.ring_bis_m * 1.15
    bbox = pa.bbox_um(a.lat, a.lon, halbkante)
    aoi = {"id": a.name, "bbox": bbox, "aufloesung_m": a.aufloesung}
    # Metrisches Analysegitter (R1): dieselbe Projektion wie punktauswertung,
    # damit beide Skripte pixelgenau dieselben Rohkacheln teilen.
    epsg, bbox_utm, _, _ = nb.aoi_gitter(aoi)
    x0, y0 = ch.punkt_nach(epsg, a.lon, a.lat)
    lauf = pa.SchlankerLauf()

    jahre, lagen, n_lagen = [], [], []
    for jahr in range(a.von_jahr, a.bis_jahr + 1):
        daten, _ = pa.hole_jahr(lauf, aoi, jahr, False, cache_dir=pa.CACHE)
        ndvi, n = daten
        ndvi = np.where(np.isfinite(ndvi) & (n >= 2), ndvi, np.nan)
        jahre.append(jahr); lagen.append(ndvi); n_lagen.append(n)
    stapel = np.stack(lagen).astype(np.float64)
    jahre = np.array(jahre)

    abstand = pa.abstandsraster(bbox_utm, stapel.shape[1:], x0, y0)
    kern = abstand <= a.kern_m
    ring = (abstand >= a.ring_von_m) & (abstand <= a.ring_bis_m)

    # --- Schritt 1: Jahreseffekt herausnehmen -----------------------------
    ringmedian = np.array([np.nanmedian(l[ring]) for l in stapel])
    anomalie = stapel - ringmedian[:, None, None]

    # --- Schritt 2+3: Steigung und Signifikanz ----------------------------
    print(f"Trendanalyse {a.name}, {len(jahre)} Jahre, "
          f"{stapel.shape[1]}x{stapel.shape[2]} Bildpunkte")
    print("  Theil-Sen und Mann-Kendall werden gerechnet …")
    steigung = theil_sen(anomalie, jahre)
    S, p = mann_kendall(anomalie)
    genug = np.sum(np.isfinite(anomalie), axis=0) >= 8
    signifikant = genug & np.isfinite(p) & (p < a.alpha)

    # --- Schritt 4: erst jetzt zusammenfassen -----------------------------
    def fasse(maske, label):
        m = maske & genug & np.isfinite(steigung)
        if m.sum() < 5:
            print(f"  {label:<26} zu wenig auswertbare Punkte")
            return None
        st = steigung[m]
        zu = int((signifikant & m & (steigung > 0)).sum())
        ab = int((signifikant & m & (steigung < 0)).sum())
        print(f"  {label:<26} Median {np.median(st):+.5f}/Jahr   "
              f"signifikant: {zu} zunehmend, {ab} abnehmend, "
              f"{m.sum()-zu-ab} ohne Trend  ({100*(zu+ab)/m.sum():.1f} % mit Trend)")
        return {"median_steigung": round(float(np.median(st)), 5),
                "punkte": int(m.sum()), "signifikant_zunahme": zu,
                "signifikant_abnahme": ab,
                "anteil_mit_trend": round(float((zu + ab) / m.sum()), 4)}

    print()
    z_kern = fasse(kern, "Kern (500 m)")
    z_ring = fasse(ring, "Kontrollring")
    # Die innere, auffaellig helle Flaeche: dauerhaft unter dem Ring.
    dauerhaft_arm = np.nanmean(anomalie, axis=0) < -0.03
    z_innen = fasse(kern & dauerhaft_arm, "davon dauerhaft aermer")
    z_rest = fasse(kern & ~dauerhaft_arm, "davon unauffaellig")

    # --- Bild --------------------------------------------------------------
    BILDER.mkdir(parents=True, exist_ok=True)
    bild = Image.fromarray(trendfarbe(steigung, signifikant))
    bild = bild.resize((900, 900), Image.NEAREST)
    d = ImageDraw.Draw(bild, "RGBA")
    # Kreismittelpunkt exakt aus dem metrischen Gitter: nach dem Gitter-Snap
    # liegt der Punkt nicht mehr genau in der Bildmitte (Versatz <= 1 Zelle).
    w_utm, s_utm, o_utm, n_utm = bbox_utm
    mx = (x0 - w_utm) / (o_utm - w_utm) * bild.width
    my = (n_utm - y0) / (n_utm - s_utm) * bild.height
    for radius_m, farbe, breite in ((a.kern_m, (20, 20, 20, 255), 3),
                                    (a.ring_von_m, (20, 20, 20, 90), 1),
                                    (a.ring_bis_m, (20, 20, 20, 90), 1)):
        r = radius_m / (o_utm - w_utm) * bild.width
        d.ellipse([mx - r, my - r, mx + r, my + r], outline=farbe, width=breite)
    ziel = BILDER / f"{a.name}-trend.png"
    bild.save(ziel)
    print(f"\n  Karte: {ziel.relative_to(REPO)}")

    ergebnis = {
        "erzeugt": nb.jetzt_utc(),
        "aufruf": ch.aufruf_protokoll(),
        "punkt": {"lat": a.lat, "lon": a.lon},
        "crs": f"EPSG:{epsg}",
        "aufloesung_m": a.aufloesung,
        "fenster": pa.fenster_text(),
        "jahre": [int(j) for j in jahre],
        "methode": (
            "Je Jahr die Abweichung jedes Bildpunkts vom Median des "
            "Kontrollrings (entfernt den Jahreseffekt, also den Regen). "
            "Darauf Theil-Sen-Steigung (Median aller paarweisen Steigungen, "
            "robust gegen einzelne Ausreisserjahre) und Mann-Kendall-Test "
            "(verteilungsfrei). Bildpunkte mit weniger als acht auswertbaren "
            "Jahren bleiben aussen vor."
        ),
        "alpha": a.alpha,
        "grenzen": [
            "Zehn Jahreswerte sind wenig: Mann-Kendall verlangt bei n=10 ein "
            "|S| von mindestens 21 fuer p<0,05. Schwache echte Trends bleiben "
            "unerkannt — ein 'kein Trend' heisst hier 'nicht nachweisbar', "
            "nicht 'nicht vorhanden'.",
            "Die Steigung sagt nicht, WARUM sich etwas veraendert.",
            "Kein Test auf Bruchpunkte: eine Stufe in der Reihe erscheint hier "
            "als flacher Trend.",
        ],
        "quellenvermerk": f"Contains modified Copernicus Sentinel data {date.today().year}",
        "kern": z_kern, "ring": z_ring,
        "kern_dauerhaft_aermer": z_innen, "kern_unauffaellig": z_rest,
    }
    ziel = AUSGABE / f"punkttrend-{a.name}.json"
    ziel.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Zahlen: {ziel.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
