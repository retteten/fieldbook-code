"""Metrisches Analyse-CRS — der gemeinsame Helfer aller Satelliten-Skripte.

Beschluss R1 (12.08.2026, docs/feldbuch-karten-rezeptur.md § 4.2): Analysen
laufen in einem metrischen CRS, nicht mehr auf einem geographischen Nenngitter.
Die bbox wird ins Ziel-CRS projiziert und nach AUSSEN auf das Analysegitter
gesnappt; damit ist jede Zelle exakt raster_m x raster_m gross und die
Zellflaeche exakt raster_m**2.

ZWEI dokumentierte Abweichungen von der Rezeptur-Notiz (§ 4.2), beide am
12.08.2026 gemessen, nicht vermutet:

1. Deutschland: WGS 84 / UTM 32N (EPSG:32632) statt ETRS89 / UTM 32N
   (EPSG:25832). Die CDSE Processing API weist 25832 ab — HTTP 500,
   woertlich: "Invalid envelope CRS. Expected: EPSG:32632, was: EPSG:25832".
   Sentinel Hub rechnet UTM nur im WGS-84-Datum (326xx/327xx) — das ist
   fachlich sogar konsequenter, denn Sentinel-2-Kacheln werden nativ in
   WGS-84-UTM ausgeliefert; nur so entfaellt Resampling wirklich.
   Projektionsparameter beider CRS sind identisch (Transverse Mercator,
   Zone 32, GRS80~WGS84); der Datumsversatz ETRS89<->WGS84 betraegt 2026
   unter 1 m (Plattendrift) — auf dem 20-m-Gitter ohne Belang. Wer fuer
   amtliche Weiterverarbeitung 25832 braucht, transformiert das fertige
   GeoTIFF (gdalwarp -t_srs EPSG:25832, sub-Meter-Shift).

2. Der Botswana-Punkt (25,85 Grad Ost) liegt in Zone 35 (24-30 Grad Ost),
   also EPSG:32735 — NICHT 32734, wie in der Rezeptur notiert. Zone 34
   haette dort ein Easting von ~994 km und ~0,26 % Massstabsfehler — genau
   die Ungenauigkeit, die R1 abschaffen soll.

Wer eine Zone erzwingen will, setzt "epsg" in der AOI-Definition (Achtung:
CDSE-Abrufe funktionieren nur mit 326xx/327xx, s. o.). Der alte Grad-Weg
unterschaetzte Flaechen um ~0,8-1 % — das faellt mit R1 weg.

Umsetzung OHNE pyproj: pyproj ist in der venv nicht installiert; rasterio
bringt aber eigene PROJ-Bindungen mit (rasterio.warp) — dieselbe
PROJ-Bibliothek, keine zusaetzliche Abhaengigkeit.

Zusaetzlich wohnt hier aufruf_protokoll(): die Kartenpass-Pflicht aus
Rezeptur § 8.1 (jeder JSON-Writer protokolliert den exakten Skriptaufruf).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from rasterio.crs import CRS
from rasterio.warp import transform as _warp_transform, transform_bounds

REPO = Path(__file__).resolve().parents[2]

WGS84 = CRS.from_epsg(4326)

# Deutschland fest auf UTM 32N — auch fuer AOIs, die rechnerisch in Zone 31
# oder 33 laegen (West-/Ostrand): ein Land, ein Gitter, wie es auch das
# amtliche EPSG:25832 haelt. WGS-84-Datum (32632) statt ETRS89 (25832), weil
# die CDSE-API nur 326xx/327xx rendert (s. Modul-Docstring, Abweichung 1).
# Die Grenzen sind bewusst grosszuegig; wer wirklich eine andere Zone braucht,
# setzt "epsg" in der AOI-Definition.
DEUTSCHLAND_EPSG = 32632
_DEUTSCHLAND_BBOX = (5.5, 47.2, 15.6, 55.1)  # W, S, O, N


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG-Code der UTM-Zone eines Punkts.

    Nordhalbkugel 326xx (WGS 84 / UTM zone xxN), Suedhalbkugel 327xx.
    Punkte in Deutschland liefern fest EPSG:32632 (WGS 84 / UTM 32N —
    warum nicht 25832, steht im Modul-Docstring, Abweichung 1).
    Beispiele: Oldenburg (8.2, 53.1) -> 32632;
    Botswana-Punkt (25.9, -24.0) -> 32735 (Zone 35S; s. Modul-Docstring,
    Abweichung 2).
    """
    w, s, o, n = _DEUTSCHLAND_BBOX
    if w <= lon <= o and s <= lat <= n:
        return DEUTSCHLAND_EPSG
    zone = min(60, max(1, int((lon + 180.0) // 6) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def aoi_epsg(aoi: dict) -> int:
    """EPSG einer AOI: explizites Feld "epsg" gewinnt, sonst aus der bbox-Mitte
    (Laenge/Breite) abgeleitet. Die AOI-Definition selbst bleibt lon/lat —
    sie ist nur die Gebietsbeschreibung, gemessen wird im Ziel-CRS."""
    if aoi.get("epsg"):
        return int(aoi["epsg"])
    w, s, o, n = aoi["bbox"]
    return utm_epsg((w + o) / 2.0, (s + n) / 2.0)


def bbox_nach_utm(bbox_lonlat, epsg: int, raster_m: float):
    """Geographische bbox -> projizierte bbox auf dem Analysegitter.

    Liefert (bbox_projiziert, breite_px, hoehe_px). Gesnappt wird nach AUSSEN
    (floor auf West/Sued, ceil auf Ost/Nord): der angefragte Ausschnitt liegt
    vollstaendig im Raster, und jede Zelle misst exakt raster_m x raster_m —
    Zellflaeche = raster_m**2, ohne Naeherungskonstante.
    """
    w, s, o, n = transform_bounds(WGS84, CRS.from_epsg(epsg), *bbox_lonlat)
    w = math.floor(w / raster_m) * raster_m
    s = math.floor(s / raster_m) * raster_m
    o = math.ceil(o / raster_m) * raster_m
    n = math.ceil(n / raster_m) * raster_m
    breite = max(1, int(round((o - w) / raster_m)))
    hoehe = max(1, int(round((n - s) / raster_m)))
    return (w, s, o, n), breite, hoehe


def _als_crs(ziel) -> CRS:
    """EPSG-Nummer oder rasterio-CRS -> rasterio-CRS."""
    return ziel if isinstance(ziel, CRS) else CRS.from_epsg(int(ziel))


def punkt_nach(ziel, lon: float, lat: float) -> tuple[float, float]:
    """Einen Punkt (lon/lat, WGS 84) ins Ziel-CRS projizieren -> (x, y)."""
    xs, ys = _warp_transform(WGS84, _als_crs(ziel), [lon], [lat])
    return float(xs[0]), float(ys[0])


def nach_lonlat(quelle, x: float, y: float) -> tuple[float, float]:
    """Einen Punkt aus dem Quell-CRS zurueck nach lon/lat (WGS 84)."""
    lons, lats = _warp_transform(_als_crs(quelle), WGS84, [x], [y])
    return float(lons[0]), float(lats[0])


def bbox_nach(ziel, bbox_lonlat):
    """Geographische bbox (W, S, O, N) ins Ziel-CRS — OHNE Gitter-Snap.

    Fuer Ausschnitts-Filter und Fensterrechnungen auf bestehenden Rastern.
    """
    return transform_bounds(WGS84, _als_crs(ziel), *bbox_lonlat)


def bbox_nach_lonlat(quelle, bbox_xy):
    """Projizierte bbox (W, S, O, N — auch rasterio-BoundingBox) zurueck
    nach lon/lat (WGS 84), z. B. fuer lesbare JSON-Metadaten."""
    return transform_bounds(_als_crs(quelle), WGS84, *bbox_xy)


def crs_url(epsg: int) -> str:
    """CRS-Angabe fuer die CDSE Processing API (OGC-URI-Form)."""
    return f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"


def pruefe_metrisch(quelle, name: str = "") -> int:
    """Stellt sicher, dass ein geoeffnetes Raster im metrischen CRS liegt.

    Altbestaende aus der Zeit vor R1 liegen in EPSG:4326 — auf ihnen waere
    jede Meter-Rechnung dieses Codes lagefalsch. Lieber laut scheitern als
    still verschieben. Liefert den EPSG-Code des Rasters.
    """
    crs = quelle.crs
    if crs is None or crs.is_geographic:
        raise SystemExit(
            f"{name or quelle.name}: Raster liegt im Grad-Gitter "
            f"({crs}) — Altbestand vor Beschluss R1 (12.08.2026). "
            f"Erst metrisch neu rechnen (ndvi_batch/duerre_baumschulen), "
            f"dann dieses Skript.")
    return int(crs.to_epsg() or 0)


def aufruf_protokoll() -> str:
    """Der exakte Kommandoaufruf dieses Laufs (Kartenpass, Rezeptur § 8.1).

    Aus sys.argv rekonstruiert: genau die Flags, mit denen der Lauf gestartet
    wurde — der veroeffentlichte Lauf muss ohne Raten wiederholbar sein.
    Der Skriptpfad wird auf Repo-Schreibweise normalisiert, Argumente mit
    Leerzeichen werden gequotet.
    """
    skript = Path(sys.argv[0]).resolve()
    try:
        skript_text = skript.relative_to(REPO).as_posix()
    except ValueError:
        skript_text = skript.name
    teile = ["python", skript_text]
    for arg in sys.argv[1:]:
        teile.append(f'"{arg}"' if (" " in arg or not arg) else arg)
    return " ".join(teile)
