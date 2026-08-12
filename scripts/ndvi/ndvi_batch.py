#!/usr/bin/env python3
"""NDVI-Monitor — lokale Rechenwerkstatt (Batch).

Erzeugt alle Artefakte des Datenvertrags (scripts/ndvi/README.md) im
Dateilayout ftp-mirror-geophora/tiles/ndvi/. Der wöchentliche Nachschub
läuft über php/ndvi-refresh.php — dieses Skript ist für die teuren,
einmaligen Läufe (Baseline, Historie, Persistenz, PCA) und für die
synthetische Demo ohne CDSE-Zugang gedacht.

Aufrufbeispiele:
    python ndvi_batch.py demo
    python ndvi_batch.py alles
    python ndvi_batch.py historie --aoi harz --jahr 2021
    python ndvi_batch.py fensterbaseline --aoi harz
"""

import argparse
import datetime as dt
import json
import math
import sys
import time
import warnings
from pathlib import Path

try:
    import numpy as np
    import requests
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    from PIL import Image, ImageDraw, ImageFont
except ImportError as fehler:
    sys.exit(
        f"Abhängigkeit fehlt ({fehler.name}). Installation:\n"
        "  python -m venv scripts/ndvi/.venv\n"
        "  scripts/ndvi/.venv/Scripts/pip install -r scripts/ndvi/requirements.txt"
    )

# Metrisches Analyse-CRS (Beschluss R1, 12.08.2026) — gemeinsamer Helfer der
# ganzen Werkstatt; liegt im selben Ordner, braucht nur rasterio (oben geprüft).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crs_helfer as ch  # noqa: E402

# ---------------------------------------------------------------------------
# Konstanten & Pfade
# ---------------------------------------------------------------------------

SKRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SKRIPT_DIR.parent.parent
STANDARD_AUSGABE = REPO_DIR / "ftp-mirror-geophora" / "tiles" / "ndvi"
CACHE_DIR = SKRIPT_DIR / "cache"          # lokaler Zwischenspeicher, nie deployen
ENV_DATEI = SKRIPT_DIR / ".env"

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
MAX_WOLKEN = 75
MAX_KANTE = 2500
QUELLE_ECHT = "Sentinel-2 L2A · Copernicus Data Space Ecosystem"
QUELLE_DEMO = "Synthetische Demo-Daten (kein Satellitenbild)"

TYP_GEOTIFF = "image/tiff; application=geotiff"
TYP_PNG = "image/png"
TYP_JSON = "application/json"

# Persistenz-Klassenfarben aus dem Vertrag; 0/1 bleiben transparent.
PERSISTENZ_FARBEN = {2: (31, 157, 85), 3: (126, 217, 87), 4: (217, 79, 43)}

# Artefakte, die der STAC-Bauer im AOI-Ordner einsammelt:
# artefakt -> Liste (asset_name, dateiname, medientyp, rollen)
ARTEFAKT_ASSETS = {
    "aktuell": [("tif", "aktuell.tif", TYP_GEOTIFF, ["data"]),
                ("png", "aktuell.png", TYP_PNG, ["visual"])],
    "aktuell-analyse": [("tif", "aktuell-analyse.tif", TYP_GEOTIFF, ["data"])],
    "baseline": [("tif", "baseline.tif", TYP_GEOTIFF, ["data"]),
                 ("png", "baseline.png", TYP_PNG, ["visual"])],
    "baseline-fenster": [("tif", "baseline-fenster.tif", TYP_GEOTIFF, ["data"]),
                         ("png", "baseline-fenster.png", TYP_PNG, ["visual"]),
                         ("json", "fenster-baseline.json", TYP_JSON, ["metadata"])],
    "persistenz": [("tif", "persistenz.tif", TYP_GEOTIFF, ["data"]),
                   ("png", "persistenz.png", TYP_PNG, ["visual"])],
    "pca": [("png", "pca.png", TYP_PNG, ["visual"]),
            ("json", "pca.json", TYP_JSON, ["metadata"])],
    "falsecolor-aktuell": [("png", "falsecolor-aktuell.png", TYP_PNG, ["visual"])],
    "falsecolor-2021": [("png", "falsecolor-2021.png", TYP_PNG, ["visual"])],
    "composite-2021": [("tif", "composite-2021.tif", TYP_GEOTIFF, ["data"]),
                       ("png", "composite-2021.png", TYP_PNG, ["visual"])],
}


# ---------------------------------------------------------------------------
# Kleine Helfer
# ---------------------------------------------------------------------------

def jetzt_utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lies_json(pfad):
    if not pfad.exists():
        return None
    return json.loads(pfad.read_text(encoding="utf-8"))


def schreibe_json(pfad, daten):
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps(daten, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def iso_wochenlabel(datum):
    jahr, woche, _ = datum.isocalendar()
    return f"{jahr}-W{woche:02d}"


def aoi_gitter(aoi, aufloesung_m=None):
    """(EPSG, projizierte bbox, Breite, Höhe) des metrischen Analysegitters.

    Beschluss R1 (12.08.2026): Die AOI-bbox (lon/lat, bleibt die Quelle der
    Gebietsdefinition) wird in die passende UTM-Zone projiziert — Deutschland
    fest EPSG:25832 — und nach außen auf das Gitter gesnappt. Jede Zelle misst
    exakt aufloesung_m × aufloesung_m; die Zellfläche ist exakt aufloesung_m².
    Die alten Näherungskonstanten (111320·cos, 110574) sind damit Geschichte.

    Kanten über MAX_KANTE sind ein Konfigurationsfehler und brechen laut ab:
    stilles Klemmen (wie früher) würde die Zellgröße ändern und damit jede
    Flächenrechnung verfälschen. Keine der bestehenden AOIs reißt die Grenze.
    """
    epsg = ch.aoi_epsg(aoi)
    bbox_utm, breite, hoehe = ch.bbox_nach_utm(
        aoi["bbox"], epsg, aufloesung_m or aoi["aufloesung_m"])
    if max(breite, hoehe) > MAX_KANTE:
        raise RuntimeError(
            f"AOI {aoi.get('id', '?')}: {breite}x{hoehe} px überschreitet "
            f"MAX_KANTE={MAX_KANTE} — bbox verkleinern oder Auflösung "
            f"vergröbern.")
    return epsg, bbox_utm, breite, hoehe


def gitter_von(aoi, aufloesung_m=None):
    """Kurzform für die Schreibwege: (projizierte bbox, EPSG)."""
    epsg, bbox_utm, _, _ = aoi_gitter(aoi, aufloesung_m)
    return bbox_utm, epsg


def raster_groesse(bbox, aufloesung_m):
    """Pixelmaße des metrischen Analysegitters — delegiert an aoi_gitter (R1).

    ACHTUNG Frontend: Die ausgelieferten GeoTIFFs liegen nach der Umstellung
    in UTM (EPSG steht in meta.json), nicht mehr als lineares Grad-Raster —
    die Klick-Umrechnung in layer-explorer/ndvi.html muss den GeoTIFF-
    Transform lesen statt die bbox linear zu teilen."""
    _, _, breite, hoehe = aoi_gitter({"bbox": list(bbox),
                                      "aufloesung_m": aufloesung_m})
    return breite, hoehe


def saison_fenster(jahr, saison):
    """Vegetationsperiode als Datumsfenster: Montag der Startwoche bis
    Sonntag der Endwoche (ISO-Wochen aus aois.json)."""
    von = dt.date.fromisocalendar(jahr, saison["start_woche"], 1)
    bis = dt.date.fromisocalendar(jahr, saison["end_woche"], 7)
    return von, bis


def kalender_fenster(jahr, ende, tage=14):
    """Dasselbe Kalenderfenster in einem früheren Jahr (M3): nur der Endtag
    wird ins Zieljahr gesetzt, der Anfang ergibt sich 14 Tage davor. So bleibt
    die Fensterlänge exakt gleich, und Fenster über den Jahreswechsel rutschen
    korrekt mit. 29.02. fällt in Nicht-Schaltjahren auf den 28.02."""
    try:
        bis = ende.replace(year=jahr)
    except ValueError:
        bis = dt.date(jahr, 2, 28)
    return bis - dt.timedelta(days=tage), bis


def saison_monate_rueckwaerts(heute, anzahl, saison):
    """Die letzten `anzahl` Saison-Monate (Mai–Sep), rückwärts geblättert;
    Wintermonate werden übersprungen. Neuester Monat zuerst."""
    monate = []
    jahr, monat = heute.year, heute.month
    while len(monate) < anzahl:
        if saison["start_monat"] <= monat <= saison["end_monat"]:
            monate.append((jahr, monat))
        monat -= 1
        if monat == 0:
            jahr, monat = jahr - 1, 12
    return monate


def monats_fenster(jahr, monat, heute):
    """Erster bis letzter Tag des Monats; der laufende Monat endet heute."""
    von = dt.date(jahr, monat, 1)
    if monat == 12:
        bis = dt.date(jahr, 12, 31)
    else:
        bis = dt.date(jahr, monat + 1, 1) - dt.timedelta(days=1)
    return von, min(bis, heute)


# ---------------------------------------------------------------------------
# .env & OAuth
# ---------------------------------------------------------------------------

def lade_env(pfad):
    """Minimaler KEY=WERT-Parser; #-Zeilen und Leerzeilen werden ignoriert."""
    werte = {}
    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        zeile = zeile.strip()
        if not zeile or zeile.startswith("#") or "=" not in zeile:
            continue
        schluessel, _, wert = zeile.partition("=")
        werte[schluessel.strip()] = wert.strip()
    return werte


def lade_zugangsdaten():
    if not ENV_DATEI.exists():
        sys.exit(
            "CDSE-Zugangsdaten fehlen: scripts/ndvi/.env nicht gefunden.\n"
            "Anlegen nach Muster von scripts/ndvi/env-VORLAGE.txt "
            "(CDSE_CLIENT_ID + CDSE_CLIENT_SECRET).\n"
            "Ohne Zugang funktioniert nur:  python ndvi_batch.py demo"
        )
    env = lade_env(ENV_DATEI)
    client_id = env.get("CDSE_CLIENT_ID", "")
    client_secret = env.get("CDSE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        sys.exit("scripts/ndvi/.env unvollständig: CDSE_CLIENT_ID und "
                 "CDSE_CLIENT_SECRET müssen gesetzt sein (s. env-VORLAGE.txt).")
    return client_id, client_secret


def hole_token(client_id, client_secret):
    """Token holen — wird je Lauf wiederverwendet und erst kurz vor Ablauf
    erneuert (Vertrag: nie pro Request neu holen, der Endpoint ist
    rate-limited). Liefert (Token, Lebensdauer in Sekunden)."""
    try:
        antwort = requests.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials",
                  "client_id": client_id, "client_secret": client_secret},
            timeout=30,
        )
    except requests.RequestException as fehler:
        sys.exit(f"OAuth-Endpoint nicht erreichbar: {fehler}")
    if antwort.status_code != 200:
        sys.exit(f"OAuth fehlgeschlagen (HTTP {antwort.status_code}): "
                 f"{antwort.text[:300]}")
    daten = antwort.json()
    try:
        lebensdauer = float(daten.get("expires_in", 600))
    except (TypeError, ValueError):
        lebensdauer = 600.0  # CDSE-Standard: 10 Minuten
    return daten["access_token"], lebensdauer


# ---------------------------------------------------------------------------
# Evalscripts (Vertragskonventionen: ORBIT-Mosaik, SCL-Maske, eigener Median)
# ---------------------------------------------------------------------------

_JS_BASIS = """//VERSION=3
// SCL-Klassen 3 (Schatten), 8/9 (Wolken), 10 (Zirrus), 11 (Schnee) verwerfen.
var SCL_VERWORFEN = [3, 8, 9, 10, 11];
function gueltig(probe) {
  return probe.dataMask === 1 && SCL_VERWORFEN.indexOf(probe.SCL) === -1;
}
// Median selbst gebaut: sammeln, sortieren, Mittelelement (Vertrag).
function median(werte) {
  if (werte.length === 0) return NaN;
  werte.sort(function (a, b) { return a - b; });
  var mitte = Math.floor(werte.length / 2);
  return werte.length % 2 === 1 ? werte[mitte]
                                : 0.5 * (werte[mitte - 1] + werte[mitte]);
}
function ndviWerte(proben) {
  var werte = [];
  for (var i = 0; i < proben.length; i++) {
    var p = proben[i];
    if (!gueltig(p)) continue;
    var nenner = p.B08 + p.B04;
    if (nenner > 0) werte.push((p.B08 - p.B04) / nenner);
  }
  return werte;
}
"""


def evalscript_ndvi_tiff():
    """NDVI-Median als FLOAT32, NoData = NaN — zweibandig.

    Band 2 (`n`) zählt die gültigen, also wolkenfreien Beobachtungen, aus
    denen der Median entstand. Ohne diese Zahl ist ein Pixelwert nicht
    einzuordnen: 0,42 aus vier Aufnahmen ist eine Messung, 0,42 aus einer
    einzigen ein Zufallstreffer zwischen zwei Wolken. n = 0 ⇒ NDVI = NaN."""
    return _JS_BASIS + """
function setup() {
  return {
    input: [{bands: ["B04", "B08", "SCL", "dataMask"]}],
    output: {bands: 2, sampleType: "FLOAT32"},
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(samples) {
  var werte = ndviWerte(samples);
  // median() liefert bei leerer Liste NaN — n = 0 und NDVI = NaN passen
  // damit automatisch zusammen.
  return [median(werte), werte.length];
}
"""


def evalscript_ndvi_png(farbskala):
    """NDVI-Median als eingefärbtes RGBA-PNG. Die Stützstellen aus aois.json
    werden hart ins Evalscript eingebettet, damit Server und Batch dieselbe
    Skala rechnen; außerhalb der Domäne wird geklemmt, NoData ist transparent."""
    schablone = _JS_BASIS + """
var STUETZEN = __STUETZEN__;
var DOM_MIN = __DMIN__, DOM_MAX = __DMAX__;
function farbe(wert) {
  var v = Math.min(Math.max(wert, DOM_MIN), DOM_MAX);
  if (v <= STUETZEN[0][0]) {
    return [STUETZEN[0][1], STUETZEN[0][2], STUETZEN[0][3]];
  }
  for (var i = 1; i < STUETZEN.length; i++) {
    if (v <= STUETZEN[i][0]) {
      var a = STUETZEN[i - 1], b = STUETZEN[i];
      var t = (v - a[0]) / (b[0] - a[0]);
      return [Math.round(a[1] + t * (b[1] - a[1])),
              Math.round(a[2] + t * (b[2] - a[2])),
              Math.round(a[3] + t * (b[3] - a[3]))];
    }
  }
  var e = STUETZEN[STUETZEN.length - 1];
  return [e[1], e[2], e[3]];
}
function setup() {
  return {
    input: [{bands: ["B04", "B08", "SCL", "dataMask"]}],
    output: {bands: 4, sampleType: "UINT8"},
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(samples) {
  var wert = median(ndviWerte(samples));
  if (isNaN(wert)) return [0, 0, 0, 0];
  var rgb = farbe(wert);
  return [rgb[0], rgb[1], rgb[2], 255];
}
"""
    return (schablone
            .replace("__STUETZEN__", json.dumps(farbskala["stuetzstellen"]))
            .replace("__DMIN__", str(farbskala["domaene"][0]))
            .replace("__DMAX__", str(farbskala["domaene"][1])))


def evalscript_falsecolor():
    """False-Color-Median (R=B08, G=B04, B=B03), je Band 2.5×Reflektanz
    geklemmt, UINT8; Pixel ohne gültigen Orbit transparent."""
    return _JS_BASIS + """
function bandMedian(proben, band) {
  var werte = [];
  for (var i = 0; i < proben.length; i++) {
    if (gueltig(proben[i])) werte.push(proben[i][band]);
  }
  return werte;
}
function strecke(wert) {
  return Math.round(255 * Math.min(Math.max(2.5 * wert, 0), 1));
}
function setup() {
  return {
    input: [{bands: ["B03", "B04", "B08", "SCL", "dataMask"]}],
    output: {bands: 4, sampleType: "UINT8"},
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(samples) {
  var nir = bandMedian(samples, "B08");
  if (nir.length === 0) return [0, 0, 0, 0];
  return [strecke(median(nir)),
          strecke(median(bandMedian(samples, "B04"))),
          strecke(median(bandMedian(samples, "B03"))),
          255];
}
"""


# ---------------------------------------------------------------------------
# Processing-API
# ---------------------------------------------------------------------------

def analyse_aufloesung(aoi):
    """Auflösung der mehrjährigen Analyse-Artefakte (Historie/Baseline/
    Persistenz/PCA/falsecolor-2021). Bewusst gröber als der Wochen-Composite:
    Saison-Requests skalieren mit der Orbitzahl — der erste Echtlauf hat
    1730 PU für EINEN 10-m-Saison-Composite gemessen; 20 m senkt das auf
    rund ein Viertel, ohne die Aussage der Mehrjahres-Analyse zu ändern."""
    return aoi.get("aufloesung_analyse_m", aoi["aufloesung_m"])


def process_roh(lauf, aoi, von, bis, evalscript, format_typ, breite, hoehe,
                max_wolken=None, mosaik_reihenfolge=None):
    """Wie process(), aber mit FESTEN Pixelmaßen statt einer Auflösung.

    Gebraucht für Anschauungsbilder (echtbild_ersatz.py), die exakt die Maße
    vorhandener Seitenbilder treffen müssen — dort zählt der Bildausschnitt,
    nicht die Bodenauflösung.
    """
    return process(lauf, aoi, von, bis, evalscript, format_typ,
                   feste_masse=(breite, hoehe), max_wolken=max_wolken,
                   mosaik_reihenfolge=mosaik_reihenfolge)


def process(lauf, aoi, von, bis, evalscript, format_typ, aufloesung_m=None,
            feste_masse=None, max_wolken=None, mosaik_reihenfolge=None):
    """Ein Bild-Output je Request (Vertrag, kein TAR). Retry genau einmal
    bei 401/429/5xx — bei 401 mit frischem Token, denn CDSE-Tokens laufen
    nach ~10 Minuten ab; PU-Verbrauch aus dem Antwort-Header wird je AOI
    verbucht.

    Seit R1 (12.08.2026) laufen alle Gitter-Requests metrisch: bbox im
    UTM-Ziel-CRS der AOI, CRS als OGC-URI (http://www.opengis.net/def/crs/
    EPSG/0/<code>) — GeoTIFF-Antworten kommen dann mit exakt diesem CRS und
    Transform zurück. Nur der feste_masse-Weg (Anschauungsbilder, die exakte
    Pixelmaße vorhandener Seitenbilder treffen müssen, s. process_roh/
    echtbild_ersatz.py) bleibt bei CRS84: dort zählt der Bildausschnitt,
    nicht die Messgeometrie."""
    if feste_masse:
        breite, hoehe = feste_masse
        grenzen = {"bbox": list(aoi["bbox"]),
                   "properties": {"crs": "http://www.opengis.net/def/crs/"
                                         "OGC/1.3/CRS84"}}
    else:
        epsg, bbox_utm, breite, hoehe = aoi_gitter(aoi, aufloesung_m)
        grenzen = {"bbox": list(bbox_utm),
                   "properties": {"crs": ch.crs_url(epsg)}}
    anfrage = {
        "input": {
            "bounds": grenzen,
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": f"{von.isoformat()}T00:00:00Z",
                                  "to": f"{bis.isoformat()}T23:59:59Z"},
                    "maxCloudCoverage": max_wolken or MAX_WOLKEN,
                    # leastCC: CDSE waehlt je Pixel die wolkenaermste Szene —
                    # noetig fuer Uebersichtsbilder, wo ein harter Wolkenfilter
                    # ganze Kacheln leer (schwarz) zuruecklassen wuerde.
                    **({"mosaickingOrder": mosaik_reihenfolge}
                       if mosaik_reihenfolge else {}),
                },
            }],
        },
        "output": {
            "width": breite, "height": hoehe,
            "responses": [{"identifier": "default",
                           "format": {"type": format_typ}}],
        },
        "evalscript": evalscript,
    }
    antwort = None
    for versuch in (1, 2):
        # Header je Versuch neu bauen: token() erneuert abgelaufene Tokens,
        # und nach einem 401 wäre der alte Bearer-Wert ohnehin unbrauchbar.
        kopf = {"Authorization": f"Bearer {lauf.token()}",
                "Accept": format_typ}
        antwort = requests.post(PROCESS_URL, headers=kopf, json=anfrage,
                                timeout=300)
        if antwort.status_code == 200:
            break
        if versuch == 1 and antwort.status_code == 401:
            lauf._token = None  # Token serverseitig abgelaufen → neu holen
            continue
        if versuch == 1 and (antwort.status_code == 429
                             or antwort.status_code >= 500):
            time.sleep(15)  # kurzer Atemzug, dann genau ein zweiter Versuch
            continue
        raise RuntimeError(f"Processing-API HTTP {antwort.status_code}: "
                           f"{antwort.text[:300]}")
    pu = antwort.headers.get("x-processingunits-spent")
    if pu:
        try:
            lauf.pu_je_aoi[aoi["id"]] = lauf.pu_je_aoi.get(aoi["id"], 0.0) + float(pu)
        except ValueError:
            pass  # Header da, aber nicht numerisch — dann eben ohne PU-Buchung
    return antwort.content


def lies_baender(quelle, mit_n):
    """Band 1 = NDVI, Band 2 = n (gültige Beobachtungen). Bestandsdateien aus
    der Zeit vor M4 sind einbandig — dann ist n None, und die Aufrufer müssen
    ohne Datengüte auskommen, statt an einem Lesefehler zu scheitern."""
    ndvi = quelle.read(1)
    if not mit_n:
        return ndvi
    return ndvi, (quelle.read(2) if quelle.count >= 2 else None)


def lies_tiff_bytes(inhalt, mit_n=False):
    """CDSE liefert GeoTIFF-Bytes; wir wollen nur die Float-Arrays."""
    with MemoryFile(inhalt) as speicher:
        with speicher.open() as quelle:
            return lies_baender(quelle, mit_n)


def lies_tiff(pfad, mit_n=False):
    with rasterio.open(pfad) as quelle:
        return lies_baender(quelle, mit_n)


# ---------------------------------------------------------------------------
# Raster-Ausgabe (COG-ähnlich) & PNG
# ---------------------------------------------------------------------------

def schreibe_tiff(pfad, daten, bbox, klassenraster=False, n=None, *, epsg):
    """GeoTIFF COG-ähnlich: gekachelt, Deflate, Overviews 2/4/8. Georeferenz
    seit R1 (12.08.2026) auf die PROJIZIERTE bbox im metrischen Ziel-CRS —
    `bbox` und `epsg` kommen aus gitter_von()/aoi_gitter(). Klassenraster
    (UINT8) bekommen Nearest-Overviews, damit keine Mischklassen entstehen.

    Mit `n` entsteht ein zweibandiges Raster (Band 2 = Zahl der gültigen
    Beobachtungen bzw. bei Aggregaten der Jahre mit gültigem Wert). Ohne `n`
    bleibt es einbandig — Persistenz und Altbestand ändern ihr Format nicht.
    Band-Beschreibungen werden gesetzt, damit gdalinfo/geotiff.js die Rollen
    lesen können, statt sie aus der Bandnummer raten zu müssen."""
    pfad.parent.mkdir(parents=True, exist_ok=True)
    hoehe, breite = daten.shape
    west, sued, ost, nord = bbox
    profil = {
        "driver": "GTiff", "width": breite, "height": hoehe,
        "count": 1 if n is None else 2,
        "dtype": "uint8" if klassenraster else "float32",
        "crs": f"EPSG:{epsg}",
        "transform": from_bounds(west, sued, ost, nord, breite, hoehe),
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "compress": "deflate",
        "nodata": 0 if klassenraster else float("nan"),
    }
    with rasterio.open(pfad, "w", **profil) as ziel:
        ziel.write(daten.astype(profil["dtype"]), 1)
        if n is not None:
            ziel.write(n.astype(profil["dtype"]), 2)
            ziel.set_band_description(1, "ndvi_median")
            ziel.set_band_description(2, "n")
        ziel.build_overviews([2, 4, 8], Resampling.nearest if klassenraster
                             else Resampling.average)


def speichere_png(pfad, rgba, demo=False):
    """RGBA-Array als PNG; Demo-Bilder bekommen den DEMO-Stempel eingebrannt."""
    pfad.parent.mkdir(parents=True, exist_ok=True)
    bild = Image.fromarray(rgba, "RGBA")
    if demo:
        bild = demo_stempel(bild)
    bild.save(pfad)


def einfaerben(ndvi, farbskala):
    """NDVI-Array → RGBA nach der Farbskala aus aois.json (linear
    interpoliert, Domäne geklemmt, NoData transparent). Identische Logik
    wie im PNG-Evalscript — nur eben lokal für Baseline & Co."""
    stuetzen = farbskala["stuetzstellen"]
    dmin, dmax = farbskala["domaene"]
    xs = [s[0] for s in stuetzen]
    gueltig = np.isfinite(ndvi)
    werte = np.clip(np.where(gueltig, ndvi, dmin), dmin, dmax)
    rgba = np.zeros(ndvi.shape + (4,), np.uint8)
    for kanal in range(3):
        ys = [s[kanal + 1] for s in stuetzen]
        rgba[..., kanal] = np.rint(np.interp(werte, xs, ys)).astype(np.uint8)
    rgba[..., 3] = np.where(gueltig, 255, 0)
    return rgba


def persistenz_einfaerben(klassen):
    """Klassenraster → RGBA nach Vertragstabelle; 0 und 1 transparent."""
    rgba = np.zeros(klassen.shape + (4,), np.uint8)
    for klasse, (r, g, b) in PERSISTENZ_FARBEN.items():
        maske = klassen == klasse
        rgba[maske] = (r, g, b, 255)
    return rgba


# ---------------------------------------------------------------------------
# Rechenkerne (werden von echten UND Demo-Läufen genutzt)
# ---------------------------------------------------------------------------

def baseline_median(jahres_stack):
    """Pixelweiser Median über die Jahres-Saison-Composites."""
    with warnings.catch_warnings():
        # nanmedian warnt bei Pixeln, die in allen Jahren NoData sind — gewollt.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(jahres_stack, axis=0).astype("float32")


def jahre_mit_wert(stack):
    """Band 2 der Aggregate: wie viele Jahre des Stapels haben an diesem Pixel
    überhaupt einen gültigen Wert? Ein Medianwert aus einem einzigen Jahr ist
    kein Normalzustand — das Frontend blendet solche Pixel aus."""
    return np.isfinite(stack).sum(axis=0).astype("float32")


def guete_kennzahlen(n):
    """Datengüte eines Composites aus Band 2 (n = wolkenfreie Beobachtungen).

    `n_median` zählt nur Pixel mit mindestens einer Aufnahme: Läge eine
    Wolkendecke über der halben AOI, zöge sie den Median sonst auf 0 und
    verdeckte, wie belastbar der Rest ist — wie groß die Lücke ist, steht
    daneben in `anteil_ohne_daten`. `anteil_n1` ist der Anteil der Pixel, die
    nur eine einzige Beobachtung haben (im Frontend hohl gezeichnet).
    `beobachtungen` exportiert min/median/max über ALLE Pixel (Rezeptur
    § 8.3 — Beobachtungsdichte gehört in die JSON, nicht nur in den Cache;
    das Minimum ist ehrlicherweise oft 0: Wolkenlücken).
    Liefert None für einbandige Altbestände."""
    if n is None:
        return None
    gesamt = int(n.size)
    mit_daten = n >= 1
    return {
        "n_median": (round(float(np.median(n[mit_daten])), 1)
                     if mit_daten.any() else 0.0),
        "anteil_ohne_daten": round(float(int((~mit_daten).sum()) / gesamt), 4),
        "anteil_n1": round(float(int((n == 1).sum()) / gesamt), 4),
        "beobachtungen": {"min": int(n.min()),
                          "median": round(float(np.median(n)), 1),
                          "max": int(n.max())},
    }


def klassifiziere_persistenz(monats_stack, baseline, klassifikation):
    """Klassenraster nach Vertragstabelle. monats_stack: neuester Monat zuerst.

    0 = NoData/zu wenig Monate · 1 = stabil · 2 = wieder grün (stabil) ·
    3 = ergrünt (jung) · 4 = Vegetationsverlust."""
    d_gruen = klassifikation["anomalie_gruen"]
    d_verlust = klassifikation["anomalie_verlust"]     # negativ, s. aois.json
    min_monate = klassifikation["persistenz_min_monate"]

    gueltig = np.isfinite(monats_stack)
    anzahl_gueltig = gueltig.sum(axis=0)
    with np.errstate(invalid="ignore"):  # NaN-Vergleiche sollen False sein
        ueber = ((monats_stack > baseline + d_gruen) & gueltig).sum(axis=0)
        unter = ((monats_stack < baseline + d_verlust) & gueltig).sum(axis=0)
        # Aktuellster gültiger Monat je Pixel (argmax findet das erste True).
        erste_idx = np.argmax(gueltig, axis=0)
        neuester = np.take_along_axis(monats_stack, erste_idx[None], axis=0)[0]
        jung = neuester > baseline + d_gruen

    klassen = np.ones(baseline.shape, np.uint8)
    klassen[jung] = 3                       # jung …
    klassen[unter >= min_monate] = 4        # … es sei denn dauerhaft darunter
    klassen[ueber >= min_monate] = 2        # … oder dauerhaft darüber
    klassen[(anzahl_gueltig < 2) | ~np.isfinite(baseline)] = 0
    return klassen


def berechne_pca(jahres_stack):
    """Multitemporale PCA nach Vertrag: Pixel mit NoData in irgendeinem Jahr
    fallen raus, Bänder werden zentriert, PCA über die Kovarianz (SVD).
    Liefert (RGBA aus PC1–3, je 2.–98. Perzentil gestreckt, Varianzanteile)."""
    n_jahre = jahres_stack.shape[0]
    if n_jahre < 3:
        raise RuntimeError(f"PCA braucht mindestens 3 Jahres-Composites "
                           f"(vorhanden: {n_jahre}).")
    gueltig = np.isfinite(jahres_stack).all(axis=0)
    if not gueltig.any():
        raise RuntimeError("PCA: kein Pixel ist in allen Jahren gültig.")

    x = jahres_stack[:, gueltig].astype("float64").T   # (Pixel × Jahre)
    x -= x.mean(axis=0)
    kovarianz = x.T @ x / max(x.shape[0] - 1, 1)
    u, s, _ = np.linalg.svd(kovarianz)
    varianz_anteile = [round(float(a), 4) for a in s / s.sum()]
    hauptkomponenten = x @ u[:, :3]                    # (Pixel × 3)

    rgba = np.zeros(gueltig.shape + (4,), np.uint8)
    flach_maske = gueltig.ravel()
    for kanal in range(3):
        werte = hauptkomponenten[:, kanal]
        p2, p98 = np.percentile(werte, [2, 98])
        if p98 <= p2:
            p98 = p2 + 1e-9
        kanal_flach = np.zeros(flach_maske.size, np.uint8)
        kanal_flach[flach_maske] = np.clip(
            (werte - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        rgba[..., kanal] = kanal_flach.reshape(gueltig.shape)
    rgba[..., 3] = np.where(gueltig, 255, 0)
    return rgba, varianz_anteile


FENSTER_METHODIK = ("Tagesgenau gematchte Klimatologie: für jedes Baseline-"
                    "Jahr derselbe 14-Tage-Kalenderausschnitt wie der aktuelle "
                    "Composite (Analyse-Auflösung, SCL-maskierter Median), "
                    "darüber pixelweiser Median über die Jahre. Band 1 = "
                    "Median-NDVI, Band 2 = Zahl der Jahre mit gültigem Wert. "
                    "Damit entfällt der phänologische Versatz, den ein "
                    "Hochsommerfenster gegen den Saisonmedian Jun–Aug erzeugt "
                    "(baseline.tif bleibt als Saison-Normal daneben stehen).")


def fenster_baseline_bericht(von, bis, jahre, n_jahre, demo=False):
    """Sidecar zur Fensterbaseline. `n_jahre_min/median` laufen bewusst über
    ALLE Pixel — Randpixel ohne jede Aufnahme drücken das Minimum auf 0, und
    genau diese Ehrlichkeit braucht das Frontend für seine 3-Jahres-Schwelle."""
    bericht = {
        "fenster": {"von": von.isoformat(), "bis": bis.isoformat()},
        "jahre": list(jahre),
        "n_jahre_min": int(n_jahre.min()),
        "n_jahre_median": float(np.median(n_jahre)),
        "methodik": FENSTER_METHODIK,
        # Kartenpass-Pflicht (Rezeptur § 8.1): der exakte Skriptaufruf.
        "aufruf": ch.aufruf_protokoll(),
    }
    if demo:
        bericht["demo"] = True
        bericht["methodik"] += " Demo-Lauf: synthetische Daten."
    return bericht


PCA_METHODIK = ("Multitemporale PCA über die NDVI-Saison-Composites der "
                "Baseline-Jahre: NoData-Pixel verworfen, Bänder je Jahr "
                "zentriert, Kovarianz per SVD zerlegt. RGB = PC1–PC3, je "
                "2.–98. Perzentil auf 0–255 gestreckt. PC1 ≈ mittlere "
                "Grünheit, PC2/PC3 ≈ Veränderungsmuster.")


# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------

def aktualisiere_meta(lauf, aoi, **felder):
    """meta.json fortschreiben statt überschreiben: PU kumuliert über die
    Läufe (Vertrag: „kumuliert je Lauf"), bestehende Felder bleiben stehen."""
    meta_pfad = lauf.ausgabe / aoi["id"] / "meta.json"
    meta = lies_json(meta_pfad) or {}
    pu_delta = round(lauf.pu_je_aoi.get(aoi["id"], 0.0) - lauf.pu_gebucht.get(aoi["id"], 0.0), 2)
    lauf.pu_gebucht[aoi["id"]] = lauf.pu_je_aoi.get(aoi["id"], 0.0)
    meta.setdefault("aoi", aoi["id"])
    meta.setdefault("max_wolken", MAX_WOLKEN)
    # Georeferenz + Kartenpass (R1 / Rezeptur § 8): CRS des Analysegitters
    # und der exakte Aufruf des letzten Laufs, der diese meta angefasst hat.
    meta["crs"] = f"EPSG:{ch.aoi_epsg(aoi)}"
    # bbox des ANALYSE-Gitters (aufs Gitter gesnappt, Meter): der Wochen-Cron
    # (ndvi-refresh.php) liest sie von HIER, statt in PHP Geodäsie zu treiben —
    # nur so trifft seine Woche exakt das Gitter von Baseline und Anomalie.
    meta["bbox_utm"] = [round(float(k), 3) for k in
                        gitter_von(aoi, analyse_aufloesung(aoi))[0]]
    meta["aufruf"] = ch.aufruf_protokoll()
    # demo/quelle nur bei explizitem Kwarg umschalten: Ein Teilbefehl (etwa
    # 'woche') erneuert nur ein Artefakt und darf den Demo-Zustand der
    # übrigen nicht fälschlich für beendet erklären — das tut erst
    # cmd_alles nach einem vollständigen, fehlerfreien Echtlauf.
    meta["demo"] = felder.pop("demo", meta.get("demo", False))
    meta["quelle"] = felder.pop("quelle", meta.get("quelle", QUELLE_ECHT))
    meta["pu_verbraucht"] = round(float(meta.get("pu_verbraucht", 0.0)) + pu_delta, 2)
    meta["aktualisiert"] = jetzt_utc()
    meta.update(felder)
    schreibe_json(meta_pfad, meta)


# ---------------------------------------------------------------------------
# STAC (1.0.0, minimal aber valide, alle Links relativ)
# ---------------------------------------------------------------------------

def bbox_polygon(bbox):
    west, sued, ost, nord = bbox
    return {"type": "Polygon",
            "coordinates": [[[west, sued], [ost, sued], [ost, nord],
                             [west, nord], [west, sued]]]}


def artefakt_zeitraum(artefakt, meta, saison):
    """(start, ende) als ISO-Zeitstempel je Artefakt; Quelle ist meta.json
    bzw. die Saisonfenster der Baseline-Jahre."""
    jahre = saison["baseline_jahre"]
    if (artefakt in ("aktuell", "aktuell-analyse", "falsecolor-aktuell")
            and meta.get("zeitfenster")):
        zf = meta["zeitfenster"]
        return f"{zf['von']}T00:00:00Z", f"{zf['bis']}T23:59:59Z"
    if artefakt in ("composite-2021", "falsecolor-2021"):
        von, bis = saison_fenster(2021, saison)
        return f"{von.isoformat()}T00:00:00Z", f"{bis.isoformat()}T23:59:59Z"
    if artefakt in ("baseline", "pca", "baseline-fenster") and jahre:
        # Die Fensterbaseline datiert auf die gematchten Kalenderfenster, nicht
        # auf die Saison — sonst behauptete das Item einen Zeitraum, den die
        # Daten gar nicht abdecken. Ohne bekanntes Fenster (meta noch ohne
        # zeitfenster) fällt sie auf die Saisondatierung zurück.
        fenster_ende = None
        if artefakt == "baseline-fenster":
            try:
                fenster_ende = dt.date.fromisoformat(meta["zeitfenster"]["bis"])
            except (KeyError, TypeError, ValueError):
                fenster_ende = None
        if fenster_ende:
            von = kalender_fenster(jahre[0], fenster_ende)[0]
            bis = kalender_fenster(jahre[-1], fenster_ende)[1]
        else:
            von = saison_fenster(jahre[0], saison)[0]
            bis = saison_fenster(jahre[-1], saison)[1]
        return f"{von.isoformat()}T00:00:00Z", f"{bis.isoformat()}T23:59:59Z"
    if artefakt.startswith("woche-"):
        jahr, woche = artefakt[len("woche-"):].split("-W")
        von = dt.date.fromisocalendar(int(jahr), int(woche), 1)
        bis = dt.date.fromisocalendar(int(jahr), int(woche), 7)
        return f"{von.isoformat()}T00:00:00Z", f"{bis.isoformat()}T23:59:59Z"
    # persistenz & alles Übrige: nur der Stand der letzten Aktualisierung
    return None, meta.get("aktualisiert") or jetzt_utc()


def stac_item(aoi, artefakt, assets, meta, saison):
    """Ein STAC-Item je Artefakt; Asset-Hrefs relativ zu <aoi>/items/."""
    start, ende = artefakt_zeitraum(artefakt, meta, saison)
    eigenschaften = {"datetime": ende}
    if start:
        eigenschaften["start_datetime"] = start
        eigenschaften["end_datetime"] = ende
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": f"{aoi['id']}-{artefakt}",
        "collection": f"ndvi-{aoi['id']}",
        "geometry": bbox_polygon(aoi["bbox"]),
        "bbox": list(aoi["bbox"]),
        "properties": eigenschaften,
        "links": [
            {"rel": "root", "href": "../../catalog.json", "type": TYP_JSON},
            {"rel": "parent", "href": "../collection.json", "type": TYP_JSON},
            {"rel": "collection", "href": "../collection.json", "type": TYP_JSON},
        ],
        "assets": {
            name: {"href": f"../{unterpfad}", "type": typ, "roles": rollen}
            for name, unterpfad, typ, rollen in assets
        },
    }


def schreibe_stac_item(lauf, aoi, artefakt, assets):
    """Item nur schreiben, wenn wenigstens ein Asset wirklich existiert."""
    aoi_dir = lauf.ausgabe / aoi["id"]
    vorhandene = [a for a in assets if (aoi_dir / a[1]).exists()]
    if not vorhandene:
        return None
    meta = lies_json(aoi_dir / "meta.json") or {}
    item = stac_item(aoi, artefakt, vorhandene, meta, lauf.konfig["saison"])
    schreibe_json(aoi_dir / "items" / f"{artefakt}.json", item)
    return item


def baue_stac(lauf):
    """catalog.json + je AOI collection.json + items/*.json aus dem, was an
    Artefakten tatsächlich auf der Platte liegt."""
    saison = lauf.konfig["saison"]
    for aoi in lauf.aois:
        aoi_dir = lauf.ausgabe / aoi["id"]
        if not aoi_dir.exists():
            lauf.merke(aoi["id"], "stac", "ok (keine Artefakte, übersprungen)")
            continue
        try:
            item_namen = []
            for artefakt, assets in ARTEFAKT_ASSETS.items():
                if schreibe_stac_item(lauf, aoi, artefakt, assets):
                    item_namen.append(artefakt)
            # Wochen-TIFFs bekommen je ein eigenes Item (ein Item je Artefakt).
            for wochen_tif in sorted((aoi_dir / "wochen").glob("*.tif")):
                artefakt = f"woche-{wochen_tif.stem}"
                assets = [("tif", f"wochen/{wochen_tif.name}", TYP_GEOTIFF, ["data"])]
                if schreibe_stac_item(lauf, aoi, artefakt, assets):
                    item_namen.append(artefakt)

            meta = lies_json(aoi_dir / "meta.json") or {}
            jahre = saison["baseline_jahre"]
            zeit_von = (f"{saison_fenster(jahre[0], saison)[0].isoformat()}"
                        "T00:00:00Z") if jahre else None
            sammlung = {
                "type": "Collection",
                "stac_version": "1.0.0",
                "id": f"ndvi-{aoi['id']}",
                "title": f"NDVI-Monitor · {aoi['name']}",
                "description": aoi.get("beschreibung", ""),
                "license": "CC-BY-4.0",
                "providers": [
                    {"name": "Copernicus / ESA",
                     "roles": ["producer", "licensor"],
                     "url": "https://dataspace.copernicus.eu/"},
                    {"name": "GEOPHORA", "roles": ["processor", "host"],
                     "url": "https://geophora.de/"},
                ],
                "extent": {
                    "spatial": {"bbox": [list(aoi["bbox"])]},
                    "temporal": {"interval": [[zeit_von,
                                               meta.get("aktualisiert")
                                               or jetzt_utc()]]},
                },
                "links": (
                    [{"rel": "root", "href": "../catalog.json", "type": TYP_JSON},
                     {"rel": "parent", "href": "../catalog.json", "type": TYP_JSON}]
                    + [{"rel": "item", "href": f"./items/{name}.json",
                        "type": "application/geo+json"} for name in item_namen]
                ),
            }
            schreibe_json(aoi_dir / "collection.json", sammlung)
            lauf.merke(aoi["id"], "stac", f"ok ({len(item_namen)} Items)")
        except Exception as fehler:
            lauf.merke(aoi["id"], "stac", f"FEHLER: {fehler}")

    # Der Katalog listet alle AOIs, deren Collection existiert — auch die,
    # die in diesem Lauf per --aoi gar nicht dran waren.
    kinder = []
    for aoi in lauf.konfig["aois"]:
        if (lauf.ausgabe / aoi["id"] / "collection.json").exists():
            kinder.append({"rel": "child",
                           "href": f"./{aoi['id']}/collection.json",
                           "type": TYP_JSON, "title": aoi["name"]})
    katalog = {
        "type": "Catalog",
        "stac_version": "1.0.0",
        "id": "geophora-ndvi",
        "title": "GEOPHORA NDVI-Monitor",
        "description": ("Wöchentliche NDVI-Composites, saisonale Baseline, "
                        "Persistenz- und PCA-Auswertungen aus Sentinel-2 L2A "
                        "(Copernicus Data Space Ecosystem)."),
        "links": [{"rel": "root", "href": "./catalog.json", "type": TYP_JSON}]
                 + kinder,
    }
    schreibe_json(lauf.ausgabe / "catalog.json", katalog)


# ---------------------------------------------------------------------------
# CDSE-Abrufe (gemeinsame Bausteine der Befehle)
# ---------------------------------------------------------------------------

def hole_ndvi_array(lauf, aoi, von, bis, aufloesung_m=None, mit_n=False):
    """Liefert das NDVI-Array; mit mit_n=True zusätzlich das n-Band."""
    inhalt = process(lauf, aoi, von, bis, evalscript_ndvi_tiff(), "image/tiff",
                     aufloesung_m=aufloesung_m)
    return lies_tiff_bytes(inhalt, mit_n=mit_n)


def jahres_cache_pfad(aoi_id, jahr):
    return CACHE_DIR / aoi_id / f"jahr-{jahr}.tif"


def fenster_cache_pfad(aoi_id, jahr, ende):
    """Fenster-Composites sind über (AOI, Jahr, Endtag) geschlüsselt: verschiebt
    sich das aktuelle Fenster um einen Tag, entsteht ein neuer Cache-Eintrag
    statt eines stillen Fehltreffers."""
    return CACHE_DIR / aoi_id / f"fenster-{jahr}-{ende:%m%d}.tif"


def cache_gueltig(pfad, aoi, aufloesung_m=None):
    """Cache-Treffer nur akzeptieren, wenn CRS, Pixelmaße und bbox zum
    aktuellen metrischen Gitter passen — der Cache ist nur über (AOI,
    Zeitraum) geschlüsselt, nach einer bbox-/Auflösungs-/CRS-Änderung würde
    sonst ein altes Raster stillschweigend mit falscher Georeferenz
    weiterverwendet. Altbestände im Grad-Gitter (EPSG:4326, vor R1) fallen
    hier automatisch durch und werden neu geholt."""
    epsg, bbox_utm, breite, hoehe = aoi_gitter(aoi, aufloesung_m)
    try:
        with rasterio.open(pfad) as quelle:
            if (quelle.width, quelle.height) != (breite, hoehe):
                return False
            if quelle.crs is None or quelle.crs.to_epsg() != epsg:
                return False
            grenzen = (quelle.bounds.left, quelle.bounds.bottom,
                       quelle.bounds.right, quelle.bounds.top)
    except rasterio.errors.RasterioIOError:
        return False  # kaputte/halbe Datei zählt wie ein Fehltreffer
    # Toleranz (1 mm) fängt nur Float-Rundung der Georeferenz ab, keine
    # echten Verschiebungen.
    return all(abs(ist - soll) <= 1e-3
               for ist, soll in zip(grenzen, bbox_utm))


def sichere_jahres_composite(lauf, aoi, jahr):
    """Saison-Composite eines Jahres in den lokalen Cache holen (idempotent);
    veraltete Treffer (bbox/Auflösung geändert) werden neu geholt."""
    pfad = jahres_cache_pfad(aoi["id"], jahr)
    if pfad.exists() and cache_gueltig(pfad, aoi, analyse_aufloesung(aoi)):
        return pfad
    von, bis = saison_fenster(jahr, lauf.konfig["saison"])
    daten, n = hole_ndvi_array(lauf, aoi, von, bis,
                               aufloesung_m=analyse_aufloesung(aoi), mit_n=True)
    bbox_utm, epsg = gitter_von(aoi, analyse_aufloesung(aoi))
    schreibe_tiff(pfad, daten, bbox_utm, n=n, epsg=epsg)
    return pfad


def baseline_fenster_tage(konfig):
    """Fensterlänge der Klimatologie — bewusst LÄNGER als die 14 Tage des
    aktuellen Composites.

    Gemessen am Harz (05.08.2026): mit 14 Tagen trugen einzelne Baseline-Jahre
    nur 7,5 % bzw. 30 % gültige Pixel bei, meist aus einer einzigen Aufnahme.
    Einzelszenen sind dunstanfällig und liegen dann systematisch zu niedrig —
    die Baseline sackt ab und die Anomalie färbt großflächig grün. Ein breiteres
    Fenster um denselben Kalendertag bleibt phänologisch vergleichbar
    (Hochsommer-Plateau), liefert aber mehrere wolkenfreie Überflüge je Jahr.
    """
    return int(konfig["saison"].get("baseline_fenster_tage", 42))


def sichere_fenster_composite(lauf, aoi, jahr, ende_aktuell, tage=None):
    """Denselben Kalenderausschnitt wie der aktuelle Composite in einem
    früheren Jahr holen (Analyse-Auflösung, idempotent gecacht)."""
    von, bis = kalender_fenster(jahr, ende_aktuell,
                                tage=tage or baseline_fenster_tage(lauf.konfig))
    pfad = fenster_cache_pfad(aoi["id"], jahr, bis)
    if pfad.exists() and cache_gueltig(pfad, aoi, analyse_aufloesung(aoi)):
        return pfad
    daten, n = hole_ndvi_array(lauf, aoi, von, bis,
                               aufloesung_m=analyse_aufloesung(aoi), mit_n=True)
    bbox_utm, epsg = gitter_von(aoi, analyse_aufloesung(aoi))
    schreibe_tiff(pfad, daten, bbox_utm, n=n, epsg=epsg)
    return pfad


def lade_jahres_stack(lauf, aoi):
    """Alle Baseline-Jahre aus dem Cache; fehlende Jahre werden nachgeholt."""
    jahre = lauf.konfig["saison"]["baseline_jahre"]
    return jahre, np.stack([lies_tiff(sichere_jahres_composite(lauf, aoi, j))
                            for j in jahre])


# ---------------------------------------------------------------------------
# Befehle
# ---------------------------------------------------------------------------

def cmd_woche(lauf):
    """Aktueller Wochen-Composite — dasselbe, was der PHP-Cron tut, nur lokal
    und mit COG-Ausgabe.

    Zwei Gitter, bewusst getrennt (M2): `aktuell.tif/.png` bleibt das scharfe
    Anzeigeprodukt in `aufloesung_m`; alle Analysen laufen auf dem Gitter
    `aufloesung_analyse_m`. Der Analyse-Composite wird identisch als
    `aktuell-analyse.tif` und `wochen/<ISO>.tif` abgelegt — dieselben Bytes,
    damit Popup-Zeitreihe und Anomaliefarbe aus demselben Pixel stammen und
    die Reihe an ihrem Endpunkt keinen Sprung bekommt."""
    heute = dt.date.today()
    von, bis = heute - dt.timedelta(days=14), heute
    iso = iso_wochenlabel(heute)
    for aoi in lauf.aois:
        try:
            aoi_dir = lauf.ausgabe / aoi["id"]
            daten, n = hole_ndvi_array(lauf, aoi, von, bis, mit_n=True)
            analyse, analyse_n = hole_ndvi_array(
                lauf, aoi, von, bis, aufloesung_m=analyse_aufloesung(aoi),
                mit_n=True)
            png_bytes = process(lauf, aoi, von, bis,
                                evalscript_ndvi_png(lauf.konfig["farbskala"]),
                                "image/png")
            # Zwei Gitter, zwei Georeferenzen: Anzeige (10 m) und Analyse
            # (20 m) snappen getrennt aufs metrische Raster.
            bbox_10, epsg = gitter_von(aoi)
            bbox_20, _ = gitter_von(aoi, analyse_aufloesung(aoi))
            schreibe_tiff(aoi_dir / "aktuell.tif", daten, bbox_10, n=n,
                          epsg=epsg)
            schreibe_tiff(aoi_dir / "aktuell-analyse.tif", analyse,
                          bbox_20, n=analyse_n, epsg=epsg)
            schreibe_tiff(aoi_dir / "wochen" / f"{iso}.tif", analyse,
                          bbox_20, n=analyse_n, epsg=epsg)
            (aoi_dir / "aktuell.png").write_bytes(png_bytes)

            index = lies_json(aoi_dir / "wochen" / "index.json") or {"wochen": []}
            index["wochen"] = sorted(set(index["wochen"]) | {iso})
            schreibe_json(aoi_dir / "wochen" / "index.json", index)

            hinweis = "" if (aoi_dir / "baseline.tif").exists() \
                else "Baseline fehlt noch"
            # Güte bezieht sich auf den Analyse-Composite: Er trägt die
            # Anomalie und die Zeitreihe, also zählt seine Wolkenfreiheit.
            extra = {}
            guete = guete_kennzahlen(analyse_n)
            if guete:
                extra["guete"] = guete
            aktualisiere_meta(lauf, aoi, iso_woche=iso,
                              zeitfenster={"von": von.isoformat(),
                                           "bis": bis.isoformat()},
                              hinweis=hinweis, **extra)
            for artefakt, assets in (
                    ("aktuell", ARTEFAKT_ASSETS["aktuell"]),
                    ("aktuell-analyse", ARTEFAKT_ASSETS["aktuell-analyse"]),
                    (f"woche-{iso}",
                     [("tif", f"wochen/{iso}.tif", TYP_GEOTIFF, ["data"])])):
                schreibe_stac_item(lauf, aoi, artefakt, assets)
            lauf.merke(aoi["id"], f"woche {iso}", "ok")
        except Exception as fehler:
            lauf.merke(aoi["id"], f"woche {iso}", f"FEHLER: {fehler}")


def wochen_index_neu(aoi_dir):
    """index.json aus dem tatsächlichen Dateibestand neu schreiben — die Liste
    ist Anzeigequelle der Klick-Zeitreihe und darf nie auf gelöschte oder
    fremde Dateien zeigen."""
    wo = aoi_dir / "wochen"
    wochen = sorted(p.stem for p in wo.glob("*.tif")) if wo.is_dir() else []
    if wochen or wo.is_dir():
        schreibe_json(wo / "index.json", {"wochen": wochen})
    return wochen


def cmd_aufraeumen(lauf):
    """Demo-Reste aus dem Wochen-Archiv entfernen.

    Grund (2026-08-05 aufgefallen): Der Demo-Lauf legt synthetische Wochen im
    SELBEN Archiv ab wie echte Läufe. Nach dem Umstieg auf echte Daten blieben
    sie liegen — die Klick-Zeitreihe mischte damit Rauschen mit Messwerten und
    widersprach der (korrekt gerechneten) Anomalie.

    Erkennungsmerkmal: echte Wochen liegen auf einem der beiden erlaubten
    Gitter der AOI (Anzeige- ODER Analyse-Auflösung) — seit M2 schreibt der
    Live-Lauf das Analyse-Gitter, der Rückblick ebenfalls; nur die Demo rechnet
    gröber. Zusätzlich werden verwaiste STAC-Items entfernt.
    """
    for aoi in lauf.aois:
        try:
            aoi_dir = lauf.ausgabe / aoi["id"]
            referenz = aoi_dir / "aktuell.tif"
            if not referenz.exists():
                lauf.merke(aoi["id"], "aufraeumen", "FEHLER: aktuell.tif fehlt")
                continue
            # ZWEI gültige Gitter: Anzeige-Auflösung (Wochen aus dem Live-Lauf)
            # und Analyse-Auflösung (Rückblick, Fensterbaseline). Nur was zu
            # keinem von beiden passt, stammt aus dem Demo-Lauf.
            erlaubt = {raster_groesse(aoi["bbox"], aoi["aufloesung_m"]),
                       raster_groesse(aoi["bbox"], analyse_aufloesung(aoi))}
            entfernt = []
            for tif in sorted((aoi_dir / "wochen").glob("*.tif")):
                with rasterio.open(tif) as q:
                    passt = (q.width, q.height) in erlaubt
                if not passt:
                    tif.unlink()
                    (aoi_dir / "items" / f"woche-{tif.stem}.json").unlink(missing_ok=True)
                    entfernt.append(tif.stem)
            rest = wochen_index_neu(aoi_dir)
            lauf.merke(aoi["id"], "aufraeumen",
                       f"ok ({len(entfernt)} Demo-Wochen entfernt, {len(rest)} echte verbleiben)")
        except Exception as fehler:
            lauf.merke(aoi["id"], "aufraeumen", f"FEHLER: {fehler}")


def cmd_rueckblick(lauf):
    """Vergangene Wochen-Composites echt nachrechnen (Archiv füllen).

    Jede Woche bekommt dasselbe 14-Tage-Fenster wie der Live-Lauf, nur
    rückdatiert. Standard ist die Analyse-Auflösung (20 m statt 10 m): Der
    Rückblick dient der Zeitreihe, nicht der Detailansicht — das kostet rund
    ein Viertel der Processing Units. Vorhandene echte Wochen werden nicht
    erneut geholt.
    """
    anzahl = lauf.args.wochen if getattr(lauf.args, "wochen", None) else 8
    aufl = getattr(lauf.args, "aufloesung", None)
    heute = dt.date.today()
    for aoi in lauf.aois:
        aoi_dir = lauf.ausgabe / aoi["id"]
        meter = aufl or analyse_aufloesung(aoi)
        soll = raster_groesse(aoi["bbox"], meter)
        for k in range(1, anzahl + 1):
            bis = heute - dt.timedelta(days=7 * k)
            von = bis - dt.timedelta(days=14)
            iso = iso_wochenlabel(bis)
            ziel = aoi_dir / "wochen" / f"{iso}.tif"
            try:
                if ziel.exists():
                    with rasterio.open(ziel) as q:
                        # Gültig ist nur, was dem AKTUELLEN Vertrag entspricht:
                        # richtiges Gitter UND zweibandig (Band 2 = Zahl der
                        # wolkenfreien Beobachtungen). Einbandige Altbestände
                        # werden neu geholt, sonst bliebe die Zeitreihe ohne
                        # Güteangabe — genau die Information, die fehlte.
                        aktuell_gueltig = (q.width, q.height) == soll and q.count >= 2
                    if aktuell_gueltig and not lauf.force:
                        lauf.merke(aoi["id"], f"rueckblick {iso}", "ok (vorhanden)")
                        continue
                daten, n = hole_ndvi_array(lauf, aoi, von, bis,
                                           aufloesung_m=meter, mit_n=True)
                bbox_utm, epsg = gitter_von(aoi, meter)
                schreibe_tiff(ziel, daten, bbox_utm, n=n, epsg=epsg)
                schreibe_stac_item(lauf, aoi, f"woche-{iso}",
                                   [("tif", f"wochen/{iso}.tif", TYP_GEOTIFF, ["data"])])
                lauf.merke(aoi["id"], f"rueckblick {iso}", "ok")
            except Exception as fehler:
                lauf.merke(aoi["id"], f"rueckblick {iso}", f"FEHLER: {fehler}")
        wochen_index_neu(aoi_dir)


def cmd_historie(lauf):
    """Jahres-Saison-Composites in den lokalen Cache; 2021 zusätzlich als
    Referenz-Artefakte (composite-2021.*, falsecolor-2021.png) ins Layout."""
    jahre = [lauf.jahr] if lauf.jahr else lauf.konfig["saison"]["baseline_jahre"]
    for aoi in lauf.aois:
        fehlgeschlagen = False
        for jahr in jahre:
            try:
                sichere_jahres_composite(lauf, aoi, jahr)
                lauf.merke(aoi["id"], f"historie {jahr}", "ok")
            except Exception as fehler:
                fehlgeschlagen = True
                lauf.merke(aoi["id"], f"historie {jahr}", f"FEHLER: {fehler}")
        if 2021 not in jahre or fehlgeschlagen:
            continue
        try:
            aoi_dir = lauf.ausgabe / aoi["id"]
            daten, n = lies_tiff(jahres_cache_pfad(aoi["id"], 2021), mit_n=True)
            bbox_utm, epsg = gitter_von(aoi, analyse_aufloesung(aoi))
            schreibe_tiff(aoi_dir / "composite-2021.tif", daten, bbox_utm,
                          n=n, epsg=epsg)
            # PNG lokal einfärben statt zweitem Request — identische Skala,
            # spart Processing Units.
            speichere_png(aoi_dir / "composite-2021.png",
                          einfaerben(daten, lauf.konfig["farbskala"]))
            von, bis = saison_fenster(2021, lauf.konfig["saison"])
            fc = process(lauf, aoi, von, bis, evalscript_falsecolor(),
                         "image/png", aufloesung_m=analyse_aufloesung(aoi))
            (aoi_dir / "falsecolor-2021.png").write_bytes(fc)
            aktualisiere_meta(lauf, aoi)
            lauf.merke(aoi["id"], "composite-2021", "ok")
        except Exception as fehler:
            lauf.merke(aoi["id"], "composite-2021", f"FEHLER: {fehler}")


def cmd_baseline(lauf):
    """Saisonale Baseline = pixelweiser Median der Jahres-Composites (Jun–Aug).
    Fehlende Jahre im Cache werden automatisch nachgeholt. Band 2 hält die Zahl
    der Jahre, die an diesem Pixel überhaupt einen Wert beisteuern konnten.

    Bleibt als eigene Ebene „Saison-Normal" erhalten; die Anomalie rechnet seit
    M3 gegen baseline-fenster.tif, die Persistenz weiterhin gegen diese hier
    (dort werden Monate verglichen, da ist der Saisonmedian der richtige Bezug).
    """
    for aoi in lauf.aois:
        try:
            _, stack = lade_jahres_stack(lauf, aoi)
            baseline = baseline_median(stack)
            aoi_dir = lauf.ausgabe / aoi["id"]
            bbox_utm, epsg = gitter_von(aoi, analyse_aufloesung(aoi))
            schreibe_tiff(aoi_dir / "baseline.tif", baseline, bbox_utm,
                          n=jahre_mit_wert(stack), epsg=epsg)
            speichere_png(aoi_dir / "baseline.png",
                          einfaerben(baseline, lauf.konfig["farbskala"]))
            aktualisiere_meta(lauf, aoi)
            lauf.merke(aoi["id"], "baseline", "ok")
        except Exception as fehler:
            lauf.merke(aoi["id"], "baseline", f"FEHLER: {fehler}")


def cmd_fensterbaseline(lauf):
    """Tagesgenau gematchte Klimatologie (M3) — Bezug für die Anomalie.

    Ein 14-Tage-Hochsommerfenster gegen den Jun–Aug-Saisonmedian zu stellen,
    misst den Kalender, nicht die Vegetation (gemessener Versatz +0,08 bei
    einer Schwelle von 0,1). Deshalb: für jedes Baseline-Jahr denselben
    Kalenderausschnitt rechnen wie für den aktuellen Composite, darüber den
    pixelweisen Median. Ergebnis liegt auf dem Analyse-Gitter, also exakt
    deckungsgleich mit aktuell-analyse.tif und den Wochen.
    """
    heute = dt.date.today()
    von_aktuell, bis_aktuell = heute - dt.timedelta(days=14), heute
    jahre = lauf.konfig["saison"]["baseline_jahre"]
    for aoi in lauf.aois:
        try:
            aoi_dir = lauf.ausgabe / aoi["id"]
            # Je Jahr Wert UND Beobachtungszahl lesen: Jahre, die an einem Pixel
            # nur eine einzige Aufnahme beisteuern, werden verworfen. Eine
            # Einzelszene mit Dunstschleier zöge die Klimatologie sonst nach
            # unten und erzeugte eine Grün-Anomalie, die es nicht gibt.
            werte, zaehler = [], []
            for jahr in jahre:
                a, n_jahr = lies_tiff(
                    sichere_fenster_composite(lauf, aoi, jahr, bis_aktuell),
                    mit_n=True)
                if n_jahr is not None:
                    a = np.where(n_jahr >= 2, a, np.nan)
                werte.append(a)
                zaehler.append(n_jahr)
            stack = np.stack(werte)
            fenster_baseline = baseline_median(stack)
            n_jahre = jahre_mit_wert(stack)
            bbox_utm, epsg = gitter_von(aoi, analyse_aufloesung(aoi))
            schreibe_tiff(aoi_dir / "baseline-fenster.tif", fenster_baseline,
                          bbox_utm, n=n_jahre, epsg=epsg)
            speichere_png(aoi_dir / "baseline-fenster.png",
                          einfaerben(fenster_baseline,
                                     lauf.konfig["farbskala"]))
            schreibe_json(aoi_dir / "fenster-baseline.json",
                          fenster_baseline_bericht(von_aktuell, bis_aktuell,
                                                   jahre, n_jahre))
            aktualisiere_meta(lauf, aoi)
            schreibe_stac_item(lauf, aoi, "baseline-fenster",
                               ARTEFAKT_ASSETS["baseline-fenster"])
            lauf.merke(aoi["id"], "fensterbaseline", "ok")
        except Exception as fehler:
            lauf.merke(aoi["id"], "fensterbaseline", f"FEHLER: {fehler}")


def cmd_persistenz(lauf):
    """Monats-Composites der letzten Saison-Monate gegen die Baseline →
    Klassenraster. Der laufende Monat wird immer frisch geholt und nie
    gecacht, sonst bliebe ein unfertiger Teilstand nach dem Monatswechsel
    dauerhaft als vollständiger Monat liegen."""
    heute = dt.date.today()
    klassifikation = lauf.konfig["klassifikation"]
    monate = saison_monate_rueckwaerts(
        heute, klassifikation["persistenz_fenster_monate"],
        lauf.konfig["saison"])
    for aoi in lauf.aois:
        try:
            aoi_dir = lauf.ausgabe / aoi["id"]
            baseline_pfad = aoi_dir / "baseline.tif"
            if not baseline_pfad.exists():
                raise RuntimeError("baseline.tif fehlt — erst "
                                   "'baseline' laufen lassen.")
            baseline = lies_tiff(baseline_pfad)
            bbox_utm, epsg = gitter_von(aoi, analyse_aufloesung(aoi))

            stapel = []
            for jahr, monat in monate:               # neuester zuerst
                pfad = CACHE_DIR / aoi["id"] / f"monat-{jahr}-{monat:02d}.tif"
                if (jahr, monat) == (heute.year, heute.month):
                    # Laufender Monat: nur im Speicher verwenden, nie cachen
                    # — ein Teil-Composite gälte nach dem Monatswechsel
                    # sonst für immer als voller Monat. Altlasten früherer
                    # Läufe gleich mit entsorgen.
                    pfad.unlink(missing_ok=True)
                    von, bis = monats_fenster(jahr, monat, heute)
                    daten, _ = hole_ndvi_array(
                        lauf, aoi, von, bis,
                        aufloesung_m=analyse_aufloesung(aoi), mit_n=True)
                    stapel.append(daten)
                elif pfad.exists() and cache_gueltig(pfad, aoi,
                                                     analyse_aufloesung(aoi)):
                    stapel.append(lies_tiff(pfad))
                else:
                    von, bis = monats_fenster(jahr, monat, heute)
                    daten, n = hole_ndvi_array(
                        lauf, aoi, von, bis,
                        aufloesung_m=analyse_aufloesung(aoi), mit_n=True)
                    schreibe_tiff(pfad, daten, bbox_utm, n=n, epsg=epsg)
                    stapel.append(daten)

            klassen = klassifiziere_persistenz(np.stack(stapel), baseline,
                                               klassifikation)
            schreibe_tiff(aoi_dir / "persistenz.tif", klassen, bbox_utm,
                          klassenraster=True, epsg=epsg)
            speichere_png(aoi_dir / "persistenz.png",
                          persistenz_einfaerben(klassen))
            aktualisiere_meta(lauf, aoi)
            lauf.merke(aoi["id"], "persistenz", "ok")
        except Exception as fehler:
            lauf.merke(aoi["id"], "persistenz", f"FEHLER: {fehler}")


def cmd_pca(lauf):
    """PCA über die Jahres-Composites; fehlender Cache wird nachgeholt."""
    for aoi in lauf.aois:
        try:
            jahre, stack = lade_jahres_stack(lauf, aoi)
            rgba, varianz_anteile = berechne_pca(stack)
            aoi_dir = lauf.ausgabe / aoi["id"]
            speichere_png(aoi_dir / "pca.png", rgba)
            schreibe_json(aoi_dir / "pca.json",
                          {"varianz_anteile": varianz_anteile,
                           "methodik": PCA_METHODIK, "jahre": jahre,
                           "crs": f"EPSG:{ch.aoi_epsg(aoi)}",
                           "aufruf": ch.aufruf_protokoll()})
            lauf.merke(aoi["id"], "pca", "ok")
        except Exception as fehler:
            lauf.merke(aoi["id"], "pca", f"FEHLER: {fehler}")


def cmd_falsecolor(lauf):
    """False-Color-Composite der letzten 14 Tage."""
    heute = dt.date.today()
    von, bis = heute - dt.timedelta(days=14), heute
    for aoi in lauf.aois:
        try:
            inhalt = process(lauf, aoi, von, bis, evalscript_falsecolor(),
                             "image/png")
            aoi_dir = lauf.ausgabe / aoi["id"]
            aoi_dir.mkdir(parents=True, exist_ok=True)
            (aoi_dir / "falsecolor-aktuell.png").write_bytes(inhalt)
            aktualisiere_meta(lauf, aoi)
            lauf.merke(aoi["id"], "falsecolor-aktuell", "ok")
        except Exception as fehler:
            lauf.merke(aoi["id"], "falsecolor-aktuell", f"FEHLER: {fehler}")


def cmd_stac(lauf):
    baue_stac(lauf)


def cmd_alles(lauf):
    """Sinnvolle Reihenfolge: erst der Cache (historie), dann alles, was
    darauf aufbaut, zum Schluss der Katalog über die fertigen Artefakte."""
    cmd_historie(lauf)
    cmd_baseline(lauf)
    cmd_fensterbaseline(lauf)
    cmd_persistenz(lauf)
    cmd_pca(lauf)
    cmd_falsecolor(lauf)
    cmd_woche(lauf)
    # Erst der komplette Echtlauf beendet den Demo-Zustand — Teilbefehle
    # lassen meta['demo'] bewusst stehen (s. aktualisiere_meta). Umgeflaggt
    # wird nur, wo alle Kernartefakte fehlerfrei durchliefen, sonst würden
    # verbliebene Demo-Raster als echte Copernicus-Daten präsentiert.
    for aoi in lauf.aois:
        fehlerfrei = all(status.startswith("ok")
                         for aoi_id, _, status, _ in lauf.protokoll
                         if aoi_id == aoi["id"])
        if fehlerfrei:
            aktualisiere_meta(lauf, aoi, demo=False, quelle=QUELLE_ECHT)
    baue_stac(lauf)


# ---------------------------------------------------------------------------
# Demo — synthetische Artefakte ohne CDSE-Zugang
# ---------------------------------------------------------------------------

def lade_schrift(groesse):
    """TrueType, wenn auffindbar; sonst PIL-Standard (Pillow ≥ 10.1 skaliert)."""
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf",
                 "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, groesse)
        except OSError:
            continue
    try:
        return ImageFont.load_default(groesse)
    except TypeError:
        return ImageFont.load_default()


def demo_stempel(bild):
    """Brennt ein gekacheltes, halbtransparentes „DEMO" ein — unübersehbar,
    aber die Daten darunter bleiben erkennbar."""
    bild = bild.convert("RGBA")
    breite, hoehe = bild.size
    schrift = lade_schrift(max(28, min(breite, hoehe) // 7))
    messer = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    box = messer.textbbox((0, 0), "DEMO", font=schrift, stroke_width=2)
    kachel = Image.new("RGBA", (box[2] - box[0] + 24, box[3] - box[1] + 24),
                       (0, 0, 0, 0))
    ImageDraw.Draw(kachel).text(
        (12 - box[0], 12 - box[1]), "DEMO", font=schrift,
        fill=(255, 255, 255, 110), stroke_width=2, stroke_fill=(20, 20, 20, 110))
    kachel = kachel.rotate(30, expand=True)
    lage = Image.new("RGBA", bild.size, (0, 0, 0, 0))
    for y in range(0, hoehe, kachel.height + 16):
        for x in range(0, breite, kachel.width + 16):
            lage.alpha_composite(kachel, (x, y))
    return Image.alpha_composite(bild, lage)


def demo_groesse(aoi):
    """Gleiche Formel wie echt, aber auf max. 560 px Kante herunterskaliert —
    plausibel bleibt es, nur eben nicht 200 MB schwer."""
    breite, hoehe = raster_groesse(aoi["bbox"], aoi["aufloesung_m"])
    faktor = max(breite, hoehe) / 560
    if faktor > 1:
        breite, hoehe = round(breite / faktor), round(hoehe / faktor)
    return hoehe, breite


def weiches_feld(rng, hoehe, breite, grob):
    """Weiches Rauschen: grobes Zufallsraster, bilinear hochgezogen."""
    klein = rng.random((max(2, hoehe // grob), max(2, breite // grob)))
    bild = Image.fromarray((klein * 255).astype("uint8"))
    bild = bild.resize((breite, hoehe), Image.BILINEAR)
    return np.asarray(bild).astype("float32") / 255.0


def blob(hoehe, breite, zy, zx, radius):
    """Gauß-Fleck, Zentrum und Radius relativ zur Bildgröße (0..1)."""
    y, x = np.ogrid[:hoehe, :breite]
    d2 = ((y - zy * hoehe) ** 2 + (x - zx * breite) ** 2)
    return np.exp(-0.5 * d2 / (radius * min(hoehe, breite)) ** 2).astype("float32")


def demo_n(daten, rng):
    """Synthetisches n-Band: 0, wo der Composite NoData ist (dort war Wolke),
    sonst 1–5 wolkenfreie Aufnahmen. Ohne diese Ebene hätte die Demo nichts,
    woran sich Datengüte-Layer und hohle Zeitreihenpunkte zeigen ließen."""
    weich = weiches_feld(rng, daten.shape[0], daten.shape[1], 12)
    return np.where(np.isfinite(daten), 1 + np.rint(4 * weich),
                    0).astype("float32")


def demo_szene(lauf, aoi, rng, hoehe, breite):
    """Synthetische, plausible NDVI-Welt je AOI: Jahres-, Monats- und
    Wochen-Composites aus einem gemeinsamen Grundzustand. Der Harz erzählt
    die echte Geschichte nach: Absterben ab 2019, Wiederbegrünung ab 2023."""
    heute = dt.date.today()
    saison = lauf.konfig["saison"]
    feld = weiches_feld(rng, hoehe, breite, 24)
    textur = weiches_feld(rng, hoehe, breite, 6)

    if aoi["id"] == "harz":
        basis = 0.55 + 0.30 * feld + 0.08 * (textur - 0.5)
        verlust = np.clip(blob(hoehe, breite, 0.45, 0.40, 0.35)
                          + blob(hoehe, breite, 0.60, 0.68, 0.18), 0, 1)
        wieder = blob(hoehe, breite, 0.50, 0.45, 0.16)   # Wiederbegrünung
        jung = blob(hoehe, breite, 0.38, 0.62, 0.08)     # frisch ergrünt
        frisch = blob(hoehe, breite, 0.72, 0.22, 0.09)   # neuer Verlust

        def jahres_feld(jahr):
            sterbe = min(max((jahr - 2018) / 3, 0), 1)       # 2021 voll kahl
            gruen = min(max((jahr - 2022) / 3, 0), 1)        # ab 2023 zurück
            a = basis - 0.55 * verlust * sterbe + 0.40 * wieder * gruen
            return np.clip(a, -0.2, 0.95)
    elif aoi["id"] == "jadeweserport":
        basis = 0.20 + 0.50 * feld + 0.06 * (textur - 0.5)
        hafen = blob(hoehe, breite, 0.35, 0.62, 0.22)        # Versiegelung
        ausgleich = blob(hoehe, breite, 0.75, 0.28, 0.12)
        jung = frisch = np.zeros((hoehe, breite), "float32")

        def jahres_feld(jahr):
            bau = min(max((jahr - 2019) / 6, 0), 1)
            a = (basis - 0.45 * hafen * bau
                 + 0.25 * ausgleich * min(max((jahr - 2022) / 4, 0), 1))
            return np.clip(a, -0.2, 0.95)
    else:  # oldenburg & künftige Alltags-AOIs: sanfte Dynamik
        basis = 0.25 + 0.50 * feld + 0.06 * (textur - 0.5)
        jung = frisch = np.zeros((hoehe, breite), "float32")

        def jahres_feld(jahr):
            a = basis + 0.04 * math.sin(jahr * 2.1) * (feld - 0.5)
            return np.clip(a, -0.2, 0.95)

    jahre = {j: jahres_feld(j) for j in saison["baseline_jahre"]}
    zustand = jahres_feld(heute.year)

    # Monats-Composites (neuester zuerst) mit gezielten Abweichungen, damit
    # die Persistenz-Klassen 2/3/4 auch sichtbar werden.
    monate = []
    monatsliste = saison_monate_rueckwaerts(
        heute, lauf.konfig["klassifikation"]["persistenz_fenster_monate"],
        saison)
    for i, (jahr, monat) in enumerate(monatsliste):
        a = zustand + 0.03 * (weiches_feld(rng, hoehe, breite, 24) - 0.5)
        if i <= 1:
            a = a + 0.28 * jung          # nur die 2 jüngsten Monate: „jung"
        if i <= 4:
            a = a - 0.30 * frisch        # 5 Monate unter Baseline: „Verlust"
        monate.append((f"{jahr}-{monat:02d}", np.clip(a, -0.2, 0.95)))

    # 8 Demo-Wochen (aufsteigend), gelegentlich mit Wolken-NoData.
    montag = heute - dt.timedelta(days=heute.weekday())
    wochen = []
    for k in range(7, -1, -1):
        tag = montag - dt.timedelta(weeks=k)
        a = zustand + 0.05 * (weiches_feld(rng, hoehe, breite, 24) - 0.5)
        if rng.random() < 0.4:
            a = np.where(weiches_feld(rng, hoehe, breite, 10) > 0.78,
                         np.nan, a)
        wochen.append((iso_wochenlabel(tag), np.clip(a, -0.2, 0.95)))

    # Fensterjahre (M3): dasselbe Kalenderfenster in jedem Baseline-Jahr.
    # Bewusst leicht abweichend vom Saisonfeld — sonst wären Fensterbaseline
    # und Saison-Baseline in der Demo identisch, und der ganze Punkt der
    # tagesgenauen Klimatologie bliebe unsichtbar.
    fenster = {}
    for jahr in saison["baseline_jahre"]:
        a = jahre[jahr] + 0.06 * (weiches_feld(rng, hoehe, breite, 20) - 0.5)
        if rng.random() < 0.3:
            a = np.where(weiches_feld(rng, hoehe, breite, 10) > 0.85,
                         np.nan, a)
        fenster[jahr] = np.clip(a, -0.2, 0.95)

    return {"jahre": jahre, "zustand": zustand, "monate": monate,
            "wochen": wochen, "fenster": fenster}


def demo_falsecolor(ndvi, rng):
    """False-Color aus synthetischem NDVI rückgerechnet: NIR steigt mit der
    Vegetation, Rot/Grün sinken — reicht für ein plausibles Bild."""
    gruen = np.clip(np.nan_to_num(ndvi, nan=0.0), 0, 1)
    rausch = 0.02 * (rng.random(ndvi.shape).astype("float32") - 0.5)
    b08 = 0.12 + 0.38 * gruen + rausch
    b04 = 0.16 - 0.10 * gruen + rausch
    b03 = 0.10 - 0.04 * gruen + rausch
    rgba = np.zeros(ndvi.shape + (4,), np.uint8)
    for kanal, band in enumerate((b08, b04, b03)):
        rgba[..., kanal] = (np.clip(2.5 * band, 0, 1) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(ndvi), 255, 0)
    return rgba


def demo_aoi(lauf, aoi):
    """Alle Layout-Dateien einer AOI synthetisch erzeugen."""
    heute = dt.date.today()
    farbskala = lauf.konfig["farbskala"]
    aoi_dir = lauf.ausgabe / aoi["id"]
    hoehe, breite = demo_groesse(aoi)
    # Fester Seed je AOI: Demo-Läufe sind reproduzierbar.
    rng = np.random.default_rng(int.from_bytes(aoi["id"].encode(), "little")
                                % (2 ** 32))
    szene = demo_szene(lauf, aoi, rng, hoehe, breite)

    # Georeferenz der Demo: das metrische Gitter der AOI (R1); die Demo-Arrays
    # sind kleiner skaliert, from_bounds streckt die Pixel entsprechend.
    bbox_utm, epsg = gitter_von(aoi)

    # Wochen + aktuell. Die Demo kennt nur ein Gitter, also ist
    # aktuell-analyse.tif hier inhaltsgleich mit aktuell.tif — im Echtbetrieb
    # sind es zwei Auflösungen (10 m Anzeige, 20 m Analyse).
    for iso, daten in szene["wochen"]:
        schreibe_tiff(aoi_dir / "wochen" / f"{iso}.tif", daten, bbox_utm,
                      n=demo_n(daten, rng), epsg=epsg)
    schreibe_json(aoi_dir / "wochen" / "index.json",
                  {"wochen": [iso for iso, _ in szene["wochen"]]})
    neueste_iso, neueste = szene["wochen"][-1]
    neueste_n = demo_n(neueste, rng)
    schreibe_tiff(aoi_dir / "aktuell.tif", neueste, bbox_utm, n=neueste_n,
                  epsg=epsg)
    schreibe_tiff(aoi_dir / "aktuell-analyse.tif", neueste, bbox_utm,
                  n=neueste_n, epsg=epsg)
    speichere_png(aoi_dir / "aktuell.png", einfaerben(neueste, farbskala),
                  demo=True)
    lauf.merke(aoi["id"], "demo woche", "ok")

    # Baseline, Fensterbaseline, Persistenz, PCA — über die echten Rechenkerne,
    # damit die Demo genau das zeigt, was der Echtbetrieb rechnen würde.
    jahres_stack = np.stack(list(szene["jahre"].values()))
    baseline = baseline_median(jahres_stack)
    schreibe_tiff(aoi_dir / "baseline.tif", baseline, bbox_utm,
                  n=jahre_mit_wert(jahres_stack), epsg=epsg)
    speichere_png(aoi_dir / "baseline.png", einfaerben(baseline, farbskala),
                  demo=True)
    lauf.merke(aoi["id"], "demo baseline", "ok")

    fenster_stack = np.stack(list(szene["fenster"].values()))
    fenster_baseline = baseline_median(fenster_stack)
    n_jahre = jahre_mit_wert(fenster_stack)
    schreibe_tiff(aoi_dir / "baseline-fenster.tif", fenster_baseline,
                  bbox_utm, n=n_jahre, epsg=epsg)
    speichere_png(aoi_dir / "baseline-fenster.png",
                  einfaerben(fenster_baseline, farbskala), demo=True)
    schreibe_json(aoi_dir / "fenster-baseline.json",
                  fenster_baseline_bericht(heute - dt.timedelta(days=14),
                                           heute, list(szene["fenster"].keys()),
                                           n_jahre, demo=True))
    lauf.merke(aoi["id"], "demo fensterbaseline", "ok")

    monats_stack = np.stack([daten for _, daten in szene["monate"]])
    klassen = klassifiziere_persistenz(monats_stack, baseline,
                                       lauf.konfig["klassifikation"])
    schreibe_tiff(aoi_dir / "persistenz.tif", klassen, bbox_utm,
                  klassenraster=True, epsg=epsg)
    speichere_png(aoi_dir / "persistenz.png", persistenz_einfaerben(klassen),
                  demo=True)
    lauf.merke(aoi["id"], "demo persistenz", "ok")

    rgba, varianz_anteile = berechne_pca(jahres_stack)
    speichere_png(aoi_dir / "pca.png", rgba, demo=True)
    schreibe_json(aoi_dir / "pca.json",
                  {"varianz_anteile": varianz_anteile,
                   "methodik": PCA_METHODIK + " Demo-Lauf: synthetische Daten.",
                   "jahre": list(szene["jahre"].keys()), "demo": True})
    lauf.merke(aoi["id"], "demo pca", "ok")

    # 2021-Referenz + False-Color
    composite_2021 = szene["jahre"][2021]
    schreibe_tiff(aoi_dir / "composite-2021.tif", composite_2021, bbox_utm,
                  n=demo_n(composite_2021, rng), epsg=epsg)
    speichere_png(aoi_dir / "composite-2021.png",
                  einfaerben(composite_2021, farbskala), demo=True)
    speichere_png(aoi_dir / "falsecolor-2021.png",
                  demo_falsecolor(composite_2021, rng), demo=True)
    speichere_png(aoi_dir / "falsecolor-aktuell.png",
                  demo_falsecolor(neueste, rng), demo=True)
    lauf.merke(aoi["id"], "demo falsecolor/2021", "ok")

    aktualisiere_meta(lauf, aoi, demo=True, quelle=QUELLE_DEMO,
                      iso_woche=neueste_iso,
                      zeitfenster={
                          "von": (heute - dt.timedelta(days=14)).isoformat(),
                          "bis": heute.isoformat()},
                      guete=guete_kennzahlen(neueste_n),
                      hinweis="Demo-Daten: synthetisch erzeugt, "
                              "kein Copernicus-Abruf.")
    lauf.merke(aoi["id"], "demo meta", "ok")


def cmd_demo(lauf):
    """Synthetische Artefakte für alle AOIs — komplett ohne Netz. Bricht je
    AOI ab, wenn dort echte Daten liegen (meta.json mit demo:false), außer
    --force ist gesetzt."""
    for aoi in lauf.aois:
        meta = lies_json(lauf.ausgabe / aoi["id"] / "meta.json")
        if meta and meta.get("demo") is False and not lauf.force:
            lauf.merke(aoi["id"], "demo",
                       "FEHLER: echte Daten vorhanden (meta.json demo:false) "
                       "— Überschreiben nur mit --force")
            continue
        try:
            demo_aoi(lauf, aoi)
        except Exception as fehler:
            lauf.merke(aoi["id"], "demo", f"FEHLER: {fehler}")
    baue_stac(lauf)  # STAC gehört zum Layout, also auch zur Demo


# ---------------------------------------------------------------------------
# Laufsteuerung
# ---------------------------------------------------------------------------

class Lauf:
    """Bündelt Konfiguration, Token-Cache, PU-Buchhaltung und Protokoll."""

    def __init__(self, args):
        self.args = args          # Befehle mit eigenen Optionen (rueckblick) lesen hier
        self.ausgabe = Path(args.ausgabe)
        self.konfig = self._lade_konfig()
        self.jahr = args.jahr
        self.force = args.force
        self.protokoll = []       # (aoi, artefakt, status, pu)
        self.pu_je_aoi = {}       # PU-Gesamtsumme je AOI in diesem Lauf
        self.pu_gebucht = {}      # davon bereits in meta.json verbucht
        self._pu_stand = {}       # Snapshot für die Protokollzeilen
        self._token = None
        self._token_frist = 0.0   # monotonic-Zeitpunkt, ab dem erneuert wird

        alle_ids = [a["id"] for a in self.konfig["aois"]]
        if args.aoi and args.aoi not in alle_ids:
            sys.exit(f"Unbekannte AOI '{args.aoi}'. Bekannt: "
                     f"{', '.join(alle_ids)}")
        self.aois = [a for a in self.konfig["aois"]
                     if not args.aoi or a["id"] == args.aoi]

    def _lade_konfig(self):
        # aois.json ist die Quelle der Wahrheit; bei umgelenkter Ausgabe
        # greift die versionierte Fassung im Mirror.
        for kandidat in (self.ausgabe / "aois.json",
                         STANDARD_AUSGABE / "aois.json"):
            if kandidat.exists():
                return json.loads(kandidat.read_text(encoding="utf-8"))
        sys.exit(f"aois.json nicht gefunden (gesucht in {self.ausgabe} und "
                 f"{STANDARD_AUSGABE}).")

    def token(self):
        """OAuth-Token je Lauf wiederverwenden, aber ablaufbewusst: CDSE-
        Tokens leben nur ~10 Minuten, ein 'alles'-Erstlauf dauert länger.
        Erneuert wird kurz vor Ablauf — oder sofort, wenn process() nach
        einem 401 den Token verworfen hat (_token = None)."""
        if self._token is None or time.monotonic() >= self._token_frist:
            client_id, client_secret = lade_zugangsdaten()
            self._token, lebensdauer = hole_token(client_id, client_secret)
            # 60 s Puffer vor dem echten Ablauf, damit kein Request mit
            # einem Token startet, das unterwegs stirbt.
            self._token_frist = time.monotonic() + max(lebensdauer - 60, 60)
        return self._token

    def merke(self, aoi_id, artefakt, status):
        # PU-Differenz seit der letzten Protokollzeile dieser AOI.
        stand = self.pu_je_aoi.get(aoi_id, 0.0)
        delta = stand - self._pu_stand.get(aoi_id, 0.0)
        self._pu_stand[aoi_id] = stand
        self.protokoll.append((aoi_id, artefakt, status, delta))

    def drucke_protokoll(self):
        print(f"\n{'AOI':<15} {'Artefakt':<24} {'Status':<52} {'PU':>8}")
        print("-" * 101)
        for aoi_id, artefakt, status, pu in self.protokoll:
            kurz = status if len(status) <= 52 else status[:49] + "…"
            pu_text = f"{pu:.2f}" if pu else "–"
            print(f"{aoi_id:<15} {artefakt:<24} {kurz:<52} {pu_text:>8}")
        gesamt = sum(self.pu_je_aoi.values())
        if gesamt:
            print(f"{'':<40} {'PU gesamt:':>52} {gesamt:>8.2f}")

    @property
    def hat_fehler(self):
        return any(not status.startswith("ok")
                   for _, _, status, _ in self.protokoll)


BEFEHLE = {
    "alles": cmd_alles,
    "woche": cmd_woche,
    "baseline": cmd_baseline,
    "fensterbaseline": cmd_fensterbaseline,
    "historie": cmd_historie,
    "persistenz": cmd_persistenz,
    "pca": cmd_pca,
    "falsecolor": cmd_falsecolor,
    "stac": cmd_stac,
    "demo": cmd_demo,
    "aufraeumen": cmd_aufraeumen,
    "rueckblick": cmd_rueckblick,
}


def main():
    parser = argparse.ArgumentParser(
        prog="ndvi_batch.py",
        description="NDVI-Monitor — lokale Rechenwerkstatt "
                    "(Datenvertrag: scripts/ndvi/README.md)")
    unter = parser.add_subparsers(dest="befehl", required=True)
    hilfen = {
        "alles": "historie + baseline + fensterbaseline + persistenz + pca "
                 "+ falsecolor + woche + stac",
        "woche": "aktueller Wochen-Composite (Anzeige 10 m + Analyse 20 m)",
        "baseline": "saisonale Baseline aus den Jahres-Composites (Cache)",
        "fensterbaseline": "Baseline aus demselben Kalenderfenster der "
                           "Vorjahre (Bezug der Anomalie)",
        "historie": "Jahres-Saison-Composites in den Cache, 2021 ins Layout",
        "persistenz": "Klassenraster aus Monats-Composites vs. Baseline",
        "pca": "PC1–3 der Jahres-Composites als RGB + pca.json",
        "falsecolor": "False-Color-Composite der letzten 14 Tage",
        "stac": "STAC-Katalog/Collections/Items aus den Artefakten bauen",
        "demo": "synthetische Demo-Artefakte, komplett ohne CDSE-Zugang",
        "aufraeumen": "Demo-Reste aus dem Wochen-Archiv entfernen, index.json neu",
        "rueckblick": "vergangene Wochen echt nachrechnen (Zeitreihe füllen)",
    }
    for name, hilfe in hilfen.items():
        p = unter.add_parser(name, help=hilfe)
        p.add_argument("--aoi", help="nur diese AOI (Id aus aois.json)")
        p.add_argument("--jahr", type=int,
                       help="nur dieses Jahr (wirkt bei 'historie')")
        p.add_argument("--ausgabe", default=str(STANDARD_AUSGABE),
                       help=f"Ausgabeverzeichnis (Default: {STANDARD_AUSGABE})")
        p.add_argument("--force", action="store_true",
                       help="Demo darf vorhandene ECHTE Daten überschreiben")
        p.add_argument("--wochen", type=int,
                       help="Anzahl rückwärts zu rechnender Wochen (rueckblick, Default 8)")
        p.add_argument("--aufloesung", type=float,
                       help="Meter je Pixel für 'rueckblick' (Default: aufloesung_analyse_m)")

    args = parser.parse_args()
    lauf = Lauf(args)
    BEFEHLE[args.befehl](lauf)
    lauf.drucke_protokoll()
    sys.exit(1 if lauf.hat_fehler else 0)


if __name__ == "__main__":
    main()
