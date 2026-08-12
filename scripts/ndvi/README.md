# NDVI-Monitor — Datenvertrag & Betrieb

Verbindlicher Vertrag zwischen den drei Bausteinen:
`layer-explorer/ndvi.html` (Anzeige) · `php/ndvi-refresh.php` (Wochen-Cron) ·
`scripts/ndvi/ndvi_batch.py` (lokale Rechenwerkstatt). Wer hier etwas ändert,
ändert es an allen drei Stellen.

## Quellen & Limits (verifiziert 2026-08-04)

- **OAuth:** `POST https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`,
  Body `grant_type=client_credentials&client_id=…&client_secret=…` (x-www-form-urlencoded).
  Token wiederverwenden, nie pro Request neu holen (Token-Endpoint ist rate-limited).
- **Processing:** `POST https://sh.dataspace.copernicus.eu/process/v1` (Bearer).
  `input.bounds = {bbox:[w,s,o,n], properties:{crs:"http://www.opengis.net/def/crs/OGC/1.3/CRS84"}}`,
  `input.data[0] = {type:"sentinel-2-l2a", dataFilter:{timeRange:{from,to}, maxCloudCoverage:75}}`,
  `output.width/height ≤ 2500`. **Ein Bild-Output pro Request** (kein TAR-Parsing):
  FLOAT32-GeoTIFF und PNG werden als zwei getrennte Requests geholt.
- **Kontingent (Copernicus General User, kostenlos):** 10 000 PU/Monat, 300 PU/min,
  50 000 Requests/Monat. PU-Verbrauch wird aus dem Antwort-Header
  (`x-processingunits-spent`, falls vorhanden) in `meta.json` geloggt.
- **PU-Kalibrierung (gemessen 2026-08-04):** EIN Saison-Composite Mai–Sep bei
  10 m kostete **1730 PU** — Saison-Requests skalieren mit der Orbitzahl.
  Konsequenz im Vertrag: mehrjährige **Analyse-Artefakte** (Historie/Baseline/
  Persistenz/PCA/falsecolor-2021) rechnen mit `aufloesung_analyse_m` (20 m)
  und die Jahres-Saison ist Peak-Season **Jun–Aug** (Wochen 23–35) — zusammen
  ≈ Faktor 7 billiger (~250 PU je Jahres-Composite). Der Wochenlauf (Batch
  `woche` + PHP-Cron) holt **zwei** Composites: 10 m fürs Anzeigebild und
  20 m fürs Analyse-Gitter (~44 PU, s. u.); Persistenz-Monatsfenster bleiben
  Mai–Sep (`start_monat`/`end_monat`). Die Fensterbaseline kostet einmalig
  ~44 PU je Baseline-Jahr und AOI (sechs Jahre ≈ 260 PU).

## Evalscript-Konventionen (alle Composites)

- `//VERSION=3`, `mosaicking:"ORBIT"`, Inputs `B04, B08, SCL, dataMask`.
- Wolken-/Störmaske: SCL-Klassen **3 (Schatten), 8, 9 (Wolken), 10 (Zirrus), 11 (Schnee)**
  werden verworfen; zusätzlich `dataMask === 1` verlangt.
- Composite = **Median** der gültigen NDVI-Werte pro Pixel über alle Orbits im
  Zeitfenster (selbst implementiert: sammeln, sortieren, Mittelelement).
- `NDVI = (B08 − B04) / (B08 + B04)`; Division nur bei Nenner > 0.
- FLOAT32-Ausgabe: **zwei Bänder**, `sampleType:"FLOAT32"` —
  **Band 1** = NDVI-Median (NoData = **NaN**), **Band 2** = `n`, die Zahl der
  gültigen (wolkenfreien) Beobachtungen im Zeitfenster. `n = 0` ⇒ Band 1 = NaN.
  Bei den **Aggregaten** (`baseline`, `baseline-fenster`) steht in Band 2 die
  Zahl der **Jahre** mit gültigem Wert. `persistenz.tif` bleibt einbandig.
  Lesende Seiten prüfen die Bandzahl: Bestandsdateien aus der Zeit davor sind
  einbandig und dürfen nicht brechen (Band 1 ist immer der NDVI).
- PNG-Ausgabe: NDVI über die **Farbskala aus `aois.json → farbskala`** eingefärbt
  (linear interpolierte Stützstellen, Domäne geklemmt), NoData transparent (RGBA).
- False-Color-PNG: `B08,B04,B03`-Median-Composite, je Band 2.5×reflectance geklemmt.

## Dateilayout auf dem Webspace — `ftp-mirror-geophora/tiles/ndvi/`

```
aois.json                  Quelle der Wahrheit (versioniert; alles andere ist Artefakt)
catalog.json               STAC-1.0-Root (type Catalog, links auf die Collections)
<aoi>/
  meta.json                aktueller Stand, s. Schema unten
  collection.json          STAC Collection (bbox aus aois.json, Lizenz CC BY 4.0,
                           Provider Copernicus/ESA + GEOPHORA als processor)
  aktuell.tif              Wochen-Composite in aufloesung_m — reines ANZEIGE-Produkt
                           (COG wenn per Batch erzeugt, einfaches GeoTIFF wenn vom
                           PHP-Cron durchgereicht)
  aktuell.png              dito als eingefärbtes RGBA-PNG
  aktuell-analyse.tif      derselbe Composite in aufloesung_analyse_m — ANALYSE-Gitter,
                           inhaltsgleich mit wochen/<aktuelle ISO>.tif
  baseline.tif|.png        Saison-Normal Jun–Aug: Median der Jahres-Saison-Composites
                           (Bezug der Persistenz, NICHT der Anomalie)
  baseline-fenster.tif|.png  tagesgenau gematchte Klimatologie: Median über dasselbe
                           Kalenderfenster der baseline_jahre (Bezug der Anomalie)
  fenster-baseline.json    {fenster:{von,bis}, jahre:[…], n_jahre_min,
                           n_jahre_median, methodik}
  persistenz.tif|.png      Klassenraster (UINT8, s. Klassen unten)
  pca.png + pca.json       PC1-PC3 als RGB + {varianz_anteile:[…], methodik:"…"}
  falsecolor-aktuell.png   False-Color des aktuellsten Composites
  falsecolor-2021.png      False-Color des Sommer-2021-Composites
  composite-2021.tif|.png  Sommer-2021-Referenz (Swipe-Partner)
  wochen/index.json        {"wochen":["2026-W18", …]}  (aufsteigend sortiert)
  wochen/<JJJJ>-W<WW>.tif  FLOAT32-NDVI je Woche (Klick-Zeitreihe liest hier)
  items/<artefakt>.json    STAC Items (ein Item je Artefakt, assets → tif/png)
```

Georeferenz-Konvention (seit Beschluss R1, 12.08.2026): Analysen laufen im
**metrischen UTM-CRS** der AOI (Deutschland fest EPSG:32632, Süd-AOIs 327xx;
Herleitung in `crs_helfer.py`). Die lon/lat-`bbox` in `aois.json` bleibt reine
Gebietsbeschreibung; gerechnet wird auf der nach **außen aufs Gitter
gesnappten** UTM-bbox — jede Zelle exakt `aufloesung_m²` groß. Der Batch
schreibt `crs` und `bbox_utm` in die meta.json; der Wochen-Cron
(ndvi-refresh.php) liest beides von dort (bbox in Metern + CRS-URI
`…/EPSG/0/<code>`) und leitet **keine** Geodäsie in PHP her. Breite/Höhe =
Kantenlänge/Auflösung (ganzzahlig); > 2500 px bricht ab statt zu klemmen.
Das Frontend übersetzt Klicks über den GeoTIFF-Transform in Pixel und
projiziert Overlay-Ecken mit einer eingebetteten Krüger/Karney-Umrechnung
zurück nach WGS84. Alt-Archive ohne `crs`-Feld werden weiter als lineares
Grad-Raster gelesen, vom Cron aber **nicht mehr fortgeschrieben**
(Übergangs-Sperre gegen ein Mischarchiv).

## meta.json — Schema

```json
{
  "aoi": "harz",
  "demo": false,
  "iso_woche": "2026-W31",
  "zeitfenster": {"von": "2026-07-20", "bis": "2026-08-03"},
  "max_wolken": 75,
  "pu_verbraucht": 123.4,
  "aktualisiert": "2026-08-03T02:10:00Z",
  "quelle": "Sentinel-2 L2A · Copernicus Data Space Ecosystem",
  "crs": "EPSG:32632",
  "bbox_utm": [594700.0, 5734480.0, 623380.0, 5745160.0],
  "aufruf": "python scripts/ndvi/ndvi_batch.py alles",
  "guete": {"n_median": 4.0, "anteil_ohne_daten": 0.08, "anteil_n1": 0.13},
  "hinweis": ""
}
```

`crs`/`bbox_utm` (seit R1): EPSG-Code und gesnappte bbox des
**Analysegitters** in Metern — Pflichtquelle des Wochen-Crons; fehlt `crs`,
setzt der Cron aus (Alt-Archiv). `aufruf` ist der Kartenpass (Rezeptur § 8.1).

`guete` beschreibt den **Analyse-Composite** (`aktuell-analyse.tif`, Band 2):
`n_median` = Median der wolkenfreien Beobachtungen über die Pixel mit
mindestens einer Aufnahme, `anteil_ohne_daten` = Anteil Pixel mit n = 0,
`anteil_n1` = Anteil Pixel mit genau einer Beobachtung (unsicher).

`demo:true` ⇒ das Frontend zeigt einen Demo-Banner. `pu_verbraucht` kumuliert
je Lauf, `hinweis` transportiert Warnungen („Baseline fehlt noch" o. ä.).
Der Wochen-Cron schreibt meta.json fortschreibend und **erhält `demo:true`**,
solange noch Demo-Artefakte (Baseline/Persistenz/PCA) liegen — erst der echte
Batch-Deploy (`aktualisiere_meta`) setzt `demo:false`.

## Wochen-Archiv: Provenienz (Lehre aus 2026-08-05)

Der Demo-Lauf schreibt in **dasselbe** `wochen/`-Archiv wie echte Läufe. Nach dem
Umstieg auf echte Daten blieben synthetische Wochen liegen → die Klick-Zeitreihe
mischte Rauschen mit Messwerten und widersprach der (korrekt gerechneten)
Anomalie. Regeln daraus:

- **Erkennungsmerkmal:** Echte Wochen liegen auf **einem der beiden erlaubten
  Gitter** der AOI — `raster_groesse(bbox, aufloesung_m)` oder
  `raster_groesse(bbox, aufloesung_analyse_m)`. Der Live-Lauf schreibt seit M2
  das Analyse-Gitter (identisch mit `aktuell-analyse.tif`), der Rückblick
  ebenfalls; nur die Demo rechnet gröber.
- `ndvi_batch.py aufraeumen` entfernt alles, was zu **keinem** der beiden Gitter
  passt, löscht das zugehörige STAC-Item und schreibt `wochen/index.json` aus dem
  Dateibestand neu. **Nach jedem Wechsel Demo → echt ausführen.**
- `ndvi_batch.py rueckblick --wochen 8 [--aufloesung 20]` füllt das Archiv mit
  echten, rückdatierten 14-Tage-Fenstern (Default: Analyse-Auflösung, gemessen
  **44 PU je Woche und AOI** statt ~318 PU bei 10 m).
- `wochen/index.json` ist reine Anzeigequelle — nie von Hand pflegen.

## Persistenz-Klassen (UINT8 in persistenz.tif, Farben im PNG)

Grundlage: Monats-Median-Composites der letzten `persistenz_fenster_monate`
Saison-Monate (Mai–Sep; Wintermonate zählen nicht) gegen die Baseline B.
Schwellen Δ aus `aois.json → klassifikation`.

| Wert | Klasse | Regel | PNG-Farbe |
|---|---|---|---|
| 0 | keine Aussage / NoData | zu wenig gültige Monate (< 2) | transparent |
| 1 | stabil | weder dauerhaft über noch unter B±Δ | transparent |
| 2 | **wieder grün (stabil)** | ≥ `persistenz_min_monate` Monate > B+Δ | kräftig grün #1f9d55 |
| 3 | ergrünt (jung) | aktuellster Monat > B+Δ, aber < min. Monate | hellgrün #7ed957 |
| 4 | Vegetationsverlust | ≥ `persistenz_min_monate` Monate < B−Δ | rot #d94f2b |

Anomalie („aktuell vs. Baseline") wird **nicht** vorgerechnet — sie entsteht
client-seitig in ndvi.html aus `aktuell.tif` − `baseline.tif` (geotiff.js).

## PCA (multitemporal)

Stack = Jahres-Saison-Composites (`baseline_jahre`, FLOAT32-NDVI). Pixel mit
NoData in irgendeinem Jahr ⇒ NoData. Bänder zentrieren (Mittelwert je Jahr),
PCA über Kovarianz (numpy SVD). RGB = PC1, PC2, PC3, je 2.–98. Perzentil auf
0–255 gestreckt. `pca.json` dokumentiert Varianzanteile + Kurzmethodik.
Interpretation (steht auch im UI): PC1 ≈ mittlere Grünheit, PC2/PC3 ≈
Veränderungsmuster (Sturm-/Kahlflächen, Wiederbegrünung).

## CLI der Rechenwerkstatt

```
python ndvi_batch.py alles                 # historie + baseline + fensterbaseline + persistenz + pca + falsecolor + woche + stac
python ndvi_batch.py woche                 # Wochen-Composite: Anzeige (10 m) + Analyse (20 m, = wochen/<ISO>.tif)
python ndvi_batch.py fensterbaseline       # Median über dasselbe Kalenderfenster der baseline_jahre (20 m)
python ndvi_batch.py baseline|historie|persistenz|pca|falsecolor|stac
python ndvi_batch.py demo                  # synthetische Demo-Artefakte (ohne CDSE-Zugang)
   [--aoi harz] [--jahr 2021] [--ausgabe <pfad>]
```

Credentials: `scripts/ndvi/.env` (gitignored, Vorlage `env-VORLAGE.txt`).
Ausgabe schreibt direkt nach `ftp-mirror-geophora/tiles/ndvi/` (per `--ausgabe`
umlenkbar). Artefakte sind gitignored — Deploy per FTP, danach hält der Cron
`aktuell.*` und `wochen/` nach.

## php/ndvi-refresh.php (Wochen-Cron)

Hausmuster von `backup.php`: Direktaufruf-Guard, CLI frei, HTTP nur mit
`?key=` = `NDVI_CRON_KEY` (hash_equals; leer ⇒ 403). Ablauf je AOI:
Token holen (einmal je Lauf) → TIFF-Request → PNG-Request → bei Erfolg beide
per `@file_put_contents(…, LOCK_EX)` schreiben (aktuell.* + wochen/<iso>.tif
+ index.json + meta.json + STAC-Item), bei JEDEM Fehler: error_log + alter
Stand bleibt liegen (fail-open), Antwort im Klartext wie backup.php.
Neue config.local.php-Schlüssel: `NDVI_CRON_KEY`, `CDSE_CLIENT_ID`,
`CDSE_CLIENT_SECRET`. `.htaccess`-Whitelist: `ndvi-refresh` ergänzen.
