# fieldbook-code

> **English:** Analysis scripts and result data behind the *Feldbuch* (field book),
> a German-language remote-sensing blog at
> [geophora.de/blog](https://www.geophora.de/blog/). Every published figure carries
> a "Kartenpass" (map passport) naming the exact script call that produced it —
> reproducible from this repository. Code and docs are in German, matching the
> blog's audience; the methods (Sentinel-2 median composites, metric UTM grid,
> Theil–Sen slopes with Mann–Kendall tests) are explained at
> [geophora.de/blog/methodik](https://www.geophora.de/blog/methodik/).

Versionierte Belege zum **Feldbuch** auf [geophora.de/blog/](https://www.geophora.de/blog/):
die Auswerteskripte und Datendateien hinter jeder veröffentlichten Zahl. Jede Folge
trägt im Abschnitt „Methoden & Quellen" einen **Kartenpass** mit dem exakten
Skriptaufruf — die Pfade dort sind Pfade in diesem Repository, jeder Aufruf ist hier
unverändert wiederholbar.

Die Verfahren (was sie tun, warum sie gewählt sind, was sie nicht können) erklärt die
[Methodik-Seite des Feldbuchs](https://www.geophora.de/blog/methodik/).

## Aufbau

| Pfad | Inhalt |
|---|---|
| `scripts/ndvi/` | Rechenwerkstatt: Sentinel-2-Composites über die CDSE Processing API (`ndvi_batch.py`), Punkt- und Trendauswertungen, CRS-Helfer. Details: [`scripts/ndvi/README.md`](scripts/ndvi/README.md) |
| `scripts/ortstermin/` | Auswertungen der Feldbuch-Folgen: Fundstellen, Trends (Theil-Sen + Mann-Kendall), Skalenprobe, Kartenbilder |
| `scripts/klima/` | DWD-Klimareihen als Kontext |
| `docs/daten/` | Ergebnis-JSONs — jede tragende Zahl einer Folge steht hier, inklusive des wörtlichen Aufrufs (`"aufruf"`), der sie erzeugt hat |
| `ftp-mirror-geophora/tiles/ndvi/aois.json` | Gebiets- und Fensterkonfiguration (Kopie der öffentlichen Datei von geophora.de; der Pfad spiegelt die Arbeitskopie, damit die Aufrufe unverändert laufen) |

## Nachrechnen

1. Python ≥ 3.11, dann `pip install -r scripts/ndvi/requirements.txt`
2. Kostenloses Konto im [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
   anlegen, OAuth-Client erzeugen und nach dem Muster in
   [`scripts/ndvi/env-VORLAGE.txt`](scripts/ndvi/env-VORLAGE.txt) eine `scripts/ndvi/.env`
   schreiben (wird nie eingecheckt).
3. Rohdaten-Cache aufbauen, z. B. `python scripts/ndvi/ndvi_batch.py historie --aoi oldenburg`
   — die Jahres-Composites entstehen lokal unter `scripts/ndvi/cache/` (nicht im Repo,
   mehrere MB je Jahr; das kostenlose CDSE-Kontingent reicht dafür weit).
4. Den Kartenpass-Aufruf aus der Folge kopieren und ausführen; das Ergebnis-JSON
   muss den Dateien in `docs/daten/` entsprechen.

Analysegitter ist metrisches WGS 84/UTM (Deutschland: EPSG:32632), 20 m; Trends
rechnen als Theil-Sen-Steigung mit Mann-Kendall-Test. Begründungen und Grenzen:
[Methodik-Seite](https://www.geophora.de/blog/methodik/).

## Was hier nicht liegt

Der Rohdaten-Cache (entsteht lokal, Schritt 3), die Website selbst und die
Zugangsdaten (`.env`).

## Lizenzen

- **Code:** [MIT](LICENSE)
- **Datendateien** in `docs/daten/`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.de)
  mit den Herkunftsvermerken der jeweiligen Quelle — Details je Datei in
  [`docs/daten/LIZENZ.md`](docs/daten/LIZENZ.md). Sentinel-Ableitungen enthalten
  modifizierte Copernicus-Sentinel-Daten (Contains modified Copernicus Sentinel data);
  OSM-Ableitungen stehen unter ODbL.

Robert Rettig · [robertrettig.de](https://www.robertrettig.de/) ·
[ORCID 0000-0002-4632-1286](https://orcid.org/0000-0002-4632-1286)
