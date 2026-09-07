---
doc_id: prod-fehlercodes
title: Fehlercodes und Störungsbehebung AX-Baureihe
department: Service
access: [all]
version: "2026-01"
valid_from: 2026-01-01
status: current
---

# Fehlercodes und Störungsbehebung AX-Baureihe

Gilt für AX-200, AX-300 und AX-300 Plus. Fehlercodes werden in der Steuerung im Alarmfenster angezeigt.

## 1. Fehlercodes

| Code | Bedeutung | Sofortmaßnahme | Weiteres Vorgehen |
|---|---|---|---|
| E-101 | Spindelübertemperatur | Programm anhalten, Spindel 20 Minuten abkühlen lassen | Kühlmittelstand und Spindelkühlung prüfen; bei Wiederholung Service |
| E-102 | Spindellagerverschleiß erkannt (Vibration) | Spindeldrehzahl auf max. 6.000 U/min begrenzen | Service innerhalb von 5 Arbeitstagen |
| E-115 | Werkzeugwechsler blockiert | Not-Halt lösen, Greifer manuell in Grundstellung bringen | Werkzeugaufnahme auf Späne prüfen |
| E-118 | Werkzeug nicht im Magazin gefunden | Werkzeugtabelle mit Magazinbelegung abgleichen | – |
| E-207 | Kühlmittelstand niedrig | Kühlmittel nachfüllen | Tank auf Leckage prüfen |
| E-209 | Kühlmittelfilter verstopft | Filter reinigen oder tauschen | – |
| E-301 | Schutztür nicht verriegelt | Tür schließen, Verriegelung prüfen | Bei Defekt Service (sicherheitsrelevant, kein Überbrücken) |
| E-330 | Not-Halt aktiv | Ursache beseitigen, Not-Halt entriegeln | Referenzfahrt aller Achsen durchführen |
| E-412 | Referenzfahrt Achse fehlgeschlagen | Achse per Handrad aus Endlage fahren | Referenzschalter prüfen |
| E-415 | Schleppfehler Achse | Vorschub reduzieren | Führungen auf Schmierung prüfen (Wartungsintervall) |
| E-501 | Druckluft unter 5 bar | Druckluftversorgung prüfen | Kompressor / Wartungseinheit prüfen |
| E-620 | Kommunikationsfehler Steuerung – Antrieb | Maschine aus- und nach 60 Sekunden wieder einschalten | Bei Wiederholung Service; Logdatei aus Diagnose sichern |
| E-700 | Schmiersystem: Ölstand Reservoir zu niedrig | Spindelöl nachfüllen | Wartungsintervall Spindelöl prüfen |

## 2. Allgemeine Vorgehensweise bei Störungen

1. Alarmtext und Fehlercode notieren, Bildschirmfoto der Diagnoseseite anfertigen.
2. Sofortmaßnahme aus der Tabelle durchführen.
3. Störung im Maschinenlogbuch eintragen.
4. Wenn die Störung innerhalb einer Stunde dreimal auftritt oder sicherheitsrelevant ist (E-301, E-330): Service-Hotline +49 721 000-500 kontaktieren.

## 3. Fernwartung

Für AX-300 und AX-300 Plus kann der Service per „Aurelia Remote" auf die Steuerung zugreifen. Der Kunde gibt den Zugriff pro Sitzung am Bedienpanel frei; jede Sitzung wird protokolliert.
