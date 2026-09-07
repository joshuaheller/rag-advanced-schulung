---
doc_id: it-backup-konzept
title: Backup- und Wiederherstellungskonzept
department: IT
access: [all]
version: "2025-05"
valid_from: 2025-05-01
status: current
---

# Backup- und Wiederherstellungskonzept

## 1. Grundprinzip

Die Aurelia Maschinenbau GmbH folgt der **3-2-1-Regel**: drei Kopien der Daten, auf zwei unterschiedlichen Medientypen, davon eine Kopie außer Haus (Rechenzentrum Frankfurt, georedundant). Alle Backups sind verschlüsselt (AES-256).

## 2. Sicherungsplan

| System | Vollsicherung | Inkrementell | RPO | RTO |
|---|---|---|---|---|
| SAP S/4HANA (ERP) | Sonntag 02:00 Uhr | alle 4 Stunden | 4 Stunden | 8 Stunden |
| Fileserver / SharePoint | Sonntag 03:00 Uhr | täglich 22:00 Uhr | 24 Stunden | 12 Stunden |
| Konstruktionsdaten (PLM) | Samstag 23:00 Uhr | täglich 20:00 Uhr | 24 Stunden | 24 Stunden |
| E-Mail (Microsoft 365) | kontinuierlich (Veeam for M365) | – | 1 Stunde | 4 Stunden |
| Maschinensteuerungsprogramme (CNC) | monatlich | wöchentlich | 7 Tage | 48 Stunden |

RPO = maximal tolerierter Datenverlust, RTO = maximale Wiederherstellungszeit.

## 3. Aufbewahrung

- Tägliche und inkrementelle Sicherungen: **90 Tage**
- Monatliche Vollsicherungen: 12 Monate
- Jahressicherung (31. Dezember): **10 Jahre** (steuerrechtliche Aufbewahrung)

## 4. Wiederherstellungstests

Die IT führt **quartalsweise** einen dokumentierten Restore-Test für SAP und den Fileserver durch. Einmal jährlich wird ein vollständiger Disaster-Recovery-Test im Ausweichrechenzentrum durchgeführt.

## 5. Wiederherstellung auf Anfrage

Mitarbeitende können die Wiederherstellung einzelner Dateien über ein ServiceNow-Ticket (Kategorie „Datenwiederherstellung") beantragen. Standardbearbeitungszeit: 1 Arbeitstag. Wiederherstellungen ganzer Systeme werden nur über den Incident-Response-Prozess ausgelöst.

## 6. Verantwortlichkeiten

Verantwortlich für das Backup-Konzept ist die Leitung IT-Infrastruktur. Die Überwachung der Sicherungsläufe erfolgt täglich; fehlgeschlagene Sicherungen werden als Incident der Stufe S2 behandelt.
