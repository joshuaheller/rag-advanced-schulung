---
doc_id: it-incident-response
title: Incident-Response-Prozess
department: IT
access: [all]
version: "2025-09"
valid_from: 2025-09-01
status: current
---

# Incident-Response-Prozess

## 1. Ziel

Dieser Prozess beschreibt die Meldung, Klassifizierung und Bearbeitung von IT-Störungen und Sicherheitsvorfällen bei der Aurelia Maschinenbau GmbH.

## 2. Meldung

Alle Störungen und Sicherheitsvorfälle werden an den IT-Servicedesk gemeldet: über ServiceNow, per E-Mail an servicedesk@aurelia-maschinenbau.example oder telefonisch unter Durchwahl **-4444** (Mo–Fr 07:00–18:00 Uhr). Außerhalb dieser Zeiten ist die Rufbereitschaft für Vorfälle der Stufen S1 und S2 unter der Nummer im Intranet erreichbar.

## 3. Klassifizierung und Reaktionszeiten

| Stufe | Beschreibung | Beispiele | Reaktionszeit | Lösungsziel |
|---|---|---|---|---|
| S1 – Kritisch | Produktionsstillstand oder Sicherheitsvorfall mit Datenabfluss | SAP nicht verfügbar, Ransomware, Ausfall Maschinennetz | 15 Minuten | 4 Stunden |
| S2 – Hoch | Wesentliche Einschränkung eines Bereichs | Fileserver nicht erreichbar, fehlgeschlagenes Backup, Phishing mit Klick | 1 Stunde | 8 Stunden |
| S3 – Mittel | Einzelne Person eingeschränkt | Laptop defekt, Softwarefehler | 4 Stunden | 2 Arbeitstage |
| S4 – Niedrig | Kosmetisch, Anfrage | Anfrage zu Berechtigungen, Drucker-Problem | 1 Arbeitstag | 5 Arbeitstage |

Die Reaktionszeit ist die Zeit bis zur ersten qualifizierten Rückmeldung an die meldende Person.

## 4. Sicherheitsvorfälle mit personenbezogenen Daten

Bei jedem Vorfall, bei dem personenbezogene Daten betroffen sein könnten (z. B. verlorener Laptop, falscher E-Mail-Empfänger, unbefugter Zugriff), ist **innerhalb von 24 Stunden die Datenschutzbeauftragte** zu informieren. Die Datenschutzbeauftragte entscheidet über die Meldung an die Aufsichtsbehörde gemäß Datenschutzrichtlinie.

## 5. Eskalation

S1-Vorfälle werden sofort an die Leitung IT und die Geschäftsführung eskaliert; ein Krisenstab wird innerhalb von 30 Minuten einberufen. Die Kommunikation nach außen erfolgt ausschließlich über die Geschäftsführung.

## 6. Dokumentation und Nachbereitung

Jeder S1- und S2-Vorfall wird innerhalb von 5 Arbeitstagen nach Abschluss in einem Post-Incident-Report dokumentiert (Ursache, Zeitverlauf, Maßnahmen, Lessons Learned). Der Report wird im Confluence-Bereich „IT-Security" abgelegt.
