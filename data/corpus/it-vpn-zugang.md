---
doc_id: it-vpn-zugang
title: Anleitung VPN-Zugang (GlobalProtect)
department: IT
access: [all]
version: "2026-02"
valid_from: 2026-02-01
status: current
---

# Anleitung VPN-Zugang (GlobalProtect)

## 1. Hintergrund

Seit Februar 2026 nutzt die Aurelia Maschinenbau GmbH **Palo Alto GlobalProtect** als VPN-Lösung. Der bisherige Client Cisco AnyConnect wurde am 28. Februar 2026 abgeschaltet und darf nicht mehr verwendet werden.

## 2. Voraussetzungen

- Firmengerät mit aktuellem Windows 11 oder macOS
- Aktiviertes MFA (Microsoft Authenticator)
- Der GlobalProtect-Client ist auf allen Firmengeräten vorinstalliert (Softwareportal, Kategorie „Sicherheit")

## 3. Verbindung herstellen

1. GlobalProtect über das Symbol in der Taskleiste öffnen.
2. Portaladresse eingeben: `vpn.aurelia-maschinenbau.example`
3. Mit der Firmen-E-Mail-Adresse und dem Windows-Passwort anmelden.
4. MFA-Anfrage in der Authenticator-App bestätigen.
5. Nach erfolgreicher Verbindung erscheint das Symbol grün.

## 4. Regeln

- **Split-Tunneling ist deaktiviert**: Während der VPN-Verbindung läuft der gesamte Datenverkehr über das Unternehmensnetz.
- Die VPN-Verbindung wird nach **12 Stunden** automatisch getrennt und muss neu aufgebaut werden.
- Die gleichzeitige Verbindung ist auf **zwei Geräte pro Person** begrenzt.
- Der Zugriff auf das Maschinennetz (VLAN 40) ist über VPN nur für den Bereich Service mit gesonderter Freigabe möglich.

## 5. Häufige Probleme

| Problem | Lösung |
|---|---|
| „Portal nicht erreichbar" | Internetverbindung prüfen; Hotel-WLAN-Login im Browser abschließen |
| MFA-Anfrage kommt nicht an | Uhrzeit des Smartphones synchronisieren; Authenticator-App neu starten |
| Verbindung bricht ständig ab | Energiesparmodus des WLAN-Adapters deaktivieren |
| Zugriff auf Fileserver fehlt | Verbindung trennen und neu aufbauen; Ticket an Servicedesk, falls weiterhin fehlend |

Bei allen weiteren Problemen: IT-Servicedesk, Durchwahl **-4444**.
