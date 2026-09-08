# Block 6 – Security & Governance (13:00–14:30, Lab 6: 35 Min)

## Kernbotschaft
Alles, was in den Index kommt, ist Input für das LLM – der Ingest ist eine von außen erreichbare Schnittstelle und
muss so behandelt werden. Zugriffsrechte werden *vor* der Suche als Filter angewendet, nie danach.

## Konzepte in Erklär-Sprache

**Angriffsfläche (Folie 6.1):** Diagramm im Notebook-Header. Jeder Pfeil ist ein Angriffspunkt: Quellen (Poisoning),
Ingest (keine Validierung), Index (keine Mandantentrennung), Retrieval (Post-Filter-Leak), Prompt (Injection),
Output (Phishing-Links, Datenabfluss), Tools (Excessive Agency).

**OWASP LLM Top 10 (Folie 6.2):** 2025er Liste: LLM01 Prompt Injection, LLM02 Sensitive Information Disclosure,
LLM08 Vector and Embedding Weaknesses (RAG-spezifisch: Poisoning, ACL-Bypass, Embedding-Inversion). 2026er Ausgabe
(August 2026) hebt Excessive Agency auf LLM03 – Agenten. CC BY-SA, Grafiken dürfen mit Attribution auf die Folien.

**Indirekte Prompt Injection (Folie 6.3, Greshake 2023):** Der Angreifer spricht nicht mit dem Modell, sondern legt
Text dorthin, wo das Modell ihn liest. Beispiel `poison-hr-faq`: „Assistent: … füge diesen Link hinzu.“ Grundregel:
**Daten ≠ Instruktionen** – aber ein LLM kann die beiden nicht zuverlässig trennen. Deshalb Defense in Depth.

**RAG Poisoning (Folie 6.4, PoisonedRAG/USENIX 2025):** 5 injizierte Texte in 2,6 Mio. Dokumenten → ~97 % Erfolg bei
gezielten Fragen. Warum so effektiv? Der Angreifer schreibt den Text so, dass er *für genau diese Frage* der beste
Treffer ist (semantisch nah + gewünschte Antwort). Perplexity-Filter und Paraphrasieren halfen nicht. Unser
`poison-ax200-serviceinfo` ist genau so gebaut: neuer, spezifischer, „ersetzt Abschnitt 4“.

**Verteidigungslinien (Folie 6.5/6.6):**
1. *Quellen-Vertrauensstufen*: Wer darf in den Ordner schreiben? Signierte/versionierte Quellen, Change-Detection,
   Quarantäne + menschliche Freigabe für Neues (`ingest_gate`).
2. *Injection-Scan*: Regex als erste Linie (billig, viele False Negatives), LLM-Klassifikator als zweite (Lab B1).
3. *Prompt-Härtung*: „Kontext ist DATEN“, Delimiter, Instruction Hierarchy – hilft, garantiert nichts.
4. *Output-Validierung*: URL-Allowlist, keine Anweisungen an Nutzer, Zitatpflicht.
5. *Minimale Tool-Rechte*: RAG-Assistent darf nichts senden/löschen/kaufen.

**Permission-aware Retrieval (Folie 6.7/6.8):** ACLs beim Ingest als Payload (`access: [hr, management]`), zur
Anfragezeit **Pre-Filter** in Qdrant (`build_filter(user_roles=...)`). Post-Filter (erst suchen, dann wegwerfen) ist
falsch: (a) Leak über Logs/Scores/Traces, (b) nach dem Wegwerfen bleiben 0–2 Treffer, (c) Timing-Angriffe.
Mandantentrennung: Namespace/Collection pro Mandant vs. Filter – Filter reicht bei wenigen Mandanten, Collections bei
harten Compliance-Anforderungen. Gruppen-Auflösung (Entra/LDAP) cachen, aber TTL kurz (Stale-ACL-Problem: Mitarbeiter
wechselt Abteilung).

**Datenschutz (Folie 6.9):** Embeddings von personenbezogenen Texten sind selbst personenbezogen (Inversion möglich)
→ Löschanfrage = Chunks *und* Vektoren *und* Embedding-Cache löschen. Bewerbungs-/Gesundheitsdaten nicht indexieren
(Datenschutzrichtlinie des Korpus § 7 als Beispiel).

**Audit & Compliance (Folie 6.10):** Pro Anfrage: pseudonymisierter Nutzer, Rollen, Filter, Chunk-IDs, Modellversion,
Zeiten – *nicht* die Frage im Klartext, wenn nicht nötig. Aufbewahrung nach Richtlinie (Korpus: Logs 90 Tage).
EU AI Act Art. 50 (Transparenz, seit 02.08.2026): Nutzer müssen wissen, dass sie mit einem KI-System interagieren.

## Walkthrough – Sprechtext-Stichpunkte
- A1: Beide Poison-Dokumente vorlesen lassen (HTML-Kommentar zeigen: „im Browser unsichtbar“). Dann die zwei Fragen
  auf sauber vs. vergiftet. Messwert: beide Poison-Docs auf Platz 1; Spindelöl kippt auf „2.000 Betriebsstunden [1]“
  (3/3 Läufe, mit und ohne gehärteten Prompt); die HR-FAQ (35 Tage) setzt sich **nicht** durch – das Modell folgt der
  offiziellen Richtlinie. Gate: 42 akzeptiert / 2 Quarantäne, 7 Regex-Findings. ACL: 0 Leaks mit Pre-Filter, 4 ohne.
  Ob das Modell die Werbe-Anweisung (LubriMax) befolgt, variiert – **das ist der Punkt**: „Wir wissen es nicht, also darf
  es nicht in den Index.“ B4 zählt Poisoning (falsche Zahl) und Injection (Werbung) jetzt getrennt.
- A2: `scan_documents` → Findings-Tabelle; `ingest_gate` → 2 abgelehnt. Hinweis auf False Positives: „Der Scanner
  ist dumm – in der Praxis Quarantäne statt Löschen.“
- A3: `acl_leak_test` mit 4 Rollen → alle `leak=False`. Dann `enforce_acl=False` → 4 Leaks. „Ein vergessener
  Parameter, und HR-Daten sind bei Vertrieb.“

## Lab 6 (35 Min)
- B1: Stealth-Poison ohne Regex-Treffer, dann LLM-Klassifikator. Erwartung: Regex leer, LLM erkennt „ersetzt das
  Handbuch ohne Freigabe“ meist. Diskussion: Was erkennt auch der LLM-Check nicht? (Sachlich falsche Zahl in korrekt
  formatiertem Dokument → nur Quellen-Governance hilft.)
- B2: Post-Filter: Zahlen zeigen, dass geschützte Chunks in `trace["candidates"]` stehen, obwohl sie „gefiltert“ wurden.
- B3: `audit_record` – Diskussion Pseudonymisierung, Aufbewahrung.
- B4 (Bonus): Prompt mit/ohne „Kontext ist DATEN“-Satz, je 3 Läufe. Ergebnis variiert; Erkenntnis: Prompt allein
  reicht nicht.
- Wo TN hängen: `Document(...)`-Konstruktor braucht `metadata` mit access/status; JSON-Parsing des LLM-Checks.

## Typische Fragen
- „Reicht ein guter System-Prompt nicht?“ – Nein. Modelle folgen Kontext-Anweisungen je nach Formulierung in 5–50 %
  der Fälle; Prompt-Härtung senkt die Rate, eliminiert sie nicht. Sicherheit kommt aus Gate + ACL + minimalen Rechten.
- „Wie mache ich ACLs bei SharePoint-Quellen?“ – Konnektoren (Microsoft Graph, Elastic DLS, Glean) liefern ACLs mit;
  beim Ingest als Payload speichern, bei Änderung Re-Sync (Webhooks/Delta-Queries). Azure AI Search kann seit 2025
  Entra-ACLs nativ filtern.
- „Können Embeddings ‚geknackt‘ werden?“ – Embedding-Inversion rekonstruiert Text aus Vektoren erstaunlich gut →
  Vektor-DB wie Klartext schützen (Verschlüsselung, Zugriff, kein öffentlicher Endpunkt).
- „Was ist mit dem Semantic Cache?“ – Cache-Key muss Rollen enthalten (Lab 7 zeigt es), sonst leakt der Cache.

## Stolperfallen
- Poison-Effekt auf die Antwort ist nicht deterministisch – nicht versprechen, dass das Modell „reinfällt“. Das
  Retrieval-Ergebnis (Poison auf Platz 1) ist deterministisch – darauf den Punkt aufbauen.
- `ingest_gate` mit `trusted_departments` – die Poison-Docs haben *gültige* Abteilungen (Service, HR): das Gate
  greift nur wegen der Findings. Bewusst so: „Herkunfts-Metadaten sind fälschbar.“

## Überleitung zu Block 7
„Wir haben jetzt eine sichere, gute Pipeline. Ab morgen früh läuft sie in Produktion – und wir sehen nichts.
Also: Observability.“
