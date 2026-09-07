# Trainer-Briefing 00 – Vorbereitung und Gesamtablauf

## Was dieses Briefing ist

Pro Block gibt es eine Datei `0X_...md` mit: Kernbotschaft, Konzepte in Erklär-Sprache (inkl. Analogien),
Ablauf des Walkthroughs mit Sprechtext-Stichpunkten, Lab-Ablauf mit erwarteten Ergebnissen, typische TN-Fragen
mit Antworten, technische Stolperfallen, Überleitung. Die Briefings sind so geschrieben, dass du sie neben dem
Notebook offen hast und dich daran entlanghangelst.

## Checkliste vor dem Kurs

**Bis Fr, 11.09.**
- [ ] Repo gepusht, `bash setup.sh` auf eigenem Rechner komplett durchgelaufen (inkl. Modell-Download, ~3 GB)
- [ ] `.env` mit eigenem Key, `python smoke_test.py` → „Setup OK“ **mit** OpenAI-Aufruf
- [ ] **Alle 8 Lösungs-Notebooks einmal komplett durchlaufen lassen** (`labs/solutions/`, Kernel „Python (rag-schulung)“).
      Dauer ca. 60–90 Min inkl. Modell-Laden. Kosten < 5 USD. Dabei die Ergebnistabellen notieren – das sind deine
      Referenzwerte für „erwartete Ergebnisse“ (die Zahlen in den Briefings sind Erwartungswerte, keine Garantie)
- [ ] 3 OpenAI-Keys anlegen (je TN + Trainer), Limit 20 USD/Key, nach dem Kurs deaktivieren
- [ ] Installationsangaben an IT-Schulungen, Rückfrage: Sind die VMs vorbereitet (`setup.sh` gelaufen)?
- [ ] Folien im IT-Schulungen-Template, Logo oben rechts

**Am Vortag**
- [ ] Zoom-Link aus Expertes-Portal testen, Bildschirmfreigabe + Breakout-Raum
- [ ] Docling-Modelle und Reranker im Cache (`python scripts/download_models.py` → „Alle Modelle geladen“)
- [ ] `labs/solutions/*.ipynb` mit Outputs offen haben (Fallback, falls live etwas hakt)
- [ ] Zweiter Monitor: links Zoom + Folien, rechts JupyterLab

**Kursmorgen (08:30)**
- [ ] `jupyter lab` läuft, Lab 0 offen, `smoke_test.py` grün
- [ ] Keys für TN bereit (im Zoom-Chat privat schicken, nicht in den Gruppenchat)
- [ ] Phoenix testweise gestartet und wieder beendet (Port 6006 frei)

## Zeitplan (Kurzfassung)

| Tag 1 | | Tag 2 | |
|---|---|---|---|
| 09:00 | Begrüßung, Level-Check | 09:00 | Recap, Fragen |
| 09:20 | Block 0 Basics + Lab 0 | 09:15 | Block 4b Packing/Re-Ordering + Lab 4b |
| 10:30 | Pause | 10:00 | Pause |
| 10:45 | Block 1 Failure Analysis + Lab 1 | 10:15 | Block 5 Evaluation + Lab 5 |
| 12:00 | Mittag | 12:00 | Mittag |
| 13:00 | Block 2 Retrieval + Lab 2 | 13:00 | Block 6 Security + Lab 6 |
| 14:30 | Pause | 14:30 | Pause |
| 14:45 | Block 3 Reranking + Lab 3 | 14:45 | Block 7 Betrieb + Lab 7 |
| 16:00 | Block 4a Chunking + Lab 4a | 16:00 | Block 8 Wrap-Up + Architektur-Review |
| 16:50 | Wrap-up Tag 1 | 16:45 | Abschluss (Template-Folien) |

**Puffer:** Jeder Block hat 10–15 Min Luft. Wenn es eng wird, fallen zuerst die Bonus-Aufgaben (B4) weg, dann
der Debrief-Teil C (mündlich in 3 Min). Nie das Lab kürzen – das ist, was die TN mitnehmen.

## Didaktisches Muster pro Block

1. **Folien (10–20 Min):** Problem → Konzept → Trade-off → „so sieht das im Code aus“ (eine Folie mit dem
   ragkurs-Aufruf, der gleich kommt)
2. **Walkthrough (10–15 Min):** Teil A des Notebooks. Du führst aus, TN führen *mit* aus (Bildschirm teilen,
   Zelle für Zelle, Ergebnisse kommentieren: „Seht ihr, AX-300 statt AX-200 – das ist der Near-Miss“)
3. **Lab (20–45 Min):** Teil B. TN arbeiten allein (2–3 TN → kein Breakout nötig, Zoom-Hauptraum, Kamera an,
   du gehst reihum per Bildschirmfreigabe). Nach 60 % der Zeit: kurzer Zwischenstand
4. **Debrief (5 Min):** Teil C. Eine Frage reicht, wenn die Zeit knapp ist

## Umgang mit den zwei/drei TN

Bei so wenigen TN gilt: Vorwissen früh abfragen (Level-Check, Block 0) und Tempo anpassen. Erfahrene TN bekommen
die Bonus-Aufgaben; weniger erfahrene den Walkthrough noch einmal selbst. Die Musterlösungen (`labs/solutions/`)
kannst du bei Bedarf freigeben – sie sind im Repo, das ist kein Geheimnis.

## Wenn etwas technisch klemmt

| Symptom | Ursache | Lösung |
|---|---|---|
| `OPENAI_API_KEY fehlt` | `.env` nicht im Repo-Root oder Kernel nicht neu gestartet | `.env` prüfen, Kernel-Neustart |
| `reasoning_effort` Fehler | Modell kennt den Parameter nicht | in `.env` `OPENAI_REASONING_EFFORT=` leer setzen |
| Modell-Download hängt | Firewall/Proxy blockt huggingface.co | `download_models.py` erneut; Notfall: Reranker-Labs mit `fast` überspringen |
| Qdrant `already accessed by another instance` | `QDRANT_LOCATION` auf Pfad gesetzt und zwei Kernel offen | Standard `:memory:` lassen |
| Docling sehr langsam (>2 Min/PDF) | erste Konvertierung lädt Modelle | vorab `download_models.py`; sonst pypdf-Variante zeigen |
| Phoenix-Port belegt | alter Prozess | `start_phoenix(port=6007)` |
| Notebook „hängt“ bei `run_golden` | 50 Fragen × 2 LLM-Calls | Subset nehmen: `golden[:15]` – so ist es in den Labs vorgesehen |

## Kostenrahmen

Ein kompletter Durchlauf aller Lösungs-Notebooks mit `gpt-5.6-luna`/`terra`: ca. 2–4 USD. Pro TN und Kurstag < 5 USD.
Embedding-Cache (`.cache/`) sorgt dafür, dass wiederholte Index-Builds nichts kosten.
