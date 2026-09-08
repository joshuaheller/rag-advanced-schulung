# Block 1 – Advanced RAG Failure Analysis (10:45–12:00, Lab 1: 30 Min)

## Kernbotschaft
Erst klassifizieren, dann optimieren. Die eine Frage, die alles ordnet: **War die Evidenz im Kontext?** Nein →
Retrieval-Fehler (60–80 % aller Fehler in Produktion). Ja → Generierungsfehler.

## Konzepte in Erklär-Sprache

**Failure-Taxonomie (Folie 1.3)** – die sieben Klassen mit je einem Korpus-Beispiel:
- *Missing Evidence*: Quelle nicht in Top-k. Beispiel: Tabellenfrage, weil pypdf die Tabelle zerlegt hat.
- *Semantic Near-Miss*: ähnliches, falsches Dokument. AX-300 statt AX-200; Reisekosten 2024 statt 2026; Kündigung
  durch Arbeitgeber statt Arbeitnehmer. „Embeddings sehen die Bedeutung, nicht die Modellnummer.“
- *Partial Evidence*: Multi-Hop, nur eine von zwei Quellen (Urlaub: Richtlinie + BUrlG).
- *Context Noise*: richtige Quelle da, aber auf Platz 4 hinter drei irrelevanten Chunks → Modell übersieht sie.
  Chroma-Studie „Context Rot“ (2025): Leistung sinkt *nicht linear* mit der Kontextlänge – schon 10 irrelevante
  Chunks können die Antwortqualität messbar drücken.
- *Wrong Answer*: Quelle vorne, Antwort trotzdem falsch (Zahlendreher, falsche Zeile in der Tabelle).
- *Over-Refusal*: Evidenz da, Modell sagt „keine Information“ (zu vorsichtiger Prompt).
- *Halluzination statt Absage*: Frage unbeantwortbar (Jobticket), Modell erfindet etwas.

**LLM-as-a-Judge (Folie 1.7/5.6):** Ein zweites Modell vergleicht Antwort mit Referenz und sagt *korrekt / nicht
korrekt* plus Begründung. Warum binär? „Eine 7 von 10 kann niemand nachprüfen; ‚korrekt: nein, weil 120 statt 150 €‘
schon.“ Der Judge irrt auch – deshalb wird er in Block 5 gegen Menschen kalibriert.

**Debug-Workflow (Folie 1.8):** sammeln → klassifizieren → clustern (Fehlerklasse × Fragetyp) → Hebel wählen →
messen. Der **Hebel-Katalog** (Folie 1.9) ist die Landkarte der nächsten 1,5 Tage: Near-Miss → Hybrid/Filter/Rerank
(Block 2/3), Missing Evidence bei Tabellen → Parsing/Chunking (Block 4), Noise → Rerank/Re-Ordering (Block 3/4b),
Halluzination → Prompt/Eval (Block 5).

## Walkthrough – Sprechtext-Stichpunkte
- A1: g33 (Spindelöl AX-200). Messwert: Antwort **richtig** (500 h). Kandidaten: Wartungsplan 0,693 · AX-200 0,688 ·
  Wartungsplan **2024** 0,686 · AX-300 0,653 (400 h). Zeig die Scores: „Drei Hundertstel zwischen richtig, veraltet
  und falschem Modell. Das Modell rettet es, weil alle im Kontext stehen – bei k=3 oder engem Budget nicht mehr.“
  → Near-Miss ist ein Ranking-Problem; deshalb messen wir p@1, nicht nur Hit@5.
- A2: Judge aufrufen, JSON zeigen. „Begründung ist Pflicht, sonst kann man den Judge nicht debuggen.“
- A3: `run_golden` auf allen 62 Fragen (~3 Min, 0,02 USD). Während es läuft: erklären, was pro Zeile passiert
  (Retrieval, Antwort, Judge, Heuristik). Dann `failure_breakdown` → Kreuztabelle. „Das ist die Fehler-Landkarte.“
  Messwert: Korrektheit **0,903** (56/62), Hit@5 0,983, Refusal korrekt 3/3, 1,15 s/Frage. Die 6 Fehler:
  g19 stale → `over_refusal` (drei Reisekosten-Fassungen im Kontext, 4 vs. 5 vs. 6 Wochen – das Modell verweigert),
  g38 + g50 multi-hop → `wrong_answer` (Teilantwort: 2,5 % genannt, Eurobetrag fehlt), g39 tabelle →
  `partial_evidence` (AX-300-Handbuch statt AX-200), g52 near-miss → `semantic_near_miss` (AX-100 → Wartungsplan/AX-200),
  g57 near-miss → `wrong_answer` (Export-Lieferzeit, Kernaussage richtig, Details vermischt). Judge-Urteile sind
  streng-fair; je Lauf kann 1 Frage kippen (g19 war im Einzeltest richtig).

## Lab 1 (30 Min)
- B1: Heuristik nachprüfen – TN sollen mindestens einen Fall finden, wo die Heuristik falsch liegt (z. B. `context_noise`
  vergeben, obwohl es ein Zahlendreher war). Das ist gewollt: „Automatik schlägt vor, Mensch entscheidet.“
- B2: Kreuztabelle + Summe Retrieval vs. Generierung. Messwert: **4 Generierung, 2 Retrieval** – das Gegenteil der
  60–80-%-Regel. Erklärung liefern: kleiner Korpus (Retrieval findet fast alles), starkes Modell, aber Multi-Hop-
  Rechnungen und Versionskonflikte bleiben liegen. „Die Regel gilt für große Korpora; die Methode gilt immer.“
- B3: Priorisierung nach Schaden – Diskussion. Zielaussage: falsche Kündigungsfrist/Preis = hoher Schaden, verweigerte
  Antwort = niedriger Schaden. Metrik dafür in Produktion: Korrektheit auf Golden-Subset „kritische Fakten“, No-Answer-Rate.
- B4 (Bonus): k=20 statt 5 – Messwert: beide Retrieval-Fehler sind bei k=20 dabei (g39 Rang 2, g52 Rang 5) →
  Reranking (Block 3) ist der Hebel, nicht ein anderes Retrieval.
- Wo TN hängen: `df.loc[df.id == ..., "failure"] = ...` Syntax; `display()` gibt es nur im Notebook.

## Typische Fragen
- „Kann ich den Judge nicht einfach fragen, ob die Antwort richtig ist, ohne Referenz?“ – Ohne Referenz bewertet
  er nur Plausibilität (Faithfulness zum Kontext); ob der Kontext stimmt, weiß er nicht. Referenz = Golden Set.
- „Was, wenn das Golden Set selbst Fehler hat?“ – Passiert ständig. Jede Judge-Abweichung ist ein Review-Anlass für
  die Referenz. Golden Set ist Code: versioniert, reviewt.
- „Wie viele Fehler muss ich sammeln?“ – 30–50 reichen für erste Cluster; für Statistik (5 pp Unterschied sicher
  erkennen) ~100+ Fragen.

## Stolperfallen
- `run_golden` auf allen 64 Fragen dauert 3–5 Min – im Lab bei 20 bleiben.
- Judge-Modell `gpt-5.6-terra` ist teurer als der Generator; für den Kurs ok (< 0,10 USD pro Lauf).

## Überleitung zu Block 2
„Der größte Cluster bei fast allen Systemen: Near-Misses und fehlende Evidenz. Beides Retrieval. Also fangen wir
beim Retrieval an – mit der ältesten Technik der Suchmaschinen: BM25.“
