# Block 0 – Basics-Recap & Level-Check (09:20–10:30, Lab 0: 15 Min)

## Kernbotschaft
RAG = Suchmaschine + LLM. Die Suchmaschine liefert Textstücke, das LLM formuliert daraus eine Antwort. Fast alle
Probleme produktiver Systeme entstehen, weil die Suchmaschine das Falsche liefert – nicht, weil das LLM dumm ist.

## Konzepte in Erklär-Sprache

**Warum RAG?** Ein LLM weiß nichts über eure Reisekostenrichtlinie. Statt es umzutrainieren (teuer, veraltet
sofort), geben wir ihm die relevanten Absätze zur Laufzeit in den Prompt. „Open-Book-Prüfung statt Auswendiglernen.“

**Die Pipeline (Folie 0.3):** Dokumente → in Stücke schneiden (Chunks) → jedes Stück in einen Zahlenvektor
übersetzen (Embedding) → in einer Vektor-DB speichern → Frage ebenfalls einbetten → ähnlichste Chunks holen
(Retrieval) → Chunks + Frage in den Prompt → Antwort. Zeig an der Folie mit dem Finger, wo Fehler entstehen können:
an *jedem* Pfeil.

**Embeddings:** „Ein Embedding ist ein Punkt im Raum, bei dem ähnliche Bedeutungen nah beieinander liegen.“
Cosine-Ähnlichkeit = Winkel zwischen zwei Punkten. Wichtig für den Kurs: Embeddings sind gut bei *Bedeutung*,
schlecht bei *exakten Bezeichnern* (AX-200 vs. AX-300 liegen fast am selben Punkt). Das ist die Saat für Block 2.

**Chunking:** Wie groß die Stücke sind, bestimmt, was gefunden werden kann. Zu klein → Kontext fehlt (Tabelle ohne
Überschrift). Zu groß → Rauschen, und die Antwort „versteckt“ sich im Chunk. Details Block 4.

**Vektor-DB (Qdrant):** speichert Vektor + Payload (Metadaten: Abteilung, Zugriff, Status). Filter auf den Payload
sind später (Block 6) die Grundlage für Zugriffsrechte. Wir nutzen Qdrant *im Python-Prozess* – „kein Server, aber
dieselbe API wie in Produktion“.

**Basis-Metriken:** Hit@k („war das richtige Dokument unter den Top-k?“), MRR („auf welchem Platz?“). Beide messen nur
das Retrieval – Überleitung auf Block 5: „Ob die Antwort richtig ist, sagen sie nicht.“

## Level-Check (Folie 0.2) – Fragen an die TN
1. Womit habt ihr RAG gebaut (LangChain, LlamaIndex, eigene Pipeline)? Welche Vektor-DB?
2. Wie messt ihr aktuell, ob das System gut ist?
3. Welcher Fehler ist euch zuletzt aufgefallen?
4. Läuft etwas produktiv? Wer nutzt es? Wer darf was sehen?
5. Was wollt ihr am Ende der zwei Tage können?

Antworten auf Whiteboard/Notiz – du kommst in Block 8 (Architektur-Review) darauf zurück. Wenn beide TN
sicher wirken: Folien 0.4–0.6 in 5 Min durchgehen, Zeit für Lab 1 gewinnen.

## Walkthrough (Teil A) – Sprechtext-Stichpunkte
- A1: „Erst schauen, dass alle denselben Stand haben.“ Modelle zeigen, Key vorhanden.
- A2: Korpus vorstellen: „Fiktiver Maschinenbauer, 42 Dokumente – aber mit den Fallen, die eure echten Korpora auch
  haben.“ Die 5 Fallen nennen (Near-Miss, Stale, ACL, Multi-Hop, Tabellen). PDF-Ausgabe zeigen: „Das ist, was pypdf
  aus einer Tabelle macht – jede Zelle eine Zeile, Zuordnung weg.“
- A3: Golden Set: „64 Fragen, wir wissen die richtige Antwort. Ohne so etwas ist jede Optimierung Raten.“
- A4: Baseline bauen, eine Frage stellen, `result.show()` → Treffer mit Scores, Zeiten, Tokens. „Das `trace` ist
  unser Röntgenbild – Block 1 lebt davon.“

## Lab 0 (15 Min)
- B1: drei Fragen (faktisch / near-miss / negativ). Erwartung: faktisch meist richtig, Near-Miss (g12 Kündigung
  Arbeitnehmer) oft falsch oder Arbeitgeber-Tabelle, Negativ-Frage manchmal Halluzination.
- B2: Hit-Rate@5 der Baseline. Erwartung mit echten Embeddings: ~0,6–0,75 gesamt, `tabelle` und `near-miss` schlecht.
- Wo TN hängen: `user_roles=["employee"]` vergessen → ACL-Fragen liefern Gehaltsbänder (guter Aufhänger für Block 6).

## Typische Fragen
- „Warum OpenAI und nicht lokal?“ – Für den Kurs am einfachsten; alles in `ragkurs/llm.py` ist austauschbar (Ollama via
  LangChain in 3 Zeilen). Reranker/Parsing sind bereits lokal.
- „Warum Qdrant?“ – Open Source, Hybrid + Filter nativ, lokaler Modus ohne Server. Konzepte gelten für alle DBs.
- „Ist text-embedding-3-small für Deutsch gut genug?“ – Solide; für sehr fachliche deutsche Korpora multilinguale
  Alternativen (bge-m3, e5-multilingual) benchmarken – mit dem Golden Set aus Block 5.

## Überleitung zu Block 1
„Wir haben jetzt eine Zahl: Hit-Rate X. Bevor wir irgendetwas verbessern, schauen wir uns an, *woran* das System
scheitert – sonst optimieren wir an der falschen Stelle.“
