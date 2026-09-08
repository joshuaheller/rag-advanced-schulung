# Block 3 – Reranking & Relevance Optimization (14:45–16:00, Lab 3: 30 Min)

## Kernbotschaft
Der Retriever sorgt für Recall (die richtige Quelle ist irgendwo in den Top-20), der Reranker für Precision (sie
steht auf Platz 1). Der Reranker repariert die Reihenfolge, nicht den Recall – was nicht in den Kandidaten ist,
kann er nicht nach vorne holen.

## Konzepte in Erklär-Sprache

**Bi-Encoder vs. Cross-Encoder (Folie 3.1):** Bi-Encoder = Frage und Chunk werden *getrennt* zu je einem Vektor
komprimiert, Vergleich per Winkel. Schnell (Chunks vorab berechenbar), aber grob – Details wie „durch den
Arbeitgeber“ vs. „durch den Arbeitnehmer“ gehen in der Komprimierung unter. Cross-Encoder = Frage **und** Chunk
zusammen in ein Modell, das direkt einen Relevanz-Score ausgibt. Präzise, aber pro Paar ein Modellaufruf → nur für
20–100 Kandidaten bezahlbar. Analogie: „Bi-Encoder ist das Inhaltsverzeichnis, Cross-Encoder liest die Seite.“

**Late Interaction / ColBERT (Folie 3.3):** Mittelweg. Für jeden *Token* ein Vektor (Chunk-Seite vorab berechenbar);
zur Laufzeit wird für jedes Query-Token der ähnlichste Chunk-Token gesucht (MaxSim) und aufsummiert. Fast so präzise
wie Cross-Encoder, viel schneller, aber mehr Speicher (Vektor pro Token). `answerai-colbert-small-v1` (33M Parameter)
ist primär englisch trainiert – auf Deutsch erwartbar schwächer; das *ist* ein Lernpunkt.

**LLM-Reranker (Folie 3.4):** Listwise: „Hier sind 10 Passagen, sortiere nach Relevanz.“ Stark bei komplexen
Kriterien („bevorzuge die aktuellste Version“), aber langsam und teuer. Sinnvoll bei geringer Anfragezahl oder als
Judge-Stufe.

**Trade-off-Matrix (Folie 3.5):** Qualität × Latenz × Kosten × Selbst-Hosting. Die Zahlen kommen aus Lab 3 B1 –
Folie mit Platzhaltern zeigen, im Lab füllen, danach in die Folie übernehmen (oder Screenshot).

**Kaskade (Folie 3.6):** Hybrid Top-100 → schneller Reranker Top-20 → starker Reranker Top-5. Budget pro Stufe.
„Jede Stufe darf teurer sein als die vorherige, weil sie weniger Kandidaten sieht.“

**Auswahlkriterien (Folie 3.7):** Sprache (multilingual!), Chunk-Länge vs. Modell-Fenster (mmarco: 512 Token, bge-v2-m3:
8k), Durchsatz, Datenschutz (API-Reranker sehen die Chunks!), Lizenz (jina-reranker-v2: CC BY-NC → nicht kommerziell).

## Walkthrough – Sprechtext-Stichpunkte
- A1: `get_reranker("fast")` – Ladezeit nennen (aus Cache Sekunden, sonst Download).
- A2: Kündigungsfrist-Frage (9 Jahre, Arbeitgeber). Vorher/Nachher-Liste zeigen: „Der Cross-Encoder zieht die
  Arbeitgeber-Tabelle nach vorn, weil er ‚kann Aurelia kündigen‘ und ‚Kündigung durch den Arbeitgeber‘ zusammen liest.“
  Messwert: Arbeitgeber-Abschnitt von Platz 2 auf Platz 1, 728 ms für 20 Kandidaten beim ersten Aufruf (danach ~200 ms).
- A3: `compare_retrieval` hybrid vs. hybrid+rerank. Messwert: Hit@5 0,984 → 1,0 (eine Frage rutscht von Rang 6 in die
  Top-5), **MRR 0,946 → 0,975, p@1 0,917 → 0,950**, 9 → 222 ms. Kernaussage: „Recall bleibt, Precision steigt.“
- A4: prefetch_k-Sweep – Messwert: 5 → MRR 0,975 / 56 ms; **10 → 0,984 / 109 ms**; 20 → 0,975 / 213 ms; 40 → 0,975 /
  412 ms. Sättigung bei 10, Latenz linear.

## Lab 3 (30 Min)
- B1: Trade-off-Tabelle. Messwert: fast MRR 0,975, p50 216 / p95 289 ms; **quality MRR 0,992, p50 2,3 s / p95 3,8 s**,
  Laden 7 s (aus dem Cache), ~3 GB RAM; **colbert wird mit Fehlermeldung übersprungen** (Bibliothekskonflikt
  rerankers/transformers 5 – erwartet, Konzept auf Folie 3.3 erklären). B4: LLM-Reranker 1,0 auf 10 Fragen, 1,1 s.
- B2: Kaskade vs. direkt. Messwert: gleiche Qualität (MRR 0,992), Kaskade **p50 1,9 s** vs. quality direkt **5,4 s**.
- B3: Auswahl für den eigenen Fall – 5 Min Markdown, dann jeder TN 1 Min vorstellen.
- B4 (Bonus): LLM-Reranker auf 10 Fragen – langsam (~2–4 s/Frage), gut. Diskussion: wann ok?
- Wo TN hängen: Reranker-Objekte sind gecacht (`get_reranker`), das große Modell nur einmal laden; `retrieval_row`
  braucht ein `RunResult` (Beispiel in der Lösung).

## Typische Fragen
- „Reranker auf 5 Kandidaten – warum bringt das nichts?“ – Er kann nur umsortieren, was da ist; bei 5 Kandidaten
  ist die richtige Quelle oft gar nicht dabei. Erst Recall (prefetch 20–50), dann Precision.
- „Kann der Reranker einen Score-Schwellwert liefern, um ‚keine Antwort‘ zu erkennen?“ – Ja, Cross-Encoder-Scores
  sind halbwegs kalibriert (bge: Sigmoid). Schwelle mit dem Golden Set (negativ-Fragen) bestimmen – Lab 7 B1 nutzt
  die Idee mit dem Top-Score.
- „Cohere Rerank 3.5?“ – Sehr gut, multilingual, aber API (Daten verlassen das Haus, Kosten pro 1k Suchen). Für
  on-prem: bge-reranker-v2-m3.
- „GPU?“ – Für < 10 Anfragen/s reicht CPU mit dem `fast`-Modell; bge-m3 in Produktion eher mit GPU oder als
  Kaskade dahinter.

## Stolperfallen
- Erster Aufruf von `quality` lädt 2,2 GB – falls Download nicht vorbereitet: B1 nur mit fast.
- Speicher: bge-m3 + Docling gleichzeitig in einem Kernel kann bei 8-GB-VMs eng werden → Kernel neu starten.

## Überleitung zu Block 4
„Reranker sortieren, was da ist. Was das Parsing zerstört oder das Chunking vom Produktnamen trennt, kann er nicht
nach vorn holen. Also: einen Schritt zurück in der Pipeline – Parsing, Struktur, Breadcrumb.“
