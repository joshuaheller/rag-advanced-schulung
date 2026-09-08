# Block 4 – Context Engineering & Knowledge Structuring (Tag 1 16:00–16:50 + Tag 2 09:15–10:00)

## Kernbotschaft
Die Qualität wird *vor* dem Embedding entschieden: Was das Parsing zerstört, findet kein Retrieval wieder. Und was
im Prompt an der falschen Stelle steht, liest das Modell nicht.

## Konzepte in Erklär-Sprache

**Parsing (Folie 4.1):** PDF ist ein Druckformat, keine Datenstruktur. pypdf liest Zeichen in Leserichtung – bei
Tabellen stehen Zellen untereinander, Spaltenzuordnung weg (Lab 0 gezeigt). Docling (IBM, MIT) erkennt Layout
(Überschriften, Absätze, Tabellen mit TableFormer) und gibt Markdown aus. Alternativen: Unstructured (`hi_res`),
Azure Document Intelligence, LlamaParse (API). „Erst parsen, dann chunken – niemals umgekehrt.“

**Struktur-erhaltendes Chunking (Folie 4.2):** Nach Überschriften schneiden, nicht nach Zeichenzahl. Jeder Chunk
bekommt einen **Breadcrumb** `[Betriebshandbuch AX-200 > 4. Wartungsintervalle]` – das ist der Grund, warum der
Chunk „weiß“, zu welchem Produkt er gehört (Near-Miss-Schutz!). Tabellen bleiben ganz (`_split_keep_tables`).

**Tabellen (Folie 4.3):** Markdown-Tabelle + Beschreibungssatz davor. Bei sehr großen Tabellen: Zeilen-Chunks mit
wiederholtem Header („Wartungsarbeit: Spindelöl wechseln | Intervall: 500 h“).

**Chunk-Größe (Folie 4.4):** Chroma-Studie: ~200–400 Token oft optimal für Recall, aber abhängig von Fragetyp. Zu
klein → Kontext fehlt, zu groß → Rauschen. Deshalb Sweep im Lab (B1) statt Daumenregel.

**Late Chunking (Folie 4.5):** Jina 2024: Erst das ganze Dokument durch den Transformer, dann pro Chunk poolen –
jeder Chunk-Vektor „kennt“ den Dokumentkontext. Braucht Long-Context-Embedding-Modelle; nur erwähnen.

**RAPTOR (Folie 4.6):** Baum aus Zusammenfassungen (Chunks → Cluster → Summary → …), alle Ebenen indexiert.
Gut für „Worum geht es in dem Bericht?“-Fragen. Nur erwähnen, nicht im Lab.

**Parent-Child (Folie 4.8):** Klein suchen (350 Zeichen treffen präzise), groß liefern (der ganze Abschnitt gibt
dem LLM den Kontext). In LangChain: `ParentDocumentRetriever`; in `ragkurs`: `chunk_parent_child` + `parent_expand=True`.

**Context Packing (Folie 4.9):** Token-Budget, Deduplikation (zwei Kinder desselben Elternteils = ein Kontext),
Quellen-Tags `[1] Quelle: …` – die Tags sind es, die Zitate ermöglichen.

**Lost in the Middle (Folie 4.10):** Liu et al. 2023: Modelle nutzen Anfang und Ende des Kontexts besser als die
Mitte (U-Kurve). Re-Ordering: Rang 1,3,5 vorne, 2,4 hinten. Effekt wächst mit k.

**Long-Context vs. RAG (Folie 4.11):** 1M-Token-Fenster machen „alles rein“ verlockend. Gegenargumente: Kosten pro
Anfrage (× Anfragen/Tag), Latenz, Context Rot, **Zugriffsrechte** (alles im Kontext = jeder sieht alles), keine
Zitierbarkeit. Sinnvoll: kleine Korpora (< 50 Dokumente), wenige Anfragen, oder Hybrid (Retrieval liefert 20 große
Abschnitte statt 5 kleine). Lab B5 rechnet es vor.

## Walkthrough Tag 1 – Sprechtext-Stichpunkte
- A1: pypdf vs. Docling am AX-200-Handbuch. Docling-Ausgabe zeigen: Markdown-Tabelle intakt. Dauer nennen (erstes
  PDF ~20–60 s auf CPU, danach schneller).
- A2: Chunk-Anzahl und Abschnittsnamen: mit pypdf ein Klumpen ohne Überschriften, mit Docling 6 saubere Abschnitte.
  Tabellen-Chunk mit Breadcrumb zeigen.
- A3: Drei Varianten auf den 17 Tabellenfragen. Messwert: Hit@5 **überall 1,0**, MRR 1,0 / 0,971 / 1,0. Ehrlich sagen:
  „Auf Dokument-Ebene findet auch pypdf die Tabelle – die Wörter sind ja da. Der Unterschied ist Struktur: 2 Chunks
  vs. 6 Abschnitte, Breadcrumb, lesbarer Kontext, Zitierbarkeit.“ Nicht behaupten, pypdf sei schlecht.

## Lab 4a (25 Min, Tag 1)
- B1: Chunk-Größen-Sweep. Messwert: 400 → 261 Chunks / MRR 0,964 / 1.672 Zeichen; 800 → 240 / 0,956 / 1.816;
  1500 → 238 / 0,964 / 1.826; 3000 → 238 / 0,964 / 1.826 (identisch – kein Abschnitt ist länger als 1.500). Hit@5
  überall 1,0. Befund: hier egal; Botschaft „Chunk-Größe ist ein Messwert“ trotzdem – bei langen Fließtexten anders.
- B2: End-to-End auf Tabellenfragen. Messwert: Korrektheit **1,0 vs. 1,0** (pypdf 1,27 s, Docling 1,10 s). gpt-5.6
  rekonstruiert zweispaltige Tabellen aus dem Zeilensalat. Diskussion: 5 Spalten, Fußnoten, verbundene Zellen, Scans.
- B3: Breadcrumb-Ablation auf 13 Near-Miss-Fragen. Messwert: **p@1 0,846 mit vs. 0,692 ohne**, MRR 0,910 vs. 0,833 –
  der deutlichste Einzeleffekt des Kurses. „Vier Zeichen ‚AX-200‘ im Chunk-Kopf machen den Unterschied.“
- Wo TN hängen: Docling-Ladezeit; `load_corpus(pdf_parser="docling")` einmal in Variable halten, nicht mehrfach aufrufen.

## Walkthrough Tag 2 – Sprechtext-Stichpunkte
- A4: Parent-Child: 215 Kinder, 145 Eltern. `retrieve_only` zeigt `source="parent"` + `child_id`. Vergleich flat vs.
  parent-child: Hit-Rate ähnlich, Kontext größer – „Effekt sieht man erst end-to-end“.
- A5: `build_context` mit Re-Ordering: Reihenfolge [1,3,5,4,2] zeigen.

## Lab 4b (20 Min, Tag 2)
- A4: Parent-Child (316 Kinder, 238 Eltern). Messwert: identisch mit flach (MRR 0,966 vs. 0,964) – Nutzen ist
  Kontextqualität für die Antwort, nicht Retrieval.
- B4: Re-Ordering bei k=10, 14 Fragen (~1 Min). Messwert: **0,933 vs. 0,933 – kein Effekt**. Zwei Gründe nennen:
  aktuelle Modelle sind robuster als die 2023er der Studie; 14 Fragen können 5 % nicht auflösen. Billig, schadet
  nicht, kein Wundermittel.
- B5: Long-Context: Korpus komplett (~19k Token) in den Prompt, 10 Fragen. Messwert: Korrektheit **1,0 vs. 1,0**,
  Latenz 1,40 vs. 1,38 s, **Kosten 0,038 vs. 0,002 USD (19×)** → bei 10k Anfragen/Tag 38 vs. 2 USD. Diskussion:
  „Und wer darf die Gehaltsbänder sehen?“

## Typische Fragen
- „Docling in Produktion – zu langsam?“ – Ingest ist Batch, nicht Anfragezeit. 1 PDF/Minute/CPU-Kern ist ok für
  nächtliche Läufe; GPU beschleunigt 10×. Inkrementell nur geänderte Dokumente.
- „Was ist mit Bildern/Diagrammen?“ – Docling extrahiert Bilder; Beschreibung per Vision-Modell als Text-Chunk
  (Multimodal-RAG) – außerhalb des Kurses, Konzept gleich.
- „Semantic Chunking (Embedding-basiert)?“ – Splittet, wo sich die Bedeutung ändert. Teurer, oft nicht besser als
  Überschriften-Chunking bei strukturierten Dokumenten (Chroma-Studie). Bei unstrukturiertem Fließtext nützlich.
- „Ist Parent-Child nicht dasselbe wie größere Chunks?“ – Nein: Suche bleibt präzise (kleiner Vektor), nur der
  gelieferte Kontext wächst. Größere Chunks verschlechtern die Suche.

## Stolperfallen
- Docling-Modelle müssen vorab geladen sein (`download_models.py`). Falls nicht: Lab 4 mit `corpus_src/*.md`
  (Markdown-Originale der PDFs) fahren – Effekt ist dann nur konzeptionell.
- Speicher: Docling + bge-m3 in einem Kernel → 8-GB-VM knapp.

## Überleitung zu Block 5
„Wir haben jetzt fünf Hebel bewegt und jedes Mal Hit-Rate und MRR angeschaut. Aber Hit-Rate sagt nichts darüber, ob
die *Antwort* stimmt. Zeit für richtige Evaluation.“
