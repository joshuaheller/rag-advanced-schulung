# Block 8 – Wrap-Up, Ausblick & Architektur-Review (16:00–16:45)

## Kernbotschaft
RAG ist kein Feature, sondern ein Wissenssystem mit Betrieb: Datenpflege, Eval, Security, Monitoring. Der Weg vom
PoC zur Enterprise-Lösung hat klare Stufen mit Exit-Kriterien.

## Inhalte

**RAG als Wissenssystem (Folie 8.1):** Ownership (wer pflegt Korpus, Golden Set, Prompts?), Datenqualität als
Dauerthema, Eval als Infrastruktur. „Das Modell ist der austauschbarste Teil.“

**RAG + Agenten (Folie 8.2):** Retrieval als *Tool* eines Agenten (der entscheidet, ob und wie oft er sucht),
Self-RAG/CRAG-Ideen (Modell bewertet Treffer, sucht nach, korrigiert), MCP als Schnittstelle zu Wissensquellen.
Wann Pipeline, wann Agent: Pipeline für definierte Q&A (planbar, testbar, günstig), Agent für Multi-Step-Recherche
und Tool-Kombination (teurer, schwerer zu evaluieren, OWASP Excessive Agency).

**Roadmap PoC → Enterprise (Folie 8.3):**
1. Baseline + Golden Set (Exit: Hit-Rate und Korrektheit gemessen)
2. Retrieval-Qualität: Parsing, Chunking, Hybrid, Rerank (Exit: Ziel-Korrektheit auf Golden Set)
3. Security: Gate, ACL, Prompt, Audit (Exit: Leak-Test grün, Pen-Test)
4. Observability + CI-Eval (Exit: Traces, Alerts, Regressions-Gate im PR)
5. Skalierung + Betrieb: Caching, Re-Index-Prozess, Feedback-Loop, Kostenmodell (Exit: Runbook, SLOs)

**Die 10 Entscheidungen eines RAG-Architekten (Folie 8.5):** Parser, Chunking-Strategie, Embedding-Modell,
Hybrid ja/nein, Reranker-Klasse, k/Kontextbudget, Metadaten-/ACL-Modell, Eval-Set und Judge, Observability-Stack,
Re-Index-Strategie.

## Architektur-Review (20 Min)
Ablauf:
1. Jeder TN skizziert in 8 Min (Whiteboard/Markdown/Zeichnung) die Ziel-Architektur für den *eigenen* Use Case
   (aus dem Level-Check Tag 1): Quellen, Ingest, Index, Retrieval, Security, Eval, Betrieb.
2. 5 Min pro TN: Vorstellung, du stellst Reviewer-Fragen (unten), der andere TN ergänzt.
3. 2 Min: Was ist der erste Schritt am Montag?

Reviewer-Fragen (auswählen):
- Woher kommen die Dokumente, wer darf sie ändern, wie merkt ihr Änderungen?
- Welche drei Fragetypen sind kritisch – und wie messt ihr die?
- Wie sieht ein Near-Miss in eurem Korpus aus? Was fangt ihr ihn ab?
- Wer darf was sehen? Wo im System wird das entschieden – vor oder nach der Suche?
- Was passiert, wenn das Embedding-Modell abgekündigt wird?
- Wie viele Anfragen pro Tag, was kostet eine, wo ist die Latenz?
- Welcher Alert weckt euch nachts?
- Was sagt ihr dem Datenschutzbeauftragten in einem Satz?

## Abschluss
- Folie 8.6 Ressourcen (Liste aus `00_Recherche_Quellen_und_Lizenzen.md`): OWASP LLM Top 10, Ragas/DeepEval-Docs,
  Anthropic Contextual Retrieval, Chroma Context Rot, PoisonedRAG, Hamel Husain Evals, Docling, Phoenix.
- Repo bleibt öffentlich; Musterlösungen in `labs/solutions/`.
- Dann die IT-Schulungen-Abschlussfolien (Zertifikat/Badge, nächster Schritt, Kontakt, Danke) und Hinweis auf das
  Feedback-Formular per E-Mail.
