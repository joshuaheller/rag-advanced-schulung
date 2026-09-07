# Messwerte aus dem Testlauf (07.09.2026) – und was sie bedeuten

Quelle: `python scripts/run_all_labs.py`, echte Modelle (gpt-5.6-luna / gpt-5.6-terra / text-embedding-3-small), alle 8 Labs OK,
Gesamtlaufzeit 47 Min. Retrieval-Kennzahlen auf **n = 60** beantwortbaren Fragen (64 minus 2 `negativ` minus 2 `acl`;
ACL-Fragen werden seit diesem Lauf in `evaluate_retrieval`/`run_golden` übersprungen, weil sie für `employee` per Definition
unauffindbar sind – die Werte unten sind entsprechend umgerechnet). Query-Transformationen und Judge sind LLM-abhängig,
Abweichungen von ±1 Frage (≈ 0,017) zwischen Läufen sind normal.

**Die eine Botschaft für den ganzen Kurs:** Bei 42 Dokumenten ist die naive Baseline schon sehr gut (Hit@5 ≈ 0,98,
Korrektheit ≈ 0,95). Die Unterschiede zwischen den Techniken liegen bei 1–3 Fragen und zeigen sich fast nur in **p@1 / MRR**
(steht die richtige Quelle *oben*?), in **Latenz** und **Kosten**. Das ist kein Schönheitsfehler des Kurses, sondern die
Realität: Die Hebel wirken mit der Korpusgröße – und ohne Golden Set würde man keinen davon sehen. Nicht überverkaufen;
die Zahlen ehrlich zeigen und die Frage stellen: „Was passiert bei 4.000 Dokumenten?“

## Lab 0 / 1 – Baseline (fixed 800 → dense → k=5)

| Kennzahl | Wert | Kommentar |
|---|---|---|
| Hit@5 | 0,984 (59/60) | ein echter Retrieval-Miss |
| Recall | 0,975 | Multi-Hop: fast immer beide Quellen |
| MRR / p@1 | 0,950 / 0,917 | in jeder 12. Frage steht das Richtige nicht oben |
| Korrektheit (Judge, 19 Fragen) | 0,95–1,0 | 1,75 s/Frage, 0,00026 USD/Frage |
| Index | 124 Chunks, 2,8 s Aufbau | |

Seit diesem Lauf hat die Baseline `status_filter=None` (naiv: alle Fassungen im Index). Lab 1 läuft auf allen 62 Fragen
(~2 Min, ~0,02 USD). Erwartung: zusätzlich 1–3 Fehler bei den `stale`-Fragen (Reisekosten 2024/2025 statt 2026) – **vor dem
Kurs einmal Lab 0+1 neu laufen lassen** (`python scripts/run_all_labs.py 0 1`) und die tatsächlichen Fehler notieren.

g33 (Spindelöl AX-200): Baseline antwortet **richtig** (500 h). Top-5-Scores: Wartungsplan 0,693 · AX-200 0,688 · AX-300 0,653 –
der Abstand ist hauchdünn, das Modell pickt die richtige Zeile. Auf der Folie „Semantic Near-Misses“ steht deshalb nicht mehr
„Modell antwortet falsch“, sondern „richtig – aber nur, weil beide im Kontext sind“. Der echte Near-Miss zeigt sich in Lab 2 A2:
BM25 setzt AX-300 auf Platz 1 (weil im AX-300-Handbuch „gegenüber der AX-200“ steht).

## Lab 2 – Retrieval-Strategien

| Konfiguration | Hit@5 | Recall | MRR | p@1 | ms |
|---|---|---|---|---|---|
| dense | 0,984 | 0,939 | 0,946 | 0,917 | 4 |
| sparse / BM25 | 0,984 | 0,958 | 0,884 | **0,800** | 3 |
| hybrid / RRF (pf20) | 0,984 | 0,964 | 0,946 | 0,917 | 8 |
| hybrid / DBSF (pf10/20) | 1,000 | 0,981 | 0,952 | 0,917 | 9 |
| hybrid / DBSF (pf50) | 0,984 | 0,969 | 0,937 | 0,900 | 9 |

Query-Transformationen (Hybrid/RRF, pf20):

| Transform | Hit@5 | Recall | MRR | p@1 | ms |
|---|---|---|---|---|---|
| keine | 0,984 | 0,964 | 0,946 | 0,917 | 8 |
| rewrite | 1,000 | 0,978 | 0,953 | 0,917 | 135 |
| multi (3 Varianten) | 1,000 | 0,981 | 0,964 | 0,933 | 582 |
| decompose | 0,966 | 0,953 | 0,937 | 0,917 | 87 |
| hyde | 1,000 | 0,989 | 0,964 | 0,933 | 199 |

Lesart: BM25 allein ist beim Ranking klar schlechter (p@1 0,80), Hybrid = Dense beim Ranking, aber +Recall. DBSF minimal besser
als RRF (1 Frage). prefetch_k 10 vs. 20 vs. 50: kein Unterschied (bei 50 leicht schlechter). Query-Transformationen bringen
+1 Frage Hit / +1 Frage p@1 für das 15- bis 70-Fache an Latenz; **decompose schadet** (Teilfragen holen Nachbarn). Status-Filter:
ohne Filter stehen Reisekosten 2024 und 2025 *vor* 2026 (BM25). Contextual Retrieval auf HR-Docs (n=27): kein Effekt (0,963 → 0,963).

## Lab 3 – Reranking (Hybrid Top-20 → Top-5)

| Konfiguration | Hit@5 | MRR | p@1 | p50 / p95 ms | Laden |
|---|---|---|---|---|---|
| hybrid ohne Rerank | 0,984 | 0,946 | 0,917 | 9 | – |
| + fast (mmarco-MiniLM-L12) | 1,000 | 0,975 | 0,950 | 216 / 289 | 4,6 s |
| + quality (bge-reranker-v2-m3) | 1,000 | **0,992** | – | 2.321 / 3.848 | 6,9 s |
| + llm listwise (10 Fragen, pf10) | 1,000 | 1,000 | 1,000 | ~1.100 | – |
| colbert | – | – | – | – | Bibliothekskonflikt → wird übersprungen (Konzept auf Folie) |

prefetch_k-Sweep (fast): 5 → MRR 0,975 / 56 ms · 10 → **0,984** / 109 ms · 20 → 0,975 / 213 ms · 40 → 0,975 / 412 ms.
Kaskade (Top-40 → fast Top-10 → quality Top-5): MRR 0,992, p50 **1,9 s** vs. quality direkt auf 40: MRR 0,992, p50 **5,4 s**.

Lesart: Der Reranker verbessert **nur die Reihenfolge** (MRR 0,946 → 0,975 → 0,992), Hit@5 bleibt (per Definition). Sättigung
bei prefetch_k 10 – mehr Kandidaten kosten nur Latenz. Der große Reranker ist auf CPU ~10× langsamer; die Kaskade liefert dieselbe
Qualität in einem Drittel der Zeit. Lab 3 A2 (Kündigungsfrist Arbeitgeber, 9 Jahre): Cross-Encoder zieht den Arbeitgeber-Abschnitt
von Platz 2 auf Platz 1 (728 ms für 20 Kandidaten beim ersten Aufruf).

## Lab 4 – Context Engineering

Parsing, 17 Tabellenfragen (nur Retrieval): pypdf+fixed Hit 1,0 / MRR 1,0 · pypdf+headings 1,0 / 0,971 · docling+headings 1,0 / 1,0.
End-to-End (Judge) auf den 17 Tabellenfragen: **beide 1,0** (pypdf 1,27 s · docling 1,10 s). Docling-Parsing: 8 s für die 5 PDFs;
AX-200-Handbuch 2 Chunks (pypdf) vs. 6 Abschnitts-Chunks (docling).

**Ehrlich sagen:** Auf Dokument-Ebene findet auch pypdf die Tabellen (die Wörter sind ja da), und gpt-5.6 rekonstruiert zweispaltige
Tabellen aus dem Zeilensalat. Der Docling-Gewinn ist hier **Struktur** (Abschnitte, Breadcrumb, lesbarer Kontext, Zitierbarkeit) –
nicht Korrektheit. Bei 5-spaltigen Tabellen, Fußnoten, verbundenen Zellen sieht das anders aus. Nicht behaupten, pypdf sei
„schlecht auf Tabellenfragen“ – die Folie und das Lab sagen das inzwischen auch nicht mehr.

Chunk-Sweep (docling, by_headings): 400 → 261 Chunks, MRR 0,964, Ø 1.672 Zeichen Kontext · 800 → 240 / 0,956 / 1.816 ·
1500 → 238 / 0,964 / 1.826 · 3000 → 238 / 0,964 / 1.826. Hit@5 überall 1,0. Lesart: Abschnitte sind fast alle < 1.500 Zeichen,
deshalb ändert sich ab 1500 nichts mehr; 400 zerschneidet Abschnitte, ohne zu schaden – bei diesem Korpus. Kein Sweet Spot
sichtbar → „Chunk-Größe ist ein Messwert“ bleibt die Botschaft, nur eben: hier egal.

Breadcrumb-Ablation (13 Near-Miss-Fragen): **mit** MRR 0,910 / p@1 0,846 · **ohne** MRR 0,833 / p@1 0,692. Das ist der
deutlichste Einzeleffekt des ganzen Kurses (2 von 13 Fragen kippen auf Platz 1).

Parent-Child vs. flach (n=60): identisch (MRR 0,964 vs. 0,966). Lost-in-the-Middle, k=10, 14 Fragen mit Judge: **0,933 vs. 0,933** –
kein Effekt. Sagen: die U-Kurve stammt aus 2023-Modellen; aktuelle Modelle sind robuster; und 14 Fragen können einen Effekt
von 5 % gar nicht auflösen. Long-Context (ganzer Korpus, ~18.900 Input-Tokens) vs. RAG (k=5, ~800 Tokens), 10 Fragen: Korrektheit
**1,0 vs. 1,0**, Latenz 1,40 s vs. 1,38 s, Kosten **0,038 vs. 0,002 USD** (19×). Bei 10.000 Anfragen/Tag: 38 USD vs. 2 USD/Tag.

## Lab 5 – Evaluation

Baseline vs. Hybrid+Rerank (14 Fragen, Judge): Korrektheit 0,933 vs. 0,867 – d. h. die „bessere“ Pipeline verliert **eine** Frage
(g04, Multi-Hop: Context-Recall 0,25, Faithfulness 0). Genau das ist die Lektion: 1 Frage = 7 Prozentpunkte; n=14 ist kein Testset.
Metriken (Nachbau): Context Precision 0,87 / 0,90 · Context Recall 0,80 / 0,79 · Faithfulness 0,88 / 0,84 · Answer Relevancy
0,67 / 0,65 (Relevancy ist bei knappen Antworten wie „2 Arbeitstage“ systematisch niedrig – Embedding-Ähnlichkeit generierter
Rückfragen). Judge vs. 8 Trainer-Labels: Precision/Recall 1,0 (alle 8 korrekt – im Kurs die TN selbst labeln lassen, sonst ist es
trivial). Synthetische Fragen: 6 aus 8 Dokumenten in ~1 Min, brauchbar, teils Duplikate (9 € / 9,99 € Amtsträger) → Review-Pflicht.
pytest Ebene 1: 3 Tests in 6 s ohne Key. Regression-Gate im Lab: ref 0,87 → cand 0,93, PASS.

## Lab 6 – Security

44 Dokumente mit Poison. Beide Poison-Docs landen auf **Platz 1**; die Spindelöl-Antwort kippt auf „2.000 Betriebsstunden [1]“.
Die HR-FAQ (35 Urlaubstage) setzt sich **nicht** durch – das Modell folgt der offiziellen Richtlinie (30 Tage). Regex-Scan: 7 Findings,
Gate: 42 akzeptiert / 2 Quarantäne. ACL-Leak-Test 4 Rollen × 2 Fragen: 0 Leaks mit Pre-Filter; ohne (`enforce_acl=False`) 4 Leaks.
Post-Filter (B2): 2 von 64 Anfragen haben < 3 Treffer nach dem Filter, geschützte Chunks stehen trotzdem im Trace. Stealth-Poison:
Regex 0 Findings, LLM-Klassifikator erkennt es. Prompt-Härtung (B4, alt): 3/3 „befolgt“ mit **und ohne** Daten≠Anweisung-Satz –
die Zählung hat aber die falsche Zahl (2.000 h = Poisoning) mit der Werbe-Anweisung (LubriMax = Injection) vermischt; jetzt getrennt.
Botschaft: Gegen plausibel formulierte Falschinformation hilft kein Prompt, nur das Gate.

## Lab 7 – Betrieb

38 simulierte Anfragen (Hybrid + fast Rerank): p50 **1,39 s**, p95 **1,84 s**, p95 Retrieval+Rerank 384 ms, p95 Generierung 1,45 s,
No-Answer-Rate 5 %, Ø 1.074 Input-Tokens, **0,0003 USD/Anfrage**. Kostenmodell 10.000 Anfragen/Tag: k=3 → 1,63 USD · k=5 → 2,02 USD ·
k=10 → 3,36 USD; mit 30 % Cache-Hits 1,14 / 1,42 / 2,35 USD. Reranker: p50 124 ms → 1.220 CPU-s/Tag, ~1 Worker.
Semantic Cache mit Schwelle 0,92: **0 Hits** – Paraphrasen liegen bei 0,73–0,86 (text-embedding-3-small). Jetzt Schwelle 0,85 +
Fallfrage (Sonderurlaub darf kein Hit sein). Blue/Green: v1 (1500) MRR 0,975 vs. v2 (800) 0,964 → v2 nicht umschalten.
