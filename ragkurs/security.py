"""Security & Governance: Injection-Scan beim Ingest, Quellen-Gate, ACL-Leak-Test."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loading import Document

# Muster fuer (indirekte) Prompt Injection in Dokumenten - bewusst einfach, als erste Verteidigungslinie.
INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignorier\w*\s+(alle|saemtliche|sämtliche|die|alle anderen)\s+\w*", "Anweisung 'ignoriere ...'"),
    (r"\bignore\s+(all|previous|prior|the)\b", "Anweisung 'ignore ...' (EN)"),
    (r"\b(system|assistent|assistant|ki-assistent)\s*(anweisung|instruction)?\s*:", "Rollen-/Systemanweisung im Text"),
    (r"wenn\s+(du|der\s+assistent)\s+.*(gefragt|verwendest|nutzt).*antworte", "bedingte Antwortanweisung"),
    (r"(erwaehne|erwähne|verrate)\s+(diese|die)\s+anweisung\s+nicht", "Verschleierungsanweisung"),
    (r"<!--.*?-->", "versteckter HTML-Kommentar"),
    (r"https?://[^\s)]+", "URL im Dokument (Phishing-Risiko)"),
    (r"(passwort|password)\s+(eingeben|erforderlich|required)", "Aufforderung zur Passworteingabe"),
    (r"empfiehl\w*\s+.*\b(anbieter|produkt|shop)\b", "Produktempfehlung (Werbe-Injection)"),
]


@dataclass
class Finding:
    doc_id: str
    pattern: str
    snippet: str


def scan_text(doc_id: str, text: str) -> list[Finding]:
    out = []
    for pat, label in INJECTION_PATTERNS:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            snippet = re.sub(r"\s+", " ", m.group(0))[:120]
            out.append(Finding(doc_id, label, snippet))
    return out


def scan_documents(docs: list[Document]) -> list[Finding]:
    findings = []
    for d in docs:
        findings += scan_text(d.doc_id, d.text)
    return findings


def strip_hidden(text: str) -> str:
    """Entfernt HTML-Kommentare und Zero-Width-Zeichen (typische Verstecke fuer Injections)."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return re.sub(r"[​‌‍⁠﻿]", "", text)


@dataclass
class GateResult:
    accepted: list[Document] = field(default_factory=list)
    rejected: list[tuple[Document, str]] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"akzeptiert: {len(self.accepted)}  |  abgelehnt/quarantaene: {len(self.rejected)}"]
        for d, why in self.rejected:
            lines.append(f"  - {d.doc_id}: {why}")
        return "\n".join(lines)


def ingest_gate(
    docs: list[Document],
    trusted_departments: set[str] | None = None,
    require_metadata: tuple[str, ...] = ("doc_id", "department", "version", "access"),
    block_on_findings: bool = True,
    max_findings: int = 0,
) -> GateResult:
    """Quellen-Gate vor dem Indexieren: Herkunft, Pflicht-Metadaten, Injection-Scan.

    Produktionsmuster: Alles, was das Gate nicht besteht, geht in Quarantaene und wird von einem
    Menschen freigegeben - nicht stillschweigend verworfen.
    """
    res = GateResult()
    for d in docs:
        reasons = []
        missing = [k for k in require_metadata if not d.metadata.get(k)]
        if missing:
            reasons.append(f"fehlende Metadaten: {missing}")
        if trusted_departments is not None and d.metadata.get("department") not in trusted_departments:
            reasons.append(f"nicht vertrauenswuerdige Quelle: {d.metadata.get('department')}")
        findings = scan_text(d.doc_id, d.text)
        if block_on_findings and len(findings) > max_findings:
            reasons.append("Injection-Verdacht: " + "; ".join(f.pattern for f in findings[:3]))
        if reasons:
            res.rejected.append((d, " | ".join(reasons)))
        else:
            res.accepted.append(d)
    return res


def acl_leak_test(pipeline, golden: list[dict], roles_to_test: dict[str, list[str]]) -> list[dict]:
    """Prueft fuer ACL-Fragen, ob Rollen ohne Berechtigung an geschuetzte Dokumente kommen.

    roles_to_test: {"Mitarbeiter": ["employee"], "HR": ["hr"], ...}
    Rueckgabe: eine Zeile pro (Frage, Rolle) mit leak=True, wenn ein nicht erlaubtes Dokument im Kontext war.
    """
    rows = []
    for item in golden:
        allowed = set(item.get("allowed_roles", ["all"]))
        for label, roles in roles_to_test.items():
            res = pipeline.retrieve_only(item["question"], user_roles=roles)
            protected = [d for d in res.retrieved_doc_ids if d in item.get("source_docs", [])]
            permitted = "all" in allowed or bool(allowed & set(roles))
            rows.append(
                {
                    "id": item["id"],
                    "rolle": label,
                    "erlaubt": permitted,
                    "geschuetzte_docs_im_kontext": protected,
                    "leak": (not permitted) and bool(protected),
                }
            )
    return rows
