#!/usr/bin/env python3
"""Balanced Scholarly Citation Quality (BSC) evaluator.

Builds citation evidence JSONL from generated survey markdown and arXiv
reference mappings, then optionally asks an LLM judge to score citation quality.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DIMENSIONS = [
    "Claim-Level Citation Support",
    "Reference Faithfulness and Attribution Accuracy",
    "Multi-Reference Synthesis Coverage and Quality",
    "Citation Distribution Balance and Non-Redundancy",
]

BSC_JUDGE_PROMPT = """You are an expert reviewer evaluating citation quality in an academic survey paper.

You will be given a JSONL input for one generated survey. Evaluate only the citation quality of the survey based on the provided JSONL. Do not use external knowledge, web search, or your own memory of the cited papers.

The JSONL input contains:
1. survey_meta: global citation and sampling statistics.
2. evidence_card_bank: deduplicated evidence cards for all papers used by the selected claim records.
3. bsc_claim_context records: cited claim contexts and their full citation groups.
4. balance_summary: evidence-backed sample statistics plus global_citation_distribution, which is computed from all supported citation markers in the manuscript body independently of ref.json, year filtering, metadata availability, and evidence-card availability.

Each evidence_card may contain title, abstract, task category, keywords, method summary, innovations, datasets, results, motivation, key insights, conclusion, and limitations. Treat the evidence_card as the only authoritative evidence for the cited paper.

Important rules:
- [CITE_THIS] marks the citation or citation group currently being evaluated.
- For a single-citation claim, judge whether that one paper supports the claim.
- For a multi-citation claim, judge whether the citation group jointly supports the claim, and whether any cited paper is clearly irrelevant or unsupported.
- Do not require every paper in a multi-citation group to independently support the whole sentence, but each cited paper should make a meaningful, non-empty, and non-misleading contribution.
- Evaluate citation groups according to the type of claim: paper-specific factual claims may be supported by a single paper, while broader survey-level claims should be supported by appropriately related evidence.
- If the abstract is empty, use any remaining substantive metadata fields. Missing evaluation evidence is neither support nor contradiction: mark the record unassessable for semantic support instead of treating the manuscript citation as wrong.
- For specific claims about numbers, datasets, models, mechanisms, experimental results, or limitations, the relevant evidence should appear in the evidence_card.
- Do not reward citations merely because paper titles or keywords are topically similar.
- Do not reward large citation groups if they are mechanically stacked or only weakly related.

QWEN CALIBRATION VERSION: 20260726-v3-missing-evidence-neutral

EVIDENCE-GATED SCORE ANCHORS
Use the full 1-5 scale. Score demonstrated citation quality in assessable records and structural citation behavior in the manuscript. Do not convert missing evaluator-side metadata into a manuscript defect.
- 5 = Exceptional and close to publication-ready for this dimension. Numerous assessable records show consistently strong performance with only negligible exceptions.
- 4 = Strong. The assessable evidence is reliable and well handled across distinct records or sections, with only localized weaknesses.
- 3 = Competent but materially imperfect, or neutral/indeterminate when too little semantic evidence is available for a confident quality judgment.
- 2 = Weak, supported by positive contrary evidence of serious or widespread manuscript defects.
- 1 = Failed, supported by positive contrary evidence that the manuscript's citations are predominantly misleading or unusable.

MISSING-EVIDENCE NEUTRALITY
- Separate manuscript quality from evaluation coverage. A title-only or missing-abstract evidence card means "not assessable for semantic support"; it does not mean unsupported, irrelevant, unfaithful, fabricated, or misleading.
- For Claim-Level Citation Support and Reference Faithfulness, exclude title-only/unassessable records from both the positive and negative denominators. Never assign 1 or 2 merely because many records are title-only.
- If fewer than 20 claim contexts have substantive abstract/metadata evidence, use 3 as the neutral score unless the assessable subset contains clear positive contrary evidence of manuscript defects. A score of 4 is allowed only when at least 10 assessable contexts from several sections are consistently strong; a 5 requires broad assessable coverage and is unavailable when evidence coverage is materially incomplete.
- Evidence insufficiency must be reported only in the evidence_insufficiency diagnostic and overall assessment. It must not be relabeled as unsupported_claims, attribution_errors, weak synthesis, or citation imbalance.
- For Multi-Reference Synthesis, judge the explicit relationship expressed in the claim text: shared mechanism, contrast, lineage, boundary, trade-off, trend, or limitation. Evidence cards verify whether assessable group members fit that relationship. Unassessable members limit confidence and may prevent a 5, but do not by themselves turn synthesis into mechanical stacking.
- Citation Distribution Balance is fully independent of evidence-card completeness and must use only global_citation_distribution.

Do not begin from a presumed defect. Identify assessable records, positive evidence, verified contrary evidence, and unassessable records separately. Scores of 1-2 require verified contrary evidence; uncertainty alone yields a neutral 3 rather than a punitive score.

MANDATORY CITATION AUDIT
For each dimension, identify the strongest supporting evidence and the strongest contrary evidence before scoring. Cite representative claim_id values, paper identifiers, or named balance_summary fields in every rationale. Inspect both single-citation and multi-citation contexts when applicable. Do not use title overlap as a substitute for evidence-card support.

HARD SCORE CAPS
Apply the lowest relevant cap only when supported by assessable positive contrary evidence. Evidence insufficiency never activates a manuscript-quality cap. A verified severe diagnostic normally caps its corresponding dimension at 2; a verified moderate recurring diagnostic caps it at 3; a verified mild recurring diagnostic caps it at 4.
- Claim-Level Citation Support: recurring over-specific claims, unsupported group members, or claims that exceed the supplied evidence cap the score at 3; widespread cases cap it at 2.
- Reference Faithfulness: recurring conflation of methods, datasets, results, limitations, or paper roles caps the score at 3; systematic misattribution caps it at 2.
- Multi-Reference Synthesis: if most inspected multi-citation contexts merely stack or juxtapose papers without a shared comparison, boundary, trend, or trade-off, the score is at most 3; if meaningful synthesis is sparse, it is at most 2. A high multi-citation ratio alone is never positive evidence.
- Citation Balance: recurring exact-group reuse, strong concentration, large mechanical stacks, or weak coverage across many relevant sections caps the score at 3; a severe manuscript-wide pattern caps it at 2.

Do not force differences between manuscripts and do not tune scores to an expected method ranking. When semantic evidence is insufficient, use the neutral score 3 and state the limitation; do not infer failure. Use 4 only under the assessable-subset rule above and never guess 5.

Evaluate the following four BSC dimensions. Each dimension should receive an integer score from 1 to 5.

1. Claim-Level Citation Support
Evaluate whether each cited claim is directly or reasonably supported by its citation group. For single-citation claims, assess the cited paper alone. For multi-citation claims, assess whether the cited papers jointly support the claim and whether the group contains clearly unsupported citations.
Scoring guide:
5: Almost all cited claims are well supported by their citation groups, and multi-citation groups are selective and justified.
4: Most cited claims are supported, with only minor over-specific or weakly supported citations.
3: Mixed quality; broad claims are often supported, but many specific claims exceed the evidence or include weak group members.
2: Many claims are only weakly supported, or multi-citation groups often contain irrelevant or unsupported papers.
1: Most cited claims are unsupported, misleading, or poorly grounded in the cited evidence.

2. Reference Faithfulness and Attribution Accuracy
Evaluate whether the survey accurately represents the cited papers' paper-specific content, including their tasks, methods, contributions, datasets, experimental results, and limitations. This dimension should focus on attribution accuracy for individual cited papers, not on whether a broad synthesized claim is fully supported as a whole. If a broad or abstract synthesis is only partially supported, penalize it primarily under Claim-Level Citation Support or Multi-Reference Synthesis Coverage and Quality. Penalize Reference Faithfulness only when the survey assigns unsupported details, mechanisms, results, limitations, or roles to specific cited papers, or when compression across papers creates a misleading paper-level attribution.

Scoring guide:
5: Paper-specific representations and attributions are accurate, specific, and not misleading.
4: Mostly faithful, with minor harmless compression, omission, or paraphrasing.
3: Partially faithful, with some exaggeration, conflation across papers, or imprecise attribution of paper-specific details.
2: Frequent misrepresentation of tasks, methods, datasets, results, limitations, or paper-specific contributions.
1: Systematic misrepresentation of cited papers.

3. Multi-Reference Synthesis Coverage and Quality
Evaluate whether the survey appropriately uses multi-reference synthesis to support academic survey writing. This dimension considers both how well multi-citation claims integrate related papers and whether such synthesis is sufficiently represented across the evaluated claim contexts and citation distribution statistics.

Assess:
(a) synthesis coverage: whether multi-citation claims are represented across relevant sections and scholarly functions, such as trends, taxonomies, method families, benchmark patterns, shared limitations, comparisons, trade-offs, or future directions;
(b) synthesis quality: whether the cited papers are integrated into clear, accurate, and bounded scholarly claims rather than mechanically stacked, loosely juxtaposed, or over-abstracted.

Scoring guide:
5: The survey frequently and appropriately synthesizes multiple related papers across sections, with clear commonalities, differences, boundaries, or trade-offs.
4: The survey contains substantial multi-reference synthesis across several sections, with only minor looseness or overgeneralization.
3: Some synthesis is present, but coverage is limited or many multi-citation claims remain broad, list-like, or insufficiently bounded.
2: Multi-reference synthesis is sparse, weakly integrated, mechanically assembled, or mostly used for vague generalizations.
1: The survey provides little meaningful multi-reference synthesis and largely relies on isolated citation support.

4. Citation Distribution Balance and Non-Redundancy
Evaluate only the structural distribution, reuse, and grouping patterns of citations across the full manuscript. For this dimension, use only balance_summary.global_citation_distribution, which covers all supported citation markers parsed from the Markdown body and is independent of ref.json and evidence-card availability.

Assess section citation coverage, concentration on a small number of citation indices, repeated reuse of the same citations or exact citation groups, citation-group size patterns, and visible signs of mechanical citation stacking. Use section titles only to assess distribution across the manuscript's own research directions.

Do not evaluate whether a cited paper supports a claim, whether it is topically relevant, whether it is represented faithfully, or whether a multi-reference claim forms a high-quality synthesis; those questions belong to Dimensions 1-3. Do not use evidence_card_bank, evidence-backed selected-paper counts, or arXiv/evidence coverage to lower this score. Do not assume that multi-citation is inherently better than single-citation, and do not determine the score from the absolute number of references alone.

Scoring guide:
5: Citations are broadly and appropriately distributed across the manuscript's relevant sections, with low concentration, little redundant reuse, and no strong structural signs of mechanical stacking.
4: Citation distribution is generally balanced and non-redundant, with only mild section unevenness, concentration, repeated groups, or occasional large stacks.
3: Moderate imbalance or redundancy is visible, such as uneven section coverage, noticeable concentration on a few citation indices, repeated exact groups, or recurring large citation stacks.
2: Serious structural imbalance or redundancy, including weak coverage of many relevant sections, strong dependence on a small set of citation indices, frequent repeated groups, or persistent mechanical stacking.
1: Citation usage is severely concentrated, highly repetitive, mechanically stacked, or absent from most relevant sections of the manuscript.

Output requirements:
Return only valid JSON. Do not output markdown or extra explanation.
Each rationale should be concise but evidence-bearing, normally 2-4 sentences. It must name representative record identifiers or aggregate fields, state the strongest positive evidence, state the most important limitation, and explain any applied cap.
Scores must be integers from 1 to 5.
bsc_raw_20 must equal the sum of the four dimension scores.
Do not output a 100-point score.
diagnostics is only for explanation and does not affect the four scores. Each diagnostic field must be one of: none, mild, moderate, severe.

Return JSON schema:
{
  "BSC": {
    "Claim-Level Citation Support": {"score": 0, "rationale": ""},
    "Reference Faithfulness and Attribution Accuracy": {"score": 0, "rationale": ""},
    "Multi-Reference Synthesis Coverage and Quality": {"score": 0, "rationale": ""},
    "Citation Distribution Balance and Non-Redundancy": {"score": 0, "rationale": ""}
  },
  "bsc_raw_20": 0,
  "diagnostics": {
    "unsupported_claims": "none | mild | moderate | severe",
    "attribution_errors": "none | mild | moderate | severe",
    "weak_multi_reference_synthesis": "none | mild | moderate | severe",
    "insufficient_multi_reference_coverage": "none | mild | moderate | severe",
    "citation_stacking": "none | mild | moderate | severe",
    "citation_imbalance": "none | mild | moderate | severe",
    "evidence_insufficiency": "none | mild | moderate | severe"
  },
  "overall_assessment": ""
}

JSONL input:
"""


class BSCError(RuntimeError):
    pass


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_log(log_path: Path, lines: Sequence[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        for line in lines:
            f.write(str(line).rstrip() + "\n")


def normalize_arxiv_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("arXiv:", "").replace("arxiv:", "")
    s = s.split("/")[-1] if "/abs/" in s or "/pdf/" in s else s
    s = re.sub(r"\.pdf$", "", s, flags=re.I)
    s = re.sub(r"v\d+$", "", s, flags=re.I)
    m = re.search(r"(\d{4}\.\d{4,5})", s)
    if m:
        return m.group(1)
    digits = re.sub(r"\D", "", s)
    if len(digits) in (8, 9):
        return f"{digits[:4]}.{digits[4:]}"
    return None


def arxiv_year_month(arxiv_id: str) -> Tuple[int, str]:
    m = re.fullmatch(r"(\d{2})(\d{2})\.(\d{4,5})", arxiv_id)
    if not m:
        raise ValueError(f"Unsupported arXiv ID: {arxiv_id}")
    yy = int(m.group(1))
    year = 2000 + yy if yy <= 99 else yy
    month = m.group(2)
    return year, month


def load_ref_map(path: Path) -> Dict[str, str]:
    raw = json.loads(read_text(path))
    ref_map: Dict[str, str] = {}
    for key, value in raw.items():
        citation_index = str(key).strip()
        arxiv_id = normalize_arxiv_id(value)
        if citation_index and arxiv_id:
            ref_map[citation_index] = arxiv_id
    return ref_map


def filter_ref_map_by_year(
    ref_map: Dict[str, str], min_year: int = 2020, max_year: int = 2026
) -> Tuple[Dict[str, str], Dict[str, int]]:
    filtered: Dict[str, str] = {}
    stats = {"invalid_arxiv_id": 0, "outside_year_range": 0}
    for idx, paper_id in ref_map.items():
        try:
            year, _ = arxiv_year_month(paper_id)
        except ValueError:
            stats["invalid_arxiv_id"] += 1
            continue
        if min_year <= year <= max_year:
            filtered[idx] = paper_id
        else:
            stats["outside_year_range"] += 1
    return filtered, stats


def month_dir_for(arxiv_id: str, root: Path) -> Path:
    year, month = arxiv_year_month(arxiv_id)
    return root / f"{year}_new" / f"{year}-{month}"


def find_metadata_path(arxiv_id: str, metadata_root: Path) -> Optional[Path]:
    direct_candidates = [
        metadata_root / f"{arxiv_id}.json",
        metadata_root / f"{arxiv_id}_processed.json",
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    flat_matches = sorted(metadata_root.glob(f"*_{arxiv_id}.json"))
    if flat_matches:
        return flat_matches[0]
    topic_matches = sorted(metadata_root.glob(f"*/*_{arxiv_id}.json"))
    if topic_matches:
        return topic_matches[0]

    month_dir = month_dir_for(arxiv_id, metadata_root)
    if not month_dir.is_dir():
        return None
    matches = sorted(month_dir.glob(f"*{arxiv_id}*_processed.json"))
    if matches:
        return matches[0]
    compact = arxiv_id.replace(".", "")
    matches = sorted(month_dir.glob(f"*{compact}*_processed.json"))
    return matches[0] if matches else None


def load_metadata(arxiv_id: str, metadata_root: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    metadata_path = find_metadata_path(arxiv_id, metadata_root)
    if not metadata_path:
        return None, None
    try:
        return json.loads(read_text(metadata_path)), metadata_path
    except Exception:
        return None, metadata_path


def nested_get(data: Dict[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def as_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def candidate_base_names(arxiv_id: str, metadata_path: Optional[Path]) -> List[str]:
    bases: List[str] = []
    if metadata_path:
        stem = metadata_path.name
        stem = re.sub(r"_processed\.json$", "", stem)
        stem = re.sub(r"\.json$", "", stem)
        if stem:
            bases.append(stem)
    suffix = arxiv_id.split(".")[-1]
    bases.append(f"{suffix}_{arxiv_id}")
    bases.append(arxiv_id)
    seen = set()
    out = []
    for base in bases:
        if base not in seen:
            out.append(base)
            seen.add(base)
    return out


def resolve_mineru_md_path(
    arxiv_id: str,
    metadata_path: Optional[Path],
    metadata: Dict[str, Any],
    mineru_root: Path,
) -> Optional[Path]:
    file_path = nested_get(metadata, ["basic_info", "file_path"], "")
    if isinstance(file_path, str) and file_path:
        p = Path(file_path)
        if p.is_file() and p.suffix.lower() == ".md":
            return p
        if p.is_dir():
            matches = sorted(p.glob("*.md"))
            if matches:
                return matches[0]

    month_dir = month_dir_for(arxiv_id, mineru_root)
    if not month_dir.is_dir():
        return None

    for base in candidate_base_names(arxiv_id, metadata_path):
        candidates = [
            month_dir / base / "auto" / f"{base}.md",
            month_dir / base / base / "auto" / f"{base}.md",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    for d in sorted(month_dir.glob(f"*{arxiv_id}*")):
        if not d.is_dir():
            continue
        for candidate in [
            d / "auto" / f"{d.name}.md",
            d / d.name / "auto" / f"{d.name}.md",
        ]:
            if candidate.is_file():
                return candidate
        nested = sorted(d.glob("*/auto/*.md"))
        if nested:
            return nested[0]
        direct = sorted((d / "auto").glob("*.md")) if (d / "auto").is_dir() else []
        if direct:
            return direct[0]
    return None


def clean_abstract_text(text: str, max_chars: int) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.count("|") >= 2:
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].strip()


def is_abstract_marker(line: str) -> Optional[str]:
    s = line.strip()
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"^\d+(?:\.\d+)*\s+", "", s)
    s = s.strip(" *:_-\t")
    compact = re.sub(r"\s+", "", s).lower()
    if compact == "abstract":
        return ""
    m = re.match(r"^(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)\s*[:.-]\s*(.+)$", s, flags=re.I)
    if m:
        return m.group(1).strip()
    return None


def is_abstract_stop(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"^\d+(?:\.\d+)*\s+", "", s)
    s = s.strip(" *:_-\t").lower()
    if not s:
        return False
    stops = [
        "introduction",
        "keywords",
        "key words",
        "index terms",
        "background",
        "related work",
        "references",
        "acknowledg",
        "method",
    ]
    return any(s == stop or s.startswith(stop + ":") for stop in stops)


def extract_abstract_from_md(md_text: str, max_chars: int) -> str:
    lines = md_text.splitlines()
    start_idx: Optional[int] = None
    first_tail = ""
    for i, line in enumerate(lines[:250]):
        marker_tail = is_abstract_marker(line)
        if marker_tail is not None:
            start_idx = i + 1
            first_tail = marker_tail
            break
    if start_idx is None:
        return ""

    collected: List[str] = []
    if first_tail:
        collected.append(first_tail)
    for line in lines[start_idx : min(len(lines), start_idx + 160)]:
        if is_abstract_stop(line):
            break
        if re.match(r"^#{1,6}\s+", line.strip()) and collected:
            break
        collected.append(line)
    abstract = clean_abstract_text("\n".join(collected), max_chars)
    if len(abstract) < 30:
        return ""
    return abstract


def extract_title_from_md(md_text: str, max_chars: int = 300) -> str:
    for line in md_text.splitlines()[:80]:
        stripped = line.strip()
        if not stripped:
            continue
        heading = re.match(r"^#{1,3}\s+(.+?)\s*#*\s*$", stripped)
        if not heading:
            continue
        title = clean_sentence(heading.group(1), max_chars)
        normalized = re.sub(r"\s+", "", title).lower()
        if normalized not in {"abstract", "keywords", "introduction"}:
            return title
    return ""


def evidence_tier_name(card: Dict[str, Any]) -> str:
    has_metadata = bool(card.get("_has_metadata"))
    has_abstract = bool(card.get("_has_abstract"))
    has_title = bool(card.get("_has_title"))
    if has_metadata and has_abstract:
        return "metadata_and_abstract"
    if has_metadata:
        return "metadata_only"
    if has_abstract:
        return "abstract_only"
    if has_title:
        return "title_only"
    return "insufficient"


def evidence_tier_rank(card: Dict[str, Any]) -> int:
    order = {
        "metadata_and_abstract": 0,
        "metadata_only": 1,
        "abstract_only": 2,
        "title_only": 3,
    }
    return order.get(evidence_tier_name(card), 4)


EVIDENCE_ABSTRACT_MAX_CHARS = 900
EVIDENCE_METHOD_SUMMARY_MAX_CHARS = 600
EVIDENCE_KEY_INSIGHTS_MAX_CHARS = 450
EVIDENCE_CONCLUSION_MAX_CHARS = 350
EVIDENCE_LIMITATIONS_MAX_CHARS = 350
EVIDENCE_TITLE_MAX_CHARS = 220
EVIDENCE_TASK_CATEGORY_MAX_CHARS = 120
EVIDENCE_METHOD_NAME_MAX_CHARS = 160


def truncate_text(value: Any, max_chars: int) -> str:
    return clean_sentence(str(value or ""), max_chars)


def truncate_list_items(value: Any, max_items: int = 3, max_item_chars: int = 220) -> List[Any]:
    if value is None or value == "":
        items: List[Any] = []
    elif isinstance(value, list):
        items = value
    else:
        items = [value]
    out: List[Any] = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            compact_item: Dict[str, Any] = {}
            for key in ["dataset", "metric", "value", "comparison"]:
                if key in item and item[key] not in (None, "", []):
                    compact_item[key] = truncate_text(item[key], max_item_chars)
            if compact_item:
                out.append(compact_item)
        else:
            text = truncate_text(item, max_item_chars)
            if text:
                out.append(text)
    return out


def compact_public_evidence_card(card: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "paper_id": truncate_text(card.get("paper_id", ""), 80),
        "title": truncate_text(card.get("title", ""), EVIDENCE_TITLE_MAX_CHARS),
        "abstract": truncate_text(card.get("abstract", ""), EVIDENCE_ABSTRACT_MAX_CHARS),
        "task_category": truncate_text(card.get("task_category", ""), EVIDENCE_TASK_CATEGORY_MAX_CHARS),
        "method_name": truncate_text(card.get("method_name", ""), EVIDENCE_METHOD_NAME_MAX_CHARS),
        "method_summary": truncate_text(card.get("method_summary", ""), EVIDENCE_METHOD_SUMMARY_MAX_CHARS),
        "innovations": truncate_list_items(card.get("innovations", []), max_items=3, max_item_chars=180),
        "datasets_used_for_eval": truncate_list_items(card.get("datasets_used_for_eval", []), max_items=5, max_item_chars=120),
        "main_results": truncate_list_items(card.get("main_results", []), max_items=3, max_item_chars=180),
        "key_insights": truncate_text(card.get("key_insights", ""), EVIDENCE_KEY_INSIGHTS_MAX_CHARS),
        "conclusion": truncate_text(card.get("conclusion", ""), EVIDENCE_CONCLUSION_MAX_CHARS),
        "limitations": truncate_text(card.get("limitations", ""), EVIDENCE_LIMITATIONS_MAX_CHARS),
    }
    return {k: v for k, v in compact.items() if v not in (None, "", [])}


def public_evidence_card(card: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in card.items() if not k.startswith("_")}


def build_evidence_card(
    arxiv_id: str,
    metadata: Dict[str, Any],
    abstract: str,
    title_override: str = "",
    has_metadata: bool = False,
) -> Dict[str, Any]:
    title = nested_get(metadata, ["basic_info", "title"], "") or title_override
    metadata_abstract = nested_get(metadata, ["basic_info", "abstract"], "") or metadata.get("abstract", "")
    abstract = abstract or truncate_text(metadata_abstract, EVIDENCE_ABSTRACT_MAX_CHARS)
    card = {
        "paper_id": arxiv_id,
        "title": title,
        "abstract": abstract or "",
        "task_category": nested_get(metadata, ["categorization", "task_category"], ""),
        "keywords": as_list(nested_get(metadata, ["categorization", "keywords"], [])),
        "method_name": nested_get(metadata, ["method_details", "method_name"], ""),
        "method_summary": nested_get(metadata, ["method_details", "architecture_description"], ""),
        "innovations": as_list(nested_get(metadata, ["method_details", "innovations"], [])),
        "datasets_used_for_eval": as_list(nested_get(metadata, ["dataset_info", "datasets_used_for_eval"], [])),
        "main_results": as_list(nested_get(metadata, ["evaluation", "main_results"], [])),
        "motivation": nested_get(metadata, ["research_logic", "motivation"], ""),
        "key_insights": nested_get(metadata, ["research_logic", "key_insights"], ""),
        "conclusion": nested_get(metadata, ["research_logic", "conclusion"], ""),
        "limitations": nested_get(metadata, ["evaluation", "limitations"], ""),
        "_has_metadata": has_metadata,
        "_has_abstract": bool(abstract),
        "_has_title": bool(title),
    }
    return {k: v for k, v in card.items() if v not in (None, "", [])}


SEPARATED_CITATION_RANGE_RE = re.compile(
    r"\[\s*(\d+)\s*\]\s*(?:-{1,2}|[–—])\s*\[\s*(\d+)\s*\]"
)

UNICODE_SOURCE_CITATION_RE = re.compile(
    r"【\s*(\d+)\s*†[^】]*】"
)


def normalize_numeric_citation_syntax(text: str) -> str:
    """Canonicalize split ranges such as [39]-[56] to [39-56]."""
    return SEPARATED_CITATION_RANGE_RE.sub(
        lambda match: f"[{int(match.group(1))}-{int(match.group(2))}]",
        text,
    )


def parse_numeric_citation_bracket(text: str) -> Optional[str]:
    s = text.strip()
    if re.fullmatch(
        r"\d+(?:\s*[-–—]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-–—]\s*\d+)?)*",
        s,
    ):
        return s
    return None


def expand_citation_numbers(citation_text: str) -> List[str]:
    numbers: List[str] = []
    for part in re.split(r"[,;]", citation_text):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", part)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if start <= end and end - start <= 500:
                numbers.extend(str(i) for i in range(start, end + 1))
            continue
        if re.fullmatch(r"\d+", part):
            numbers.append(str(int(part)))
    seen = set()
    out: List[str] = []
    for n in numbers:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


CITATION_BRACKET_RE = re.compile(
    r"\[((?:\s*\d+\s*(?:[-–—]\s*\d+)?\s*)(?:[,;]\s*\d+\s*(?:[-–—]\s*\d+)?\s*)*)\]"
)


def has_supported_citation_marker(text: str) -> bool:
    normalized = normalize_numeric_citation_syntax(text)
    return bool(CITATION_BRACKET_RE.search(normalized) or UNICODE_SOURCE_CITATION_RE.search(normalized))


def iter_global_citation_matches(sentence: str) -> List[Tuple[List[str], int, int, str]]:
    """Return numeric reference groups and opaque deep-research source markers."""
    matches: List[Tuple[List[str], int, int, str]] = []
    for match in CITATION_BRACKET_RE.finditer(sentence):
        parsed = parse_numeric_citation_bracket(match.group(1))
        if not parsed:
            continue
        indices = expand_citation_numbers(parsed)
        if indices:
            style = "numeric_semicolon" if ";" in parsed else "numeric_bracket"
            matches.append((indices, match.start(), match.end(), style))
    for match in UNICODE_SOURCE_CITATION_RE.finditer(sentence):
        source_id = str(int(match.group(1)))
        matches.append(([f"source:{source_id}"], match.start(), match.end(), "unicode_source"))
    return sorted(matches, key=lambda item: item[1])


GEMINI_TRAILING_CITATION_RE = re.compile(
    r"(?<![\[\]\d./:#-])(?P<num>\d{1,3})\s*(?=\s*(?:[.,;:!?])?\s*$)"
)


def is_gemini_deep_research_method(method: str) -> bool:
    return (method or "").lower().replace("-", "_") == "gemini_deep_research"


def looks_like_trailing_link_number_after_bracket(sentence: str, start: int) -> bool:
    preceding = sentence[:start].rstrip()
    last_bracket = None
    for match in CITATION_BRACKET_RE.finditer(preceding):
        last_bracket = match
    if not last_bracket:
        return False
    return bool(re.fullmatch(r"[\s.,;:!?)]*", preceding[last_bracket.end() :]))


def probable_gemini_bare_citation_index(
    sentence: str,
    match: re.Match[str],
    ref_map: Dict[str, str],
    available_paper_ids: Iterable[str],
) -> Optional[str]:
    raw_num = match.group("num")
    citation_index = str(int(raw_num))
    paper_id = ref_map.get(citation_index)
    if not paper_id or paper_id not in available_paper_ids:
        return None
    start = match.start("num")
    end = match.end("num")
    if looks_like_trailing_link_number_after_bracket(sentence, start):
        return None
    before = sentence[:start]
    after = sentence[end:]
    if not re.fullmatch(r"\s*[.,;:!?]?\s*", after):
        return None
    if re.search(
        r"\b(?:figure|fig\.?|table|section|sec\.?|chapter|appendix|eq\.?|equation|algorithm|step|phase|version|rank|top)\s*$",
        before,
        flags=re.I,
    ):
        return None
    if re.search(
        r"\b(?:gpt|llama|claude|gemini|bert|vit|rt|dino|resnet|clip|yolo|mistral|mixtral|qwen|deepseek|palm|roberta|vgg)\s*[- ]*$",
        before,
        flags=re.I,
    ):
        return None
    return citation_index


def iter_citation_matches(
    sentence: str,
    method: str,
    ref_map: Dict[str, str],
    available_paper_ids: Iterable[str],
) -> List[Tuple[str, int, int, str]]:
    matches: List[Tuple[str, int, int, str]] = [
        (match.group(1), match.start(), match.end(), "bracket")
        for match in CITATION_BRACKET_RE.finditer(sentence)
    ]
    if is_gemini_deep_research_method(method):
        for match in GEMINI_TRAILING_CITATION_RE.finditer(sentence):
            citation_index = probable_gemini_bare_citation_index(sentence, match, ref_map, available_paper_ids)
            if citation_index:
                matches.append((citation_index, match.start("num"), match.end("num"), "gemini_bare"))
    return sorted(matches, key=lambda item: item[1])


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.count("|") >= 2:
        return True
    if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", stripped):
        return True
    return False


def clean_sentence(text: str, max_chars: int = 500) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def split_sentences(paragraph: str) -> List[str]:
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    if not paragraph:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+(?=[A-Z0-9\[【])", paragraph)
    if len(parts) == 1:
        return [paragraph]
    return [p.strip() for p in parts if p.strip()]


def is_citation_only_sentence(sentence: str) -> bool:
    text = normalize_numeric_citation_syntax(sentence)
    text = CITATION_BRACKET_RE.sub(" ", text)
    text = UNICODE_SOURCE_CITATION_RE.sub(" ", text)
    text = re.sub(r"\b\d{1,3}\b", " ", text)
    text = re.sub(r"[\s.,;:!?()\[\]{}\-–—]+", "", text)
    return not text


def is_standalone_link_number_sentence(sentence: str) -> bool:
    return bool(re.fullmatch(r"\s*\d{1,3}\s*[.)]?\s*", sentence))


def attach_standalone_citation_sentences(sentences: Sequence[str]) -> Tuple[List[str], int, int]:
    attached = 0
    skipped = 0
    merged: List[str] = []
    for sentence in sentences:
        if is_citation_only_sentence(sentence):
            if merged:
                merged[-1] = f"{merged[-1]} {sentence}"
                attached += 1
            else:
                skipped += 1
            continue
        if is_standalone_link_number_sentence(sentence):
            skipped += 1
            continue
        merged.append(sentence)
    return merged, attached, skipped


def is_reference_entry_line(line: str) -> bool:
    stripped = line.strip()
    match = re.match(r"^\[(\d{1,4})\]\s+(.+)$", stripped)
    if not match:
        return False
    rest = match.group(2)
    if re.search(
        r"\b(?:arXiv|Proceedings|Conference|Journal|Transactions|Findings|Workshop|URL|doi|preprint|ICLR|NeurIPS|ACL|EMNLP|AAAI|CVPR|ICML|SIGDIAL|UAI)\b",
        rest,
        flags=re.I,
    ):
        return True
    if re.search(r"\b(?:19|20)\d{2}\b", rest) and ("," in rest[:160] or " and " in rest[:160]):
        return True
    return False


def mark_citation_sentence(sentence: str, target_index: str) -> str:
    sentence = normalize_numeric_citation_syntax(sentence)

    def repl(match: re.Match[str]) -> str:
        nums = expand_citation_numbers(match.group(1))
        if target_index in nums:
            return "[CITE_THIS]" if len(nums) == 1 else "[CITE_THIS, CITE_OTHER]"
        return "[CITE_OTHER]"

    return clean_sentence(CITATION_BRACKET_RE.sub(repl, sentence))


def extract_citation_contexts(
    survey_md_text: str,
    ref_map: Dict[str, str],
    available_paper_ids: Iterable[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """Backward-compatible paper-centric extractor retained for callers outside BSC-v2."""
    available = set(available_paper_ids)
    contexts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    claims, stats = extract_claim_contexts(survey_md_text, ref_map, available, topic_id="000")
    for claim in claims:
        for item in claim.get("citation_group", []):
            paper_id = item.get("paper_id")
            if paper_id not in available:
                continue
            contexts[paper_id].append(
                {
                    "section": claim.get("section_title", "Untitled"),
                    "sentence": claim.get("marked_sentence", ""),
                    "citation_index": item.get("citation_index", ""),
                    "position": claim.get("position", 0),
                }
            )
    return dict(contexts), {
        "table_citations_skipped": stats.get("table_citations_skipped", 0),
        "code_citations_skipped": stats.get("code_citations_skipped", 0),
    }


def infer_claim_type(sentence: str, section: str) -> str:
    text = f"{section} {sentence}".lower()
    if re.search(r"\b(prompt injection|attack|poisoning|adversarial|vulnerability|robustness|jailbreak|security|defense|threat|malicious)\b", text):
        return "security_or_robustness"
    if re.search(r"\b(latency|cost|token|compression|quantization|pruning|edge|deployment|throughput|memory|cache|serving|efficient|efficiency)\b", text):
        return "efficiency_or_deployment"
    if re.search(r"\b(taxonomy|categorize|paradigm|classify|survey|line of work|trend|convergence|trade[- ]?off|comparison|shared limitation|family|families|direction|research area|landscape)\b", text):
        return "taxonomy_or_synthesis"
    if re.search(r"\b(outperform|achieve|accuracy|success rate|result|performance|improve|gain|score|win rate|pass rate)\b", text):
        return "result_finding"
    if re.search(r"\b(benchmark|dataset|evaluation|leaderboard)\b", text):
        return "benchmark_role"
    if re.search(r"\b(limitation|challenge|fail|failure|struggle|brittle|weakness|risk|degrade|error|hallucination)\b", text):
        return "limitation"
    if re.search(r"\b(framework|method|model|architecture|pipeline|mechanism|approach|module|agent|retrieval|planning|tool|function calling|orchestration|fine[- ]?tuning)\b", text):
        return "method_description"
    if re.search(r"\b(dataset|task|corpus|benchmark)\b", text):
        return "dataset_or_task"
    if re.search(r"\b(introduction|background)\b", section.lower()):
        return "background"
    return "other"


def citation_group_paper_ids(claim: Dict[str, Any]) -> List[str]:
    return [x["paper_id"] for x in claim.get("citation_group", []) if x.get("paper_id")]


def group_size_bin(size: int) -> str:
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if size == 3:
        return "3"
    if size == 4:
        return "4"
    if size <= 8:
        return "5-8"
    if size <= 15:
        return "9-15"
    return "16+"


def citation_group_balance_stats(sizes: Sequence[int], max_group_size: int) -> Dict[str, Any]:
    total = len(sizes)
    if total == 0:
        return {
            "large_citation_group_ratio_all": 0.0,
            "oversized_citation_group_ratio_all": 0.0,
            "avg_citation_group_size_all": 0.0,
            "max_citation_group_size_observed": 0,
            "single_citation_claim_ratio_all": 0.0,
        }
    large = sum(1 for size in sizes if size >= 5)
    oversized = sum(1 for size in sizes if size > max_group_size)
    singles = sum(1 for size in sizes if size == 1)
    return {
        "large_citation_group_ratio_all": round(large / total, 4),
        "oversized_citation_group_ratio_all": round(oversized / total, 4),
        "avg_citation_group_size_all": round(sum(sizes) / total, 4),
        "max_citation_group_size_observed": max(sizes),
        "single_citation_claim_ratio_all": round(singles / total, 4),
    }


def claim_stats_balance_fields(claim_stats: Dict[str, Any]) -> Dict[str, Any]:
    sizes = [int(size) for size in claim_stats.get("citation_group_sizes_all", [])]
    max_group_size = int(claim_stats.get("max_group_size", 15))
    return citation_group_balance_stats(sizes, max_group_size)


def is_intro_or_conclusion(section: str) -> bool:
    return bool(re.search(r"\b(introduction|conclusion)\b", section.lower()))


def claim_type_priority(claim_type: str, citation_mode: str) -> int:
    if citation_mode == "group":
        order = [
            "taxonomy_or_synthesis",
            "method_description",
            "security_or_robustness",
            "efficiency_or_deployment",
            "benchmark_role",
            "result_finding",
            "limitation",
            "dataset_or_task",
            "background",
            "other",
        ]
    else:
        order = [
            "method_description",
            "result_finding",
            "benchmark_role",
            "security_or_robustness",
            "efficiency_or_deployment",
            "limitation",
            "dataset_or_task",
            "taxonomy_or_synthesis",
            "background",
            "other",
        ]
    return order.index(claim_type) if claim_type in order else len(order)


def mark_claim_citation_group(sentence: str, valid_citation_indices: Iterable[str]) -> str:
    sentence = normalize_numeric_citation_syntax(sentence)
    valid = {str(x) for x in valid_citation_indices}

    def repl(match: re.Match[str]) -> str:
        nums = expand_citation_numbers(match.group(1))
        return "[CITE_THIS]" if any(n in valid for n in nums) else "[CITE_OTHER]"

    return clean_sentence(CITATION_BRACKET_RE.sub(repl, sentence), 900)


def remove_numeric_citations(text: str) -> str:
    text = normalize_numeric_citation_syntax(text)
    return CITATION_BRACKET_RE.sub(" ", text)


def remove_supported_citations(text: str) -> str:
    text = remove_numeric_citations(text)
    return UNICODE_SOURCE_CITATION_RE.sub(" ", text)


def marker_free_claim_text(text: str) -> str:
    text = re.sub(r"\[CITE_(?:THIS|OTHER)(?:,\s*CITE_OTHER)?\]", " ", text)
    return clean_sentence(remove_numeric_citations(text), 700)


def has_evaluable_claim_text(text: str) -> bool:
    return any(ch.isalpha() for ch in marker_free_claim_text(text))


def normalize_clause_key(text: str) -> str:
    text = remove_numeric_citations(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clauses_are_similar(a: str, b: str) -> bool:
    ka = normalize_clause_key(a)
    kb = normalize_clause_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    shorter, longer = sorted((ka, kb), key=len)
    return len(shorter) >= 40 and shorter in longer


def extract_citation_attached_clause(sentence: str, citation_start: int, citation_end: Optional[int] = None) -> Tuple[str, str]:
    """
    Return the smallest reasonable clause before the citation bracket.

    Returns:
        clause_text: citation-attached clause without citation brackets
        extraction_unit: "clause" or "sentence_fallback"
    """
    before = sentence[:citation_start]
    candidate = before
    marker_re = re.compile(
        r"\b(however|therefore|consequently|whereas|while|although|but|by contrast|conversely|furthermore|moreover|nevertheless|in contrast)\b",
        flags=re.I,
    )
    markers = list(marker_re.finditer(candidate))
    if markers:
        candidate = candidate[markers[-1].start():]
    else:
        split_positions = []
        for token in (";", ":", "—", "–"):
            pos = candidate.rfind(token)
            if pos >= 0:
                split_positions.append(pos)
        if split_positions:
            candidate = candidate[max(split_positions) + 1 :]
        elif len(candidate) > 300:
            comma_pos = candidate.rfind(",")
            if comma_pos >= 0:
                tail = candidate[comma_pos + 1 :]
                if len(clean_sentence(remove_supported_citations(tail), 700)) >= 40:
                    candidate = tail
    clause_text = clean_sentence(remove_supported_citations(candidate), 700)
    extraction_unit = "clause"
    if len(clause_text) < 40:
        fallback_sentence = sentence
        if citation_end is not None and citation_end > citation_start:
            fallback_sentence = f"{sentence[:citation_start]} {sentence[citation_end:]}"
        clause_text = clean_sentence(remove_supported_citations(fallback_sentence), 500)
        extraction_unit = "sentence_fallback"
    if len(clause_text) > 500:
        clause_text = clean_sentence(clause_text, 500)
    return clause_text, extraction_unit


def mark_clause_with_cite_this(clause_text: str) -> str:
    text = clean_sentence(remove_supported_citations(clause_text), 700)
    text = text.rstrip()
    if not text:
        return "[CITE_THIS]"
    text = text.rstrip(".!?")
    return f"{text} [CITE_THIS]."



def extract_global_citation_distribution(
    survey_md_text: str,
    method: str = "",
) -> Dict[str, Any]:
    """Summarize supported body citation markers without using ref.json."""
    events: List[Dict[str, Any]] = []
    current_section = "Untitled"
    paragraph_section = current_section
    paragraph_lines: List[str] = []
    body_sections = set()
    in_code = False
    in_html_table = False
    seen_events = set()
    skipped = {"table": 0, "code": 0, "caption": 0, "reference_entry": 0}
    syntax_counts: Counter[str] = Counter()

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph = " ".join(paragraph_lines)
        paragraph_lines = []
        body_sections.add(paragraph_section)
        sentences, _, _ = attach_standalone_citation_sentences(split_sentences(paragraph))
        for raw_sentence in sentences:
            syntax_counts["separated_range_events"] += len(SEPARATED_CITATION_RANGE_RE.findall(raw_sentence))
            sentence = normalize_numeric_citation_syntax(raw_sentence)
            clause_groups: List[Dict[str, Any]] = []
            for indices, citation_start, citation_end, citation_style in iter_global_citation_matches(sentence):
                if citation_style.startswith("numeric_"):
                    syntax_counts["numeric_bracket_events"] += 1
                    if citation_style == "numeric_semicolon":
                        syntax_counts["semicolon_group_events"] += 1
                elif citation_style == "unicode_source":
                    syntax_counts["unicode_source_citation_events"] += 1
                clause_text, scope = extract_citation_attached_clause(sentence, citation_start, citation_end)
                if not clause_text or not has_evaluable_claim_text(clause_text):
                    continue
                target = None
                for existing in clause_groups:
                    if existing["scope"] == scope and clauses_are_similar(existing["clause_text"], clause_text):
                        target = existing
                        break
                if target is None:
                    target = {"clause_text": clause_text, "scope": scope, "citation_indices": []}
                    clause_groups.append(target)
                for citation_index in indices:
                    if citation_index not in target["citation_indices"]:
                        target["citation_indices"].append(citation_index)
            for group in clause_groups:
                citation_indices = group["citation_indices"]
                key = (
                    paragraph_section,
                    normalize_clause_key(group["clause_text"]),
                    tuple(citation_indices),
                )
                if key in seen_events:
                    continue
                seen_events.add(key)
                events.append(
                    {
                        "section": paragraph_section,
                        "citation_indices": citation_indices,
                        "group_size": len(citation_indices),
                        "claim_text": clean_sentence(group["clause_text"], 320),
                    }
                )

    for raw_line in survey_md_text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if in_code and has_supported_citation_marker(stripped):
                skipped["code"] += 1
            flush_paragraph()
            in_code = not in_code
            continue
        if in_code:
            if has_supported_citation_marker(stripped):
                skipped["code"] += 1
            continue
        if is_reference_entry_line(stripped):
            flush_paragraph()
            skipped["reference_entry"] += 1
            continue
        if re.search(r"<table\b", stripped, flags=re.I):
            flush_paragraph()
            if has_supported_citation_marker(stripped):
                skipped["table"] += 1
            if not re.search(r"</table>", stripped, flags=re.I):
                in_html_table = True
            continue
        if in_html_table:
            if has_supported_citation_marker(stripped):
                skipped["table"] += 1
            if re.search(r"</table>", stripped, flags=re.I):
                in_html_table = False
            continue
        if is_table_line(stripped):
            flush_paragraph()
            if has_supported_citation_marker(stripped):
                skipped["table"] += 1
            continue
        if re.match(r"^(figure|fig\.|table)\s+\d+\b", stripped, flags=re.I):
            if has_supported_citation_marker(stripped):
                skipped["caption"] += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
        if heading:
            flush_paragraph()
            title = clean_sentence(heading.group(2), 180)
            current_section = title or current_section
            if re.fullmatch(r"references|bibliography", current_section, flags=re.I):
                current_section = "References"
            continue
        if re.fullmatch(r"references|bibliography", stripped, flags=re.I):
            flush_paragraph()
            current_section = "References"
            continue
        if current_section == "References":
            continue
        if not stripped:
            flush_paragraph()
            continue
        if not paragraph_lines:
            paragraph_section = current_section
        paragraph_lines.append(stripped)
    flush_paragraph()

    citation_use: Dict[str, Dict[str, Any]] = {}
    section_claims = Counter()
    section_mentions = Counter()
    section_unique: Dict[str, set] = defaultdict(set)
    group_use: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    group_sizes: List[int] = []
    for event in events:
        section = event["section"]
        indices = event["citation_indices"]
        section_claims[section] += 1
        section_mentions[section] += len(indices)
        section_unique[section].update(indices)
        group_sizes.append(len(indices))
        group_key = tuple(indices)
        group_info = group_use.setdefault(group_key, {"count": 0, "sections": set()})
        group_info["count"] += 1
        group_info["sections"].add(section)
        for citation_index in indices:
            info = citation_use.setdefault(citation_index, {"count": 0, "sections": set()})
            info["count"] += 1
            info["sections"].add(section)

    total_mentions = sum(info["count"] for info in citation_use.values())
    sorted_use = sorted(
        citation_use.items(),
        key=lambda item: (-item[1]["count"], (0, int(item[0])) if item[0].isdigit() else (1, item[0])),
    )
    top_reused = [
        {
            "citation_index": citation_index,
            "num_claims": info["count"],
            "num_sections": len(info["sections"]),
        }
        for citation_index, info in sorted_use[:20]
    ]
    repeated_groups = [
        {
            "citation_indices": list(group),
            "num_occurrences": info["count"],
            "num_sections": len(info["sections"]),
        }
        for group, info in group_use.items()
        if info["count"] > 1
    ]
    repeated_groups.sort(key=lambda item: (-item["num_occurrences"], -len(item["citation_indices"]), item["citation_indices"]))
    repeated_group_claims = sum(info["count"] for info in group_use.values() if info["count"] > 1)
    largest_groups = sorted(events, key=lambda event: (-event["group_size"], event["section"], event["citation_indices"]))[:10]
    group_balance = citation_group_balance_stats(group_sizes, max_group_size=15)
    sections_with_citations = set(section_claims)
    total_sections = len(body_sections)

    return {
        "scope": "all supported citation markers parsed from the Markdown body; independent of ref.json and evidence cards",
        "citation_syntax_counts": dict(sorted(syntax_counts.items())),
        "total_body_sections": total_sections,
        "sections_with_citations": len(sections_with_citations),
        "section_citation_coverage_ratio": round(len(sections_with_citations) / max(1, total_sections), 4),
        "total_cited_claims": len(events),
        "total_citation_mentions": total_mentions,
        "num_unique_citation_indices": len(citation_use),
        "single_citation_claims": sum(1 for size in group_sizes if size == 1),
        "multi_citation_claims": sum(1 for size in group_sizes if size > 1),
        "multi_citation_claim_ratio": round(sum(1 for size in group_sizes if size > 1) / max(1, len(group_sizes)), 4),
        "citation_group_size_distribution": dict(sorted(Counter(group_size_bin(size) for size in group_sizes).items())),
        "avg_citation_group_size": group_balance["avg_citation_group_size_all"],
        "max_citation_group_size": group_balance["max_citation_group_size_observed"],
        "large_citation_group_ratio": group_balance["large_citation_group_ratio_all"],
        "oversized_citation_group_ratio": group_balance["oversized_citation_group_ratio_all"],
        "top_1_citation_mention_share": round((sorted_use[0][1]["count"] if sorted_use else 0) / max(1, total_mentions), 4),
        "top_5_citation_mention_share": round(sum(info["count"] for _, info in sorted_use[:5]) / max(1, total_mentions), 4),
        "num_repeated_exact_citation_groups": len(repeated_groups),
        "repeated_exact_group_claim_ratio": round(repeated_group_claims / max(1, len(events)), 4),
        "cited_claims_per_section": dict(sorted(section_claims.items())),
        "citation_mentions_per_section": dict(sorted(section_mentions.items())),
        "unique_citations_per_section": dict(sorted((section, len(indices)) for section, indices in section_unique.items())),
        "top_reused_citations": top_reused,
        "top_repeated_exact_citation_groups": repeated_groups[:10],
        "largest_citation_groups": largest_groups,
        "skipped_non_body_citation_lines": skipped,
    }


def extract_claim_contexts(
    survey_md_text: str,
    ref_map: Dict[str, str],
    available_paper_ids: Iterable[str],
    topic_id: str,
    max_group_size: int = 15,
    method: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    available = set(available_paper_ids)
    claims: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "table_citations_skipped": 0,
        "code_citations_skipped": 0,
        "caption_citations_skipped": 0,
        "oversized_citation_group_claims_skipped": 0,
        "citation_group_size_distribution_all": {},
        "claim_scope_distribution_all": {},
        "single_citation_claims": 0,
        "multi_citation_claims": 0,
        "num_clause_scope_claims": 0,
        "num_sentence_fallback_claims": 0,
        "gemini_bare_citations_detected": 0,
        "separated_range_citations_detected": 0,
        "semicolon_citation_groups_detected": 0,
        "standalone_citation_sentences_attached": 0,
        "citation_only_sentences_skipped": 0,
        "reference_entry_lines_skipped": 0,
        "low_information_claims_skipped": 0,
        "citation_group_sizes_all": [],
        "max_group_size": max_group_size,
    }
    current_section = "Untitled"
    section_path = [current_section]
    paragraph_lines: List[str] = []
    paragraph_section = current_section
    paragraph_id = 0
    sentence_id = 0
    position = 0
    in_code = False
    in_html_table = False
    seen_claims = set()

    def add_group_size(size: int) -> None:
        key = group_size_bin(size)
        dist = stats.setdefault("citation_group_size_distribution_all", {})
        dist[key] = dist.get(key, 0) + 1
        sizes = stats.setdefault("citation_group_sizes_all", [])
        if isinstance(sizes, list):
            sizes.append(int(size))

    def add_scope(scope: str) -> None:
        dist = stats.setdefault("claim_scope_distribution_all", {})
        dist[scope] = dist.get(scope, 0) + 1
        if scope == "clause":
            stats["num_clause_scope_claims"] = stats.get("num_clause_scope_claims", 0) + 1
        elif scope == "sentence_fallback":
            stats["num_sentence_fallback_claims"] = stats.get("num_sentence_fallback_claims", 0) + 1

    def citation_group_for_text(citation_text: str) -> List[Dict[str, str]]:
        parsed = parse_numeric_citation_bracket(citation_text)
        if not parsed:
            return []
        group = []
        seen_local = set()
        for idx in expand_citation_numbers(parsed):
            if idx in seen_local:
                continue
            seen_local.add(idx)
            paper_id = ref_map.get(idx)
            if paper_id and paper_id in available:
                group.append({"citation_index": idx, "paper_id": paper_id, "evidence_ref_id": ""})
        return group

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, paragraph_id, sentence_id, position
        if not paragraph_lines:
            return
        paragraph_id += 1
        paragraph = " ".join(paragraph_lines)
        paragraph_lines = []
        sentences, attached, citation_only_skipped = attach_standalone_citation_sentences(split_sentences(paragraph))
        stats["standalone_citation_sentences_attached"] += attached
        stats["citation_only_sentences_skipped"] += citation_only_skipped
        for raw_sentence in sentences:
            sentence_id += 1
            stats["separated_range_citations_detected"] += len(SEPARATED_CITATION_RANGE_RE.findall(raw_sentence))
            sentence = normalize_numeric_citation_syntax(raw_sentence)
            matches = iter_citation_matches(sentence, method, ref_map, available)
            if not matches:
                continue
            clause_groups: List[Dict[str, Any]] = []
            for citation_text, citation_start, citation_end, citation_style in matches:
                match_group = citation_group_for_text(citation_text)
                if not match_group:
                    continue
                if citation_style == "gemini_bare":
                    stats["gemini_bare_citations_detected"] += 1
                elif ";" in citation_text:
                    stats["semicolon_citation_groups_detected"] += 1
                clause_text, scope = extract_citation_attached_clause(sentence, citation_start, citation_end)
                if not clause_text:
                    continue
                target = None
                for existing in clause_groups:
                    if existing["scope"] == scope and clauses_are_similar(existing["clause_text"], clause_text):
                        target = existing
                        break
                if target is None:
                    target = {"clause_text": clause_text, "scope": scope, "citation_group": []}
                    clause_groups.append(target)
                seen_pair = {(x["citation_index"], x["paper_id"]) for x in target["citation_group"]}
                for item in match_group:
                    pair = (item["citation_index"], item["paper_id"])
                    if pair not in seen_pair:
                        target["citation_group"].append(item)
                        seen_pair.add(pair)
            for clause in clause_groups:
                citation_group = clause["citation_group"]
                if not citation_group:
                    continue
                if not has_evaluable_claim_text(clause["clause_text"]):
                    stats["low_information_claims_skipped"] += 1
                    continue
                group_size = len(citation_group)
                add_group_size(group_size)
                add_scope(clause["scope"])
                if group_size > max_group_size:
                    stats["oversized_citation_group_claims_skipped"] += 1
                    continue
                key = (
                    paragraph_section,
                    normalize_clause_key(clause["clause_text"]),
                    tuple((x["citation_index"], x["paper_id"]) for x in citation_group),
                )
                if key in seen_claims:
                    continue
                seen_claims.add(key)
                position += 1
                citation_mode = "single" if group_size == 1 else "group"
                if citation_mode == "single":
                    stats["single_citation_claims"] += 1
                else:
                    stats["multi_citation_claims"] += 1
                claim_text = clean_sentence(clause["clause_text"], 700)
                claims.append(
                    {
                        "record_type": "bsc_claim_context",
                        "claim_id": f"{topic_id}-C{position:04d}",
                        "citation_mode": citation_mode,
                        "section_title": paragraph_section,
                        "marked_sentence": mark_clause_with_cite_this(claim_text),
                        "original_sentence_with_citations": clean_sentence(raw_sentence, 900),
                        "citation_group": citation_group,
                        "citation_group_size": group_size,
                        "citation_scope": clause["scope"],
                        "claim_type": infer_claim_type(claim_text, paragraph_section),
                        "paragraph_id": paragraph_id,
                        "sentence_id": sentence_id,
                        "position": position,
                    }
                )

    for raw_line in survey_md_text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if in_code and CITATION_BRACKET_RE.search(stripped):
                stats["code_citations_skipped"] += 1
            flush_paragraph()
            in_code = not in_code
            continue
        if in_code:
            if CITATION_BRACKET_RE.search(stripped):
                stats["code_citations_skipped"] += 1
            continue
        if is_reference_entry_line(stripped):
            flush_paragraph()
            stats["reference_entry_lines_skipped"] += 1
            continue
        if re.search(r"<table\b", stripped, flags=re.I):
            flush_paragraph()
            if CITATION_BRACKET_RE.search(stripped):
                stats["table_citations_skipped"] += 1
            if not re.search(r"</table>", stripped, flags=re.I):
                in_html_table = True
            continue
        if in_html_table:
            if CITATION_BRACKET_RE.search(stripped):
                stats["table_citations_skipped"] += 1
            if re.search(r"</table>", stripped, flags=re.I):
                in_html_table = False
            continue
        if is_table_line(stripped):
            flush_paragraph()
            if CITATION_BRACKET_RE.search(stripped):
                stats["table_citations_skipped"] += 1
            continue
        if re.match(r"^(figure|fig\.|table)\s+\d+\b", stripped, flags=re.I):
            if CITATION_BRACKET_RE.search(stripped):
                stats["caption_citations_skipped"] += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
        if heading:
            flush_paragraph()
            title = clean_sentence(heading.group(2), 180)
            current_section = title or current_section
            section_path = [current_section]
            if re.fullmatch(r"references|bibliography", current_section, flags=re.I):
                current_section = "References"
                section_path = [current_section]
            continue
        if re.fullmatch(r"references|bibliography", stripped, flags=re.I):
            flush_paragraph()
            current_section = "References"
            section_path = [current_section]
            continue
        if current_section == "References":
            continue
        if not stripped:
            flush_paragraph()
            continue
        if not paragraph_lines:
            paragraph_section = current_section
        paragraph_lines.append(stripped)
    flush_paragraph()
    stats["total_cited_claim_candidates"] = len(claims)
    stats["claim_scope_distribution_all"] = dict(sorted(Counter(c.get("citation_scope", "unknown") for c in claims).items()))
    stats["num_clause_scope_claims"] = stats["claim_scope_distribution_all"].get("clause", 0)
    stats["num_sentence_fallback_claims"] = stats["claim_scope_distribution_all"].get("sentence_fallback", 0)
    return claims, stats

def evidence_completeness_for_claim(claim: Dict[str, Any], cards: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    counts = {
        "metadata_and_abstract": 0,
        "metadata_only": 0,
        "abstract_only": 0,
        "title_only": 0,
        "missing": 0,
        "insufficient": 0,
    }
    for paper_id in citation_group_paper_ids(claim):
        card = cards.get(paper_id)
        if not card:
            counts["missing"] += 1
            continue
        tier = evidence_tier_name(card)
        if tier in counts:
            counts[tier] += 1
        else:
            counts["insufficient"] += 1
    counts["num_papers"] = len(citation_group_paper_ids(claim))
    counts["tier"] = claim_evidence_tier(claim, cards)
    return counts


def claim_evidence_tier(claim: Dict[str, Any], cards: Dict[str, Dict[str, Any]]) -> str:
    ranks = []
    for paper_id in citation_group_paper_ids(claim):
        card = cards.get(paper_id)
        if not card or evidence_tier_rank(card) >= 4:
            return "has_missing_or_insufficient"
        ranks.append(evidence_tier_rank(card))
    if not ranks:
        return "has_missing_or_insufficient"
    if max(ranks) == 0:
        return "all_metadata_and_abstract"
    if any(r == 1 for r in ranks):
        return "mixed_with_metadata_only"
    if any(r == 2 for r in ranks):
        return "mixed_with_abstract_only"
    return "title_only_or_weak"


def claim_evidence_tier_rank(claim: Dict[str, Any], cards: Dict[str, Dict[str, Any]]) -> int:
    order = {
        "all_metadata_and_abstract": 0,
        "mixed_with_metadata_only": 1,
        "mixed_with_abstract_only": 2,
        "title_only_or_weak": 3,
        "has_missing_or_insufficient": 4,
    }
    return order.get(claim_evidence_tier(claim, cards), 4)


def select_claims_for_bsc(
    claim_candidates: List[Dict[str, Any]],
    cards: Dict[str, Dict[str, Any]],
    max_claim_records: int = 100,
    max_unique_papers: int = 100,
    max_single_claims: int = 50,
    max_multi_claims: int = 50,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    bank: set[str] = set()
    per_paper_single: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    group_bin_counts: Counter[str] = Counter()
    stats: Dict[str, Any] = {
        "skipped_budget": 0,
        "skipped_missing_or_insufficient": 0,
        "skipped_quota": 0,
        "selected_multi_claims": 0,
        "selected_single_claims": 0,
        "selected_other_claims": 0,
    }
    other_claim_soft_cap = max(20, int(0.25 * max_claim_records))

    def can_add(claim: Dict[str, Any], quota: int, current_count: int) -> Tuple[bool, str]:
        if len(selected) >= max_claim_records:
            return False, "claim_record_budget"
        if current_count >= quota:
            return False, "mode_quota"
        paper_ids = citation_group_paper_ids(claim)
        if not paper_ids or claim_evidence_tier_rank(claim, cards) >= 4:
            return False, "missing_or_insufficient"
        if len(bank | set(paper_ids)) > max_unique_papers:
            return False, "paper_budget"
        return True, ""

    type_order_len = 8

    def base_sort_key(claim: Dict[str, Any]) -> Tuple[Any, ...]:
        paper_ids = citation_group_paper_ids(claim)
        marginal_cost = len(set(paper_ids) - bank)
        section = claim.get("section_title", "")
        group_size = int(claim.get("citation_group_size", 0))
        return (
            claim_evidence_tier_rank(claim, cards),
            is_intro_or_conclusion(section),
            claim_type_priority(claim.get("claim_type", "other"), claim.get("citation_mode", "single")),
            marginal_cost,
            group_size,
            int(claim.get("position", 10**9)),
        )

    multi = [c for c in claim_candidates if c.get("citation_mode") == "group" and 2 <= int(c.get("citation_group_size", 0)) <= 8]
    single = [c for c in claim_candidates if c.get("citation_mode") == "single"]

    def add_claim(claim: Dict[str, Any], reason: str) -> None:
        nonlocal bank
        c = json.loads(json.dumps(claim, ensure_ascii=False))
        c["evidence_completeness"] = evidence_completeness_for_claim(c, cards)
        c["selection_reason"] = reason
        selected.append(c)
        paper_ids = citation_group_paper_ids(c)
        bank.update(paper_ids)
        section_counts[c.get("section_title", "Untitled")] += 1
        group_bin_counts[group_size_bin(int(c.get("citation_group_size", 0)))] += 1
        if c.get("claim_type") == "other":
            stats["selected_other_claims"] += 1
        if c.get("citation_mode") == "single" and paper_ids:
            per_paper_single[paper_ids[0]] += 1

    # Multi-citation claims first. Re-sort greedily so marginal cost and coverage pressure can update.
    while stats["selected_multi_claims"] < max_multi_claims and len(selected) < max_claim_records:
        remaining = [c for c in multi if not c.get("_selected")]
        if not remaining:
            break
        def multi_key(c: Dict[str, Any]) -> Tuple[Any, ...]:
            bin_name = group_size_bin(int(c.get("citation_group_size", 0)))
            section = c.get("section_title", "Untitled")
            return (
                claim_evidence_tier_rank(c, cards),
                group_bin_counts[bin_name],
                section_counts[section],
                c.get("claim_type") == "other" and stats["selected_other_claims"] >= other_claim_soft_cap,
                is_intro_or_conclusion(section),
                claim_type_priority(c.get("claim_type", "other"), "group"),
                len(set(citation_group_paper_ids(c)) - bank),
                int(c.get("position", 10**9)),
            )
        claim = min(remaining, key=multi_key)
        claim["_selected"] = True
        ok, why = can_add(claim, max_multi_claims, stats["selected_multi_claims"])
        if not ok:
            if why == "paper_budget":
                stats["skipped_budget"] += 1
            elif why == "missing_or_insufficient":
                stats["skipped_missing_or_insufficient"] += 1
            else:
                stats["skipped_quota"] += 1
            continue
        add_claim(claim, "multi_claim_priority_with_citation_closure")
        stats["selected_multi_claims"] += 1

    single_target = max_claim_records - stats["selected_multi_claims"]

    # Single-citation claims: already-in-bank first, then new papers if budget remains.
    for phase in ("already_in_bank", "new_paper"):
        while stats["selected_single_claims"] < single_target and len(selected) < max_claim_records:
            remaining = [c for c in single if not c.get("_selected")]
            if phase == "already_in_bank":
                remaining = [c for c in remaining if set(citation_group_paper_ids(c)).issubset(bank)]
            else:
                remaining = [c for c in remaining if not set(citation_group_paper_ids(c)).issubset(bank)]
            if not remaining:
                break
            def single_key(c: Dict[str, Any]) -> Tuple[Any, ...]:
                paper_ids = citation_group_paper_ids(c)
                paper_id = paper_ids[0] if paper_ids else ""
                section = c.get("section_title", "Untitled")
                return (
                    claim_evidence_tier_rank(c, cards),
                    c.get("claim_type") == "other" and stats["selected_other_claims"] >= other_claim_soft_cap,
                    per_paper_single[paper_id],
                    section_counts[section],
                    claim_type_priority(c.get("claim_type", "other"), "single"),
                    len(set(paper_ids) - bank),
                    int(c.get("position", 10**9)),
                )
            claim = min(remaining, key=single_key)
            claim["_selected"] = True
            ok, why = can_add(claim, single_target, stats["selected_single_claims"])
            if not ok:
                if why == "paper_budget":
                    stats["skipped_budget"] += 1
                elif why == "missing_or_insufficient":
                    stats["skipped_missing_or_insufficient"] += 1
                else:
                    stats["skipped_quota"] += 1
                continue
            add_claim(claim, f"single_claim_{phase}")
            stats["selected_single_claims"] += 1

    for claim in claim_candidates:
        claim.pop("_selected", None)
    selected_paper_ids = sorted({pid for claim in selected for pid in citation_group_paper_ids(claim)})
    stats["num_selected_claims"] = len(selected)
    stats["num_selected_papers"] = len(selected_paper_ids)
    stats["single_claim_target_after_backfill"] = single_target
    stats["num_single_backfill_claims"] = max(0, stats["selected_single_claims"] - max_single_claims)
    stats["multi_claim_shortfall"] = max(0, max_multi_claims - stats["selected_multi_claims"])
    stats["other_claim_soft_cap"] = other_claim_soft_cap
    stats["selected_other_claim_ratio"] = round(stats["selected_other_claims"] / max(1, len(selected)), 4)
    stats["selected_group_size_distribution"] = dict(Counter(group_size_bin(int(c.get("citation_group_size", 0))) for c in selected))
    stats["selected_claim_type_distribution"] = dict(Counter(c.get("claim_type", "other") for c in selected))
    stats["selected_claim_scope_distribution"] = dict(Counter(c.get("citation_scope", "unknown") for c in selected))
    return selected, selected_paper_ids, stats


def build_evidence_card_bank(
    selected_paper_ids: Sequence[str],
    citation_index_by_paper: Dict[str, List[str]],
    cards: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    evidence_ref_id_by_paper: Dict[str, str] = {}
    papers = []
    for i, paper_id in enumerate(selected_paper_ids, start=1):
        evidence_ref_id = f"E{i:03d}"
        evidence_ref_id_by_paper[paper_id] = evidence_ref_id
        papers.append(
            {
                "evidence_ref_id": evidence_ref_id,
                "citation_indices": citation_index_by_paper.get(paper_id, []),
                "paper_id": paper_id,
                "title": cards[paper_id].get("title", ""),
                "evidence_tier": evidence_tier_name(cards[paper_id]),
                "evidence_card": compact_public_evidence_card(cards[paper_id]),
            }
        )
    return evidence_ref_id_by_paper, {"record_type": "evidence_card_bank", "num_papers": len(papers), "papers": papers}


def short_field(value: Any, max_chars: int = 220) -> str:
    if isinstance(value, list):
        value = "; ".join(str(x) for x in value[:3])
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = clean_sentence(str(value or ""), max_chars)
    return text


def fill_claim_evidence_ref_ids(
    claims: Sequence[Dict[str, Any]],
    evidence_ref_id_by_paper: Dict[str, str],
    cards: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    filled = []
    for claim in claims:
        c = json.loads(json.dumps(claim, ensure_ascii=False))
        for key in list(c.keys()):
            if key.startswith("_"):
                c.pop(key, None)
        for key in (
            "paragraph_context",
            "claim_sentence",
            "section_path",
            "subsection_title",
            "selection_reason",
            "citation_evidence_preview",
        ):
            c.pop(key, None)
        for item in c.get("citation_group", []):
            paper_id = item.get("paper_id", "")
            item["evidence_ref_id"] = evidence_ref_id_by_paper.get(paper_id, "")
        filled.append(c)
    return filled


def build_balance_summary(
    selected_claims: Sequence[Dict[str, Any]],
    selected_paper_ids: Sequence[str],
    cards: Dict[str, Dict[str, Any]],
    all_claim_candidates: Sequence[Dict[str, Any]],
    claim_stats: Dict[str, Any],
    global_citation_distribution: Dict[str, Any],
) -> Dict[str, Any]:
    total_claims = len(all_claim_candidates)
    total_multi = sum(1 for c in all_claim_candidates if c.get("citation_mode") == "group")
    selected_total = len(selected_claims)
    selected_multi = sum(1 for c in selected_claims if c.get("citation_mode") == "group")
    selected_group_dist = Counter(group_size_bin(int(c.get("citation_group_size", 0))) for c in selected_claims)
    all_scope_dist = Counter(c.get("citation_scope", "unknown") for c in all_claim_candidates)
    selected_scope_dist = Counter(c.get("citation_scope", "unknown") for c in selected_claims)
    all_type_dist = Counter(c.get("claim_type", "other") for c in all_claim_candidates)
    selected_type_dist = Counter(c.get("claim_type", "other") for c in selected_claims)
    selected_other = selected_type_dist.get("other", 0)
    section_dist = Counter(c.get("section_title", "Untitled") for c in selected_claims)
    paper_use: Dict[str, Dict[str, Any]] = {}
    for claim in selected_claims:
        for paper_id in citation_group_paper_ids(claim):
            info = paper_use.setdefault(paper_id, {"sections": set(), "count": 0})
            info["count"] += 1
            info["sections"].add(claim.get("section_title", "Untitled"))
    top = []
    for paper_id, info in paper_use.items():
        top.append(
            {
                "paper_id": paper_id,
                "title": cards.get(paper_id, {}).get("title", ""),
                "num_selected_claim_contexts": info["count"],
                "num_sections_used": len(info["sections"]),
            }
        )
    top.sort(key=lambda x: (-x["num_selected_claim_contexts"], x["paper_id"]))
    balance_fields = claim_stats_balance_fields(claim_stats)
    return {
        "record_type": "balance_summary",
        "global_citation_distribution": global_citation_distribution,
        "total_cited_claim_candidates": len(all_claim_candidates),
        "total_single_citation_claims": sum(1 for c in all_claim_candidates if c.get("citation_mode") == "single"),
        "total_multi_citation_claims": total_multi,
        "multi_citation_claim_ratio_all": round(total_multi / max(1, total_claims), 4),
        "single_citation_claim_ratio_all": balance_fields["single_citation_claim_ratio_all"],
        "large_citation_group_ratio_all": balance_fields["large_citation_group_ratio_all"],
        "oversized_citation_group_ratio_all": balance_fields["oversized_citation_group_ratio_all"],
        "avg_citation_group_size_all": balance_fields["avg_citation_group_size_all"],
        "max_citation_group_size_observed": balance_fields["max_citation_group_size_observed"],
        "oversized_citation_group_claims_skipped": claim_stats.get("oversized_citation_group_claims_skipped", 0),
        "standalone_citation_sentences_attached": claim_stats.get("standalone_citation_sentences_attached", 0),
        "citation_only_sentences_skipped": claim_stats.get("citation_only_sentences_skipped", 0),
        "reference_entry_lines_skipped": claim_stats.get("reference_entry_lines_skipped", 0),
        "low_information_claims_skipped": claim_stats.get("low_information_claims_skipped", 0),
        "selected_claims": selected_total,
        "selected_single_citation_claims": sum(1 for c in selected_claims if c.get("citation_mode") == "single"),
        "selected_multi_citation_claims": selected_multi,
        "multi_citation_claim_ratio_selected": round(selected_multi / max(1, selected_total), 4),
        "num_unique_selected_papers": len(selected_paper_ids),
        "citation_group_size_distribution_all": claim_stats.get("citation_group_size_distribution_all", {}),
        "citation_group_size_distribution_selected": dict(sorted(selected_group_dist.items())),
        "claim_scope_distribution_all": dict(sorted(all_scope_dist.items())),
        "claim_scope_distribution_selected": dict(sorted(selected_scope_dist.items())),
        "num_clause_scope_claims": selected_scope_dist.get("clause", 0),
        "num_sentence_fallback_claims": selected_scope_dist.get("sentence_fallback", 0),
        "claim_type_distribution_all": dict(sorted(all_type_dist.items())),
        "claim_type_distribution_selected": dict(sorted(selected_type_dist.items())),
        "selected_other_claims": selected_other,
        "selected_other_claim_ratio": round(selected_other / max(1, selected_total), 4),
        "contexts_per_section_selected": dict(sorted(section_dist.items())),
        "top_reused_papers_selected": top[:20],
        "evidence_tier_counts_selected_papers": dict(sorted(Counter(evidence_tier_name(cards[pid]) for pid in selected_paper_ids).items())),
        "selection_notes": [
            "Claim-centric BSC-v2 uses citation closure: every selected claim includes its full citation group.",
            "Citation groups larger than max_group_size are counted but excluded from the main evaluated claim sample.",
        ],
    }


def build_bsc_jsonl_records(
    method: str,
    topic_id: str,
    diagnostics: Dict[str, Any],
    selected_claims: Sequence[Dict[str, Any]],
    selected_paper_ids: Sequence[str],
    citation_index_by_paper: Dict[str, List[str]],
    cards: Dict[str, Dict[str, Any]],
    claim_stats: Dict[str, Any],
    selection_stats: Dict[str, Any],
    global_citation_distribution: Dict[str, Any],
    max_papers: int,
    max_claim_records: int,
    char_limit: int,
) -> List[Dict[str, Any]]:
    single_selected = sum(1 for c in selected_claims if c.get("citation_mode") == "single")
    multi_selected = sum(1 for c in selected_claims if c.get("citation_mode") == "group")
    total_claims = diagnostics.get("num_claim_candidates", 0)
    total_multi = diagnostics.get("num_multi_claim_candidates", 0)
    meta = {
        "record_type": "survey_meta",
        "method": method,
        "topic_id": topic_id,
        "metric": "BSC-v2",
        "evaluation_unit": "claim_context",
        "num_ref_entries": diagnostics.get("num_ref_entries", 0),
        "num_refs_after_year_filter": diagnostics.get("num_refs_after_year_filter", 0),
        "num_evidence_cards_built": diagnostics.get("num_evidence_cards_built", 0),
        "num_claim_candidates": diagnostics.get("num_claim_candidates", 0),
        "num_selected_claims": len(selected_claims),
        "num_selected_single_claims": single_selected,
        "num_selected_multi_claims": multi_selected,
        "multi_citation_claim_ratio_all": round(total_multi / max(1, total_claims), 4),
        "multi_citation_claim_ratio_selected": round(multi_selected / max(1, len(selected_claims)), 4),
        "claim_scope_distribution_all": diagnostics.get("claim_scope_distribution_all", {}),
        "claim_scope_distribution_selected": diagnostics.get("claim_scope_distribution_selected", {}),
        "num_clause_scope_claims": diagnostics.get("num_clause_scope_claims", 0),
        "num_sentence_fallback_claims": diagnostics.get("num_sentence_fallback_claims", 0),
        "num_selected_papers": len(selected_paper_ids),
        "num_global_cited_claims": global_citation_distribution.get("total_cited_claims", 0),
        "num_global_unique_citation_indices": global_citation_distribution.get("num_unique_citation_indices", 0),
        "global_section_citation_coverage_ratio": global_citation_distribution.get("section_citation_coverage_ratio", 0.0),
        "max_papers": max_papers,
        "max_claim_records": max_claim_records,
        "jsonl_char_limit": char_limit,
        "truncated_to_char_limit": False,
        "sampling_policy": {
            "max_claim_records": max_claim_records,
            "max_unique_papers": max_papers,
            "target_single_citation_claims": selection_stats.get("max_single_claims", 50),
            "max_multi_citation_claims": selection_stats.get("max_multi_claims", 50),
            "single_backfill_when_multi_insufficient": True,
            "max_group_size": selection_stats.get("max_group_size", 15),
            "selection_strategy": "budgeted claim-centric sampling with citation-closure evidence bank",
        },
    }
    evidence_ref_id_by_paper, bank_record = build_evidence_card_bank(selected_paper_ids, citation_index_by_paper, cards)
    filled_claims = fill_claim_evidence_ref_ids(selected_claims, evidence_ref_id_by_paper, cards)
    balance = build_balance_summary(
        selected_claims,
        selected_paper_ids,
        cards,
        diagnostics.get("_claim_candidates", []),
        claim_stats,
        global_citation_distribution,
    )
    records: List[Dict[str, Any]] = [meta, bank_record]
    records.extend(filled_claims)
    records.append(balance)
    validate_claim_records(records, max_papers=max_papers, max_claim_records=max_claim_records, max_group_size=selection_stats.get("max_group_size", 15), max_single_claims=selection_stats.get("max_single_claims", 50), max_multi_claims=selection_stats.get("max_multi_claims", 50))
    return records


def serialize_jsonl(records: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in records) + "\n"


def selected_ids_from_records(records: Sequence[Dict[str, Any]]) -> List[str]:
    for record in records:
        if record.get("record_type") == "evidence_card_bank":
            return [x.get("paper_id", "") for x in record.get("papers", []) if x.get("paper_id")]
    ids = set()
    for record in records:
        if record.get("record_type") == "bsc_claim_context":
            ids.update(citation_group_paper_ids(record))
    return sorted(ids)


def validate_claim_records(records: Sequence[Dict[str, Any]], max_papers: int, max_claim_records: int, max_group_size: int, max_single_claims: int, max_multi_claims: int) -> None:
    banks = [r for r in records if r.get("record_type") == "evidence_card_bank"]
    if len(banks) != 1:
        raise BSCError("JSONL must contain exactly one evidence_card_bank record")
    bank_ids = {x.get("paper_id") for x in banks[0].get("papers", []) if x.get("paper_id")}
    claims = [r for r in records if r.get("record_type") == "bsc_claim_context"]
    used = set()
    if len(bank_ids) > max_papers:
        raise BSCError(f"Evidence bank exceeds max_papers: {len(bank_ids)} > {max_papers}")
    if len(claims) > max_claim_records:
        raise BSCError(f"Claim records exceed max_claim_records: {len(claims)} > {max_claim_records}")
    single_claims = sum(1 for c in claims if c.get("citation_mode") == "single")
    multi_claims = sum(1 for c in claims if c.get("citation_mode") == "group")
    if single_claims + multi_claims != len(claims):
        raise BSCError("Selected claims contain invalid citation_mode values")
    if multi_claims > max_multi_claims:
        raise BSCError("Selected multi-citation claims exceed quota")
    forbidden_claim_fields = {
        "paragraph_context",
        "claim_sentence",
        "section_path",
        "subsection_title",
        "selection_reason",
        "citation_evidence_preview",
    }
    for claim in claims:
        present_forbidden = forbidden_claim_fields.intersection(claim)
        if present_forbidden:
            raise BSCError(f"Claim {claim.get('claim_id')} contains redundant fields: {sorted(present_forbidden)}")
        if claim.get("citation_scope") not in {"clause", "sentence_fallback"}:
            raise BSCError(f"Invalid citation_scope in {claim.get('claim_id')}: {claim.get('citation_scope')}")
        if "[CITE_THIS]" not in claim.get("marked_sentence", ""):
            raise BSCError(f"Missing [CITE_THIS] in {claim.get('claim_id')}")
        if not has_evaluable_claim_text(claim.get("marked_sentence", "")):
            raise BSCError(f"Claim lacks evaluable text in {claim.get('claim_id')}")
        if not claim.get("original_sentence_with_citations"):
            raise BSCError(f"Missing original_sentence_with_citations in {claim.get('claim_id')}")
        group = claim.get("citation_group", [])
        if not group:
            raise BSCError(f"Empty citation group in {claim.get('claim_id')}")
        if claim.get("citation_group_size") != len(group):
            raise BSCError(f"Citation group size mismatch in {claim.get('claim_id')}")
        if len(group) > max_group_size:
            raise BSCError(f"Oversized selected claim {claim.get('claim_id')}")
        for item in group:
            paper_id = item.get("paper_id")
            used.add(paper_id)
            if paper_id not in bank_ids:
                raise BSCError(f"Claim {claim.get('claim_id')} references paper missing from evidence bank: {paper_id}")
            if not item.get("evidence_ref_id"):
                raise BSCError(f"Missing evidence_ref_id in {claim.get('claim_id')}")
        completeness = claim.get("evidence_completeness", {})
        if completeness.get("tier") == "has_missing_or_insufficient" or completeness.get("missing", 0) or completeness.get("insufficient", 0):
            raise BSCError(f"Selected claim has missing/insufficient evidence: {claim.get('claim_id')}")
    if used != bank_ids:
        raise BSCError(f"Evidence bank contains unused papers: {sorted(bank_ids - used)[:5]}")


def _claim_removal_key(claim: Dict[str, Any]) -> Tuple[Any, ...]:
    tier_order = {
        "all_metadata_and_abstract": 0,
        "mixed_with_metadata_only": 1,
        "mixed_with_abstract_only": 2,
        "title_only_or_weak": 3,
        "has_missing_or_insufficient": 4,
    }
    completeness = claim.get("evidence_completeness", {})
    tier = tier_order.get(completeness.get("tier"), 4)
    weak_type = claim.get("claim_type") in {"background", "other"}
    return (tier, int(claim.get("citation_group_size", 0)), weak_type, int(claim.get("position", 10**9)))


def _rebuild_records_from_claims(records: List[Dict[str, Any]], claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    meta = dict(records[0])
    old_bank = next(r for r in records if r.get("record_type") == "evidence_card_bank")
    old_papers = {p["paper_id"]: p for p in old_bank.get("papers", [])}
    selected_ids = sorted({pid for c in claims for pid in citation_group_paper_ids(c)})
    ref_ids = {pid: f"E{i:03d}" for i, pid in enumerate(selected_ids, start=1)}
    papers = []
    for pid in selected_ids:
        paper = dict(old_papers[pid])
        paper["evidence_ref_id"] = ref_ids[pid]
        papers.append(paper)
    for claim in claims:
        for item in claim.get("citation_group", []):
            item["evidence_ref_id"] = ref_ids.get(item.get("paper_id"), "")
    meta["num_selected_claims"] = len(claims)
    meta["num_selected_single_claims"] = sum(1 for c in claims if c.get("citation_mode") == "single")
    meta["num_selected_multi_claims"] = sum(1 for c in claims if c.get("citation_mode") == "group")
    meta["multi_citation_claim_ratio_selected"] = round(
        meta["num_selected_multi_claims"] / max(1, meta["num_selected_claims"]), 4
    )
    meta["num_selected_papers"] = len(selected_ids)
    meta["truncated_to_char_limit"] = True
    bank_record = {"record_type": "evidence_card_bank", "num_papers": len(papers), "papers": papers}
    # Lightweight balance reconstruction for truncation.
    balance = next((r for r in records if r.get("record_type") == "balance_summary"), {"record_type": "balance_summary"})
    balance = dict(balance)
    balance["selected_claims"] = len(claims)
    balance["selected_single_citation_claims"] = sum(1 for c in claims if c.get("citation_mode") == "single")
    balance["selected_multi_citation_claims"] = sum(1 for c in claims if c.get("citation_mode") == "group")
    balance["multi_citation_claim_ratio_selected"] = round(
        balance["selected_multi_citation_claims"] / max(1, balance["selected_claims"]), 4
    )
    balance["num_unique_selected_papers"] = len(selected_ids)
    balance["citation_group_size_distribution_selected"] = dict(Counter(group_size_bin(int(c.get("citation_group_size", 0))) for c in claims))
    balance["claim_type_distribution_selected"] = dict(Counter(c.get("claim_type", "other") for c in claims))
    balance["selected_other_claims"] = balance["claim_type_distribution_selected"].get("other", 0)
    balance["selected_other_claim_ratio"] = round(balance["selected_other_claims"] / max(1, balance["selected_claims"]), 4)
    balance["contexts_per_section_selected"] = dict(Counter(c.get("section_title", "Untitled") for c in claims))
    return [meta, bank_record] + claims + [balance]


def write_jsonl_with_char_limit(
    records: List[Dict[str, Any]], output_path: Path, char_limit: int
) -> Tuple[int, bool, List[Dict[str, Any]]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    working = json.loads(json.dumps(records, ensure_ascii=False))
    truncated = False
    while True:
        text = serialize_jsonl(working)
        if len(text) <= char_limit:
            output_path.write_text(text, encoding="utf-8")
            return len(text), truncated, working
        truncated = True
        claims = [r for r in working if r.get("record_type") == "bsc_claim_context"]
        if not claims:
            raise BSCError(f"Cannot fit JSONL within char limit {char_limit}")
        remove = max(claims, key=_claim_removal_key)
        claims = [c for c in claims if c.get("claim_id") != remove.get("claim_id")]
        working = _rebuild_records_from_claims(working, claims)

def build_bsc_prompt(jsonl_text: str) -> str:
    return BSC_JUDGE_PROMPT + "\n" + jsonl_text


def _openai_client(api_key: str, base_url: str, timeout: int):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    choices = getattr(response, "choices", None)
    if choices:
        message = choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    output = getattr(response, "output", None)
    if output:
        chunks = []
        for item in output:
            for content in getattr(item, "content", []) or []:
                t = getattr(content, "text", None)
                if t:
                    chunks.append(t)
        if chunks:
            return "\n".join(chunks)
    return str(response)


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(read_text(path))


def selected_api_config(config: Dict[str, Any], mode_override: str = "on") -> Dict[str, Any]:
    llm = config.get("llm_api", {})
    if mode_override == "config":
        mode = llm.get("api_mode", "off")
    else:
        mode = mode_override
    key = "api_on" if mode == "on" else "api_off"
    cfg = dict(llm.get(key, {}))
    cfg["api_mode"] = mode
    cfg["api_config_key"] = key
    return cfg


def _extract_response_text_from_dict(data: Dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"])
    choices = data.get("choices")
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict):
                    chunks.append(str(part.get("text", part)))
                else:
                    chunks.append(str(part))
            return "\n".join(chunks)
    output = data.get("output")
    if output:
        chunks = []
        for item in output:
            for content in item.get("content", []) or []:
                if isinstance(content, dict):
                    text = content.get("text")
                    if text:
                        chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)
    return json.dumps(data, ensure_ascii=False)


def _http_post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: int) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise BSCError(f"HTTP {exc.code} from LLM API: {raw}") from exc
    return json.loads(raw)


def call_llm_judge(prompt: str, config: Dict[str, Any], api_mode_override: str = "off", model_name_override: str = "") -> Tuple[str, str]:
    api_cfg = selected_api_config(config, api_mode_override)
    if model_name_override:
        api_cfg["model_name"] = model_name_override
    provider = api_cfg.get("provider", "openai_compatible_chat")
    api_key = os.getenv(api_cfg.get("api_key_env", ""), api_cfg.get("api_key", "EMPTY"))
    model = api_cfg.get("model_name", "")
    base_url = str(api_cfg.get("base_url", "")).rstrip("/")
    timeout = int(api_cfg.get("timeout", 600))
    retries = int(api_cfg.get("retries", 1))
    sleep_base = float(api_cfg.get("retry_sleep_base_seconds", 5))
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            if provider == "openai_responses":
                payload = {
                    "model": model,
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                    "temperature": float(api_cfg.get("temperature", 0.2)),
                    "max_output_tokens": int(api_cfg.get("max_output_tokens", api_cfg.get("max_tokens", 4096))),
                }
                data = _http_post_json(f"{base_url}/responses", payload, api_key, timeout)
            else:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": float(api_cfg.get("temperature", 0.2)),
                    "max_tokens": int(api_cfg.get("max_tokens", api_cfg.get("max_output_tokens", 4096))),
                }
                if "enable_thinking" in api_cfg:
                    payload["chat_template_kwargs"] = {"enable_thinking": bool(api_cfg.get("enable_thinking"))}
                data = _http_post_json(f"{base_url}/chat/completions", payload, api_key, timeout)
            return _extract_response_text_from_dict(data), model
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(sleep_base * attempt)
    raise BSCError(f"LLM judge failed after {retries} attempts: {last_exc}")


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_llm_json_response(response: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text), None
    except Exception as exc1:
        obj = extract_first_json_object(response)
        if obj:
            try:
                return json.loads(obj), None
            except Exception as exc2:
                return None, f"JSON parse failed: {exc1}; extracted object parse failed: {exc2}"
        return None, f"JSON parse failed: {exc1}"


def validate_scores(parsed: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    if "scores" not in parsed and isinstance(parsed.get("BSC"), dict):
        parsed["scores"] = {}
        for dim in DIMENSIONS:
            item = parsed["BSC"].get(dim, {})
            if not item and dim == "Citation Distribution Balance and Non-Redundancy":
                item = parsed["BSC"].get("Citation Evidence Balance and Non-Redundancy", {})
            parsed["scores"][dim] = item if isinstance(item, dict) else {}
        warnings.append("Accepted LLM output schema 'BSC' and normalized it to 'scores'")
    scores = parsed.setdefault("scores", {})
    total = 0
    for dim in DIMENSIONS:
        item = scores.setdefault(dim, {})
        score = item.get("score", 1)
        if not isinstance(score, int):
            try:
                score = int(score)
            except Exception:
                warnings.append(f"Invalid score for {dim}; set to 1")
                score = 1
        if score < 1 or score > 5:
            warnings.append(f"Out-of-range score for {dim}: {score}; clipped")
            score = max(1, min(5, score))
        item["score"] = score
        item.setdefault("rationale", "")
        if "strong_examples" in item:
            item["strong_examples"] = as_list(item.get("strong_examples", []))[:5]
        if "weak_examples" in item:
            item["weak_examples"] = as_list(item.get("weak_examples", []))[:5]
        total += score
    if parsed.get("bsc_raw_20") != total:
        warnings.append(f"bsc_raw_20 recomputed from {parsed.get('bsc_raw_20')} to {total}")
        parsed["bsc_raw_20"] = total
    parsed.setdefault("diagnostics", {})
    if "overall_rationale" not in parsed and "overall_assessment" in parsed:
        parsed["overall_rationale"] = parsed.get("overall_assessment", "")
    parsed.setdefault("overall_rationale", "")
    return parsed, warnings


def ref_path_for(args: argparse.Namespace, topic_id: str) -> Path:
    return Path(args.ref_input_root) / args.method / f"ref_{topic_id}.json"


def md_path_for(args: argparse.Namespace, topic_id: str) -> Path:
    return Path(args.md_cache_root) / args.method / "md" / f"{topic_id}.md"


def jsonl_path_for(args: argparse.Namespace, topic_id: str) -> Path:
    return Path(args.md_cache_root) / args.method / "ref" / f"{topic_id}.jsonl"


def api_result_dir_name(api_mode: str) -> str:
    mode = str(api_mode or "off")
    if mode not in {"on", "off"}:
        mode = "off"
    return f"api_{mode}"


def result_path_for(args: argparse.Namespace, topic_id: str) -> Path:
    return Path(args.results_root) / args.method / "bsc" / api_result_dir_name(args.api_mode) / f"{topic_id}.json"


def topic_ids_for_all(args: argparse.Namespace) -> List[str]:
    root = Path(args.ref_input_root) / args.method
    ids = []
    for p in sorted(root.glob("ref_*.json")):
        m = re.fullmatch(r"ref_(\d{3})\.json", p.name)
        if m:
            ids.append(m.group(1))
    return ids


def build_topic_jsonl(args: argparse.Namespace, topic_id: str) -> Tuple[Path, Dict[str, Any], List[Dict[str, Any]]]:
    ref_path = ref_path_for(args, topic_id)
    md_path = md_path_for(args, topic_id)
    jsonl_path = jsonl_path_for(args, topic_id)
    if not ref_path.is_file():
        raise BSCError(f"Missing ref file: {ref_path}")
    if not md_path.is_file():
        raise BSCError(f"Missing markdown file: {md_path}")
    if jsonl_path.exists() and not args.overwrite:
        raise BSCError(f"JSONL exists; pass --overwrite: {jsonl_path}")

    ref_map = load_ref_map(ref_path)
    filtered_ref_map, year_stats = filter_ref_map_by_year(ref_map)
    citation_index_by_paper: Dict[str, List[str]] = defaultdict(list)
    for idx, paper_id in filtered_ref_map.items():
        citation_index_by_paper[paper_id].append(idx)
    for idxs in citation_index_by_paper.values():
        idxs.sort(key=lambda x: int(x) if x.isdigit() else x)

    metadata_root = Path(args.metadata_root)
    mineru_root = Path(args.mineru_root)
    cards: Dict[str, Dict[str, Any]] = {}
    missing_metadata = 0
    metadata_load_failed = 0
    missing_abstract = 0
    missing_mineru_md = 0
    missing_title = 0
    skipped_no_evidence = 0

    for paper_id in sorted(set(filtered_ref_map.values())):
        metadata, metadata_path = load_metadata(paper_id, metadata_root)
        has_metadata = metadata is not None
        if metadata is None:
            if metadata_path is None:
                missing_metadata += 1
            else:
                metadata_load_failed += 1
            metadata = {}
        md_candidate = resolve_mineru_md_path(paper_id, metadata_path, metadata, mineru_root)
        abstract = ""
        title_from_md = ""
        if md_candidate and md_candidate.is_file():
            md_text = read_text(md_candidate)
            abstract = extract_abstract_from_md(md_text, int(args.abstract_max_chars))
            title_from_md = extract_title_from_md(md_text)
        else:
            missing_mineru_md += 1
        if not abstract:
            missing_abstract += 1
        card = build_evidence_card(
            paper_id,
            metadata,
            abstract,
            title_override=title_from_md,
            has_metadata=has_metadata,
        )
        if not card.get("_has_title"):
            missing_title += 1
        if evidence_tier_rank(card) >= 4:
            skipped_no_evidence += 1
            continue
        cards[paper_id] = card

    survey_md = read_text(md_path)
    global_citation_distribution = extract_global_citation_distribution(survey_md, method=args.method)
    claim_candidates, claim_stats = extract_claim_contexts(
        survey_md,
        filtered_ref_map,
        cards.keys(),
        topic_id,
        max_group_size=int(args.max_group_size),
        method=args.method,
    )
    selected_claims, selected_paper_ids, selection_stats = select_claims_for_bsc(
        claim_candidates,
        cards,
        max_claim_records=int(args.max_claim_records),
        max_unique_papers=int(args.max_papers),
        max_single_claims=int(args.max_single_claims),
        max_multi_claims=int(args.max_multi_claims),
    )
    selection_stats["max_single_claims"] = int(args.max_single_claims)
    selection_stats["max_multi_claims"] = int(args.max_multi_claims)
    selection_stats["max_group_size"] = int(args.max_group_size)

    group_size_all = claim_stats.get("citation_group_size_distribution_all", {})
    group_size_selected = dict(Counter(group_size_bin(int(c.get("citation_group_size", 0))) for c in selected_claims))
    claim_scope_distribution_all = dict(sorted(Counter(c.get("citation_scope", "unknown") for c in claim_candidates).items()))
    claim_scope_distribution_selected = dict(sorted(Counter(c.get("citation_scope", "unknown") for c in selected_claims).items()))
    claim_type_distribution_all = dict(sorted(Counter(c.get("claim_type", "other") for c in claim_candidates).items()))
    claim_type_distribution_selected = dict(sorted(Counter(c.get("claim_type", "other") for c in selected_claims).items()))
    selected_other_claims = claim_type_distribution_selected.get("other", 0)
    balance_fields_all = claim_stats_balance_fields(claim_stats)
    total_claim_candidates = len(claim_candidates)
    total_multi_claim_candidates = sum(1 for c in claim_candidates if c.get("citation_mode") == "group")
    selected_multi_claims = sum(1 for c in selected_claims if c.get("citation_mode") == "group")
    evidence_tier_counts = Counter(evidence_tier_name(card) for card in cards.values())
    selected_evidence_tier_counts = Counter(evidence_tier_name(cards[pid]) for pid in selected_paper_ids)

    diagnostics = {
        "num_ref_entries": len(ref_map),
        "num_refs_after_year_filter": len(filtered_ref_map),
        "num_unique_refs_after_year_filter": len(set(filtered_ref_map.values())),
        "num_evidence_cards_built": len(cards),
        "num_claim_candidates": total_claim_candidates,
        "num_single_claim_candidates": sum(1 for c in claim_candidates if c.get("citation_mode") == "single"),
        "num_multi_claim_candidates": total_multi_claim_candidates,
        "multi_citation_claim_ratio_all": round(total_multi_claim_candidates / max(1, total_claim_candidates), 4),
        "single_citation_claim_ratio_all": balance_fields_all["single_citation_claim_ratio_all"],
        "large_citation_group_ratio_all": balance_fields_all["large_citation_group_ratio_all"],
        "oversized_citation_group_ratio_all": balance_fields_all["oversized_citation_group_ratio_all"],
        "avg_citation_group_size_all": balance_fields_all["avg_citation_group_size_all"],
        "max_citation_group_size_observed": balance_fields_all["max_citation_group_size_observed"],
        "multi_citation_claim_ratio_selected": round(selected_multi_claims / max(1, len(selected_claims)), 4),
        "num_selected_claims_considered": len(selected_claims),
        "num_selected_single_claims": sum(1 for c in selected_claims if c.get("citation_mode") == "single"),
        "num_selected_multi_claims": selected_multi_claims,
        "num_selected_papers_considered": len(selected_paper_ids),
        "num_contexts_considered": len(selected_claims),
        "claim_group_size_distribution_all": group_size_all,
        "claim_group_size_distribution_selected": group_size_selected,
        "claim_scope_distribution_all": claim_scope_distribution_all,
        "claim_scope_distribution_selected": claim_scope_distribution_selected,
        "num_clause_scope_claims": claim_scope_distribution_selected.get("clause", 0),
        "num_sentence_fallback_claims": claim_scope_distribution_selected.get("sentence_fallback", 0),
        "claim_type_distribution_all": claim_type_distribution_all,
        "claim_type_distribution_selected": claim_type_distribution_selected,
        "selected_other_claims": selected_other_claims,
        "selected_other_claim_ratio": round(selected_other_claims / max(1, len(selected_claims)), 4),
        "num_oversized_group_claims_skipped": claim_stats.get("oversized_citation_group_claims_skipped", 0),
        "selection_stats": selection_stats,
        "num_missing_metadata": missing_metadata,
        "num_metadata_load_failed": metadata_load_failed,
        "num_missing_mineru_md": missing_mineru_md,
        "num_missing_abstract": missing_abstract,
        "num_missing_title": missing_title,
        "num_skipped_no_evidence": skipped_no_evidence,
        "evidence_tier_counts": dict(sorted(evidence_tier_counts.items())),
        "selected_evidence_tier_counts": dict(sorted(selected_evidence_tier_counts.items())),
        "num_table_citations_skipped": claim_stats.get("table_citations_skipped", 0),
        "num_code_citations_skipped": claim_stats.get("code_citations_skipped", 0),
        "num_caption_citations_skipped": claim_stats.get("caption_citations_skipped", 0),
        "num_gemini_bare_citations_detected": claim_stats.get("gemini_bare_citations_detected", 0),
        "num_standalone_citation_sentences_attached": claim_stats.get("standalone_citation_sentences_attached", 0),
        "num_citation_only_sentences_skipped": claim_stats.get("citation_only_sentences_skipped", 0),
        "num_reference_entry_lines_skipped": claim_stats.get("reference_entry_lines_skipped", 0),
        "num_low_information_claims_skipped": claim_stats.get("low_information_claims_skipped", 0),
        "year_filter_stats": year_stats,
        "global_citation_distribution": global_citation_distribution,
        "jsonl_chars": 0,
        "truncated_to_char_limit": False,
        "_claim_candidates": claim_candidates,
    }

    records = build_bsc_jsonl_records(
        args.method,
        topic_id,
        diagnostics,
        selected_claims,
        selected_paper_ids,
        dict(citation_index_by_paper),
        cards,
        claim_stats,
        selection_stats,
        global_citation_distribution,
        int(args.max_papers),
        int(args.max_claim_records),
        int(args.jsonl_char_limit),
    )
    diagnostics.pop("_claim_candidates", None)
    jsonl_chars, truncated, final_records = write_jsonl_with_char_limit(records, jsonl_path, int(args.jsonl_char_limit))
    diagnostics["jsonl_chars"] = jsonl_chars
    diagnostics["truncated_to_char_limit"] = truncated
    final_claims = [r for r in final_records if r.get("record_type") == "bsc_claim_context"]
    final_bank = next((r for r in final_records if r.get("record_type") == "evidence_card_bank"), {"papers": []})
    diagnostics["num_selected_claims_considered"] = len(final_claims)
    diagnostics["num_selected_single_claims"] = sum(1 for c in final_claims if c.get("citation_mode") == "single")
    diagnostics["num_selected_multi_claims"] = sum(1 for c in final_claims if c.get("citation_mode") == "group")
    diagnostics["multi_citation_claim_ratio_selected"] = round(
        diagnostics["num_selected_multi_claims"] / max(1, len(final_claims)), 4
    )
    diagnostics["num_selected_papers_considered"] = len(final_bank.get("papers", []))
    diagnostics["num_contexts_considered"] = len(final_claims)
    diagnostics["claim_group_size_distribution_selected"] = dict(Counter(group_size_bin(int(c.get("citation_group_size", 0))) for c in final_claims))
    diagnostics["claim_scope_distribution_selected"] = dict(sorted(Counter(c.get("citation_scope", "unknown") for c in final_claims).items()))
    diagnostics["num_clause_scope_claims"] = diagnostics["claim_scope_distribution_selected"].get("clause", 0)
    diagnostics["num_sentence_fallback_claims"] = diagnostics["claim_scope_distribution_selected"].get("sentence_fallback", 0)
    diagnostics["claim_type_distribution_selected"] = dict(sorted(Counter(c.get("claim_type", "other") for c in final_claims).items()))
    diagnostics["selected_other_claims"] = diagnostics["claim_type_distribution_selected"].get("other", 0)
    diagnostics["selected_other_claim_ratio"] = round(diagnostics["selected_other_claims"] / max(1, len(final_claims)), 4)
    diagnostics["selected_evidence_tier_counts"] = dict(Counter(p.get("evidence_tier", "") for p in final_bank.get("papers", [])))
    return jsonl_path, diagnostics, final_records

def make_result_base(args: argparse.Namespace, topic_id: str, diagnostics: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
    return {
        "method": args.method,
        "topic_id": topic_id,
        "metric": "BSC-v2",
        "evaluated_at": now_iso(),
        "input_ref_path": str(ref_path_for(args, topic_id)),
        "input_md_path": str(md_path_for(args, topic_id)),
        "cache_jsonl_path": str(jsonl_path_for(args, topic_id)),
        "model_name": model_name,
        "api_mode": args.api_mode,
        "scores": {},
        "bsc_raw_20": None,
        "diagnostics": diagnostics,
        "overall_rationale": "",
        "warnings": [],
        "status": "unknown",
    }


def evaluate_topic(args: argparse.Namespace, topic_id: str) -> Dict[str, Any]:
    log_path = Path(args.md_cache_root) / args.method / "logs" / "bsc_log.txt"
    result_path = result_path_for(args, topic_id)
    status = "failed"
    diagnostics: Dict[str, Any] = {}
    try:
        jsonl_path, diagnostics, _records = build_topic_jsonl(args, topic_id)
        if args.build_only:
            status = "build_only_success"
            result = make_result_base(args, topic_id, diagnostics)
            result["status"] = status
            return result

        config = load_config(Path(args.config))
        jsonl_text = read_text(jsonl_path)
        raw_response, model_name = call_llm_judge(build_bsc_prompt(jsonl_text), config, args.api_mode, args.model_name)
        parsed, parse_error = parse_llm_json_response(raw_response)
        result = make_result_base(args, topic_id, diagnostics, model_name=model_name)
        if parsed is None:
            result.update(
                {
                    "status": "parse_error",
                    "parse_error": parse_error,
                    "raw_response": raw_response,
                }
            )
            write_json(result_path, result)
            status = "parse_error"
            return result

        parsed, warnings = validate_scores(parsed)
        llm_diag = parsed.get("diagnostics", {}) if isinstance(parsed.get("diagnostics", {}), dict) else {}
        merged_diag = dict(diagnostics)
        for key, value in llm_diag.items():
            merged_diag[key] = value
        result.update(
            {
                "scores": parsed.get("scores", {}),
                "bsc_raw_20": parsed.get("bsc_raw_20"),
                "diagnostics": merged_diag,
                "overall_rationale": parsed.get("overall_rationale", ""),
                "warnings": warnings,
                "status": "success",
            }
        )
        write_json(result_path, result)
        status = "success"
        return result
    except Exception as exc:
        result = make_result_base(args, topic_id, diagnostics)
        result.update({"status": "failed", "error": str(exc)})
        if not args.build_only:
            write_json(result_path, result)
        raise
    finally:
        append_log(
            log_path,
            [
                f"time: {now_iso()}",
                f"method: {args.method}",
                f"topic_id: {topic_id}",
                f"ref_path: {ref_path_for(args, topic_id)}",
                f"md_path: {md_path_for(args, topic_id)}",
                f"num_ref_entries: {diagnostics.get('num_ref_entries', 0)}",
                f"num_refs_after_year_filter: {diagnostics.get('num_refs_after_year_filter', 0)}",
                f"metadata_found: {diagnostics.get('num_evidence_cards_built', 0)}",
                f"num_evidence_cards_built: {diagnostics.get('num_evidence_cards_built', 0)}",
                f"num_missing_abstract: {diagnostics.get('num_missing_abstract', 0)}",
                f"num_missing_title: {diagnostics.get('num_missing_title', 0)}",
                f"num_skipped_no_evidence: {diagnostics.get('num_skipped_no_evidence', 0)}",
                f"evidence_tier_counts: {diagnostics.get('evidence_tier_counts', {})}",
                f"selected_evidence_tier_counts: {diagnostics.get('selected_evidence_tier_counts', {})}",
                f"num_claim_candidates: {diagnostics.get('num_claim_candidates', 0)}",
                f"num_selected_claims: {diagnostics.get('num_selected_claims_considered', 0)}",
                f"num_selected_single_claims: {diagnostics.get('num_selected_single_claims', 0)}",
                f"num_selected_multi_claims: {diagnostics.get('num_selected_multi_claims', 0)}",
                f"multi_citation_claim_ratio_all: {diagnostics.get('multi_citation_claim_ratio_all', 0)}",
                f"multi_citation_claim_ratio_selected: {diagnostics.get('multi_citation_claim_ratio_selected', 0)}",
                f"claim_group_size_distribution_all: {diagnostics.get('claim_group_size_distribution_all', {})}",
                f"claim_group_size_distribution_selected: {diagnostics.get('claim_group_size_distribution_selected', {})}",
                f"claim_scope_distribution_all: {diagnostics.get('claim_scope_distribution_all', {})}",
                f"claim_scope_distribution_selected: {diagnostics.get('claim_scope_distribution_selected', {})}",
                f"num_clause_scope_claims: {diagnostics.get('num_clause_scope_claims', 0)}",
                f"num_sentence_fallback_claims: {diagnostics.get('num_sentence_fallback_claims', 0)}",
                f"claim_type_distribution_all: {diagnostics.get('claim_type_distribution_all', {})}",
                f"claim_type_distribution_selected: {diagnostics.get('claim_type_distribution_selected', {})}",
                f"selected_other_claims: {diagnostics.get('selected_other_claims', 0)}",
                f"selected_other_claim_ratio: {diagnostics.get('selected_other_claim_ratio', 0)}",
                f"num_oversized_group_claims_skipped: {diagnostics.get('num_oversized_group_claims_skipped', 0)}",
                f"selection_stats: {diagnostics.get('selection_stats', {})}",
                f"num_selected_papers: {diagnostics.get('num_selected_papers_considered', 0)}",
                f"selected_contexts: {diagnostics.get('num_contexts_considered', 0)}",
                f"num_table_citations_skipped: {diagnostics.get('num_table_citations_skipped', 0)}",
                f"jsonl_chars: {diagnostics.get('jsonl_chars', 0)}",
                f"truncated_to_char_limit: {diagnostics.get('truncated_to_char_limit', False)}",
                f"result_path: {result_path}",
                f"status: {status}",
            ],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate BSC citation quality inputs.")
    parser.add_argument("--method", default="submission")
    parser.add_argument("--topic-id", default=None, help="Single topic id, e.g. 001")
    parser.add_argument("--all", action="store_true", help="Evaluate all ref_*.json for the method")
    parser.add_argument("--ref-input-root", default=str(PROJECT_ROOT / "eval_inputs"))
    parser.add_argument("--md-cache-root", default=str(PROJECT_ROOT / "eval_cache"))
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--metadata-root", default=str(PROJECT_ROOT / "data" / "metadata"))
    parser.add_argument("--mineru-root", default=str(PROJECT_ROOT / "data" / "mineru"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.json"))
    parser.add_argument("--max-papers", type=int, default=100)
    parser.add_argument("--max-claim-records", type=int, default=100)
    parser.add_argument("--max-single-claims", type=int, default=50)
    parser.add_argument("--max-multi-claims", type=int, default=50)
    parser.add_argument("--max-group-size", type=int, default=15)
    parser.add_argument("--jsonl-char-limit", type=int, default=1_000_000)
    parser.add_argument("--abstract-max-chars", type=int, default=1200)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-name", default="", help="Optional model override. Empty uses the selected api config model.")
    parser.add_argument(
        "--api-mode",
        choices=["on", "off", "config"],
        default="off",
        help="LLM API mode for BSC judging. Default off uses local Qwen config; on uses config llm_api.api_on.",
    )
    args = parser.parse_args()
    if not args.all and not args.topic_id:
        parser.error("Specify --topic-id or --all")
    if args.all and args.topic_id:
        parser.error("Use either --topic-id or --all, not both")
    return args


def main() -> None:
    args = parse_args()
    if args.api_mode == "config":
        config_mode = str(load_config(Path(args.config)).get("llm_api", {}).get("api_mode", "off"))
        args.api_mode = config_mode if config_mode in {"on", "off"} else "off"
    topic_ids = topic_ids_for_all(args) if args.all else [args.topic_id]
    if not topic_ids:
        raise SystemExit(f"No topics found for method {args.method}")
    failures = 0
    for topic_id in topic_ids:
        try:
            result = evaluate_topic(args, topic_id)
            print(
                json.dumps(
                    {
                        "topic_id": topic_id,
                        "status": result.get("status"),
                        "jsonl": str(jsonl_path_for(args, topic_id)),
                        "result": str(result_path_for(args, topic_id)),
                        "diagnostics": result.get("diagnostics", {}),
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            failures += 1
            print(json.dumps({"topic_id": topic_id, "status": "failed", "error": str(exc)}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
