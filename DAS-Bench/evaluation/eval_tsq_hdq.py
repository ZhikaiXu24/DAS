#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation module for Taxonomic Synthesis Quality (TSQ) and
Hierarchical Drafting Quality (HDQ).

This script uses LLM judging over PDF inputs. It does not implement BSC or MAR.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from mimetypes import guess_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


METHODS = [
    "submission",
]

EVALUATION_TOPIC = "Memory and Long-Context Mechanisms for Long-Horizon LLM Agents"

TOPIC_TITLES = {
    "001": "Tool Learning and Function Calling for LLM Agents",
    "002": "Memory and Long-Context Mechanisms for Long-Horizon LLM Agents",
    "003": "Retrieval-Augmented Generation for Large Language Models",
    "004": "Planning and Self-Reflection in Large Language Model Reasoning",
    "005": "Prompt Injection and Tool-Use Security in LLM Agents",
    "006": "Program Repair and Automated Debugging with Code LLMs",
    "007": "Multimodal Retrieval-Augmented Generation for Chart and Document Understanding",
    "008": "Vision-Language Models for Embodied Reasoning",
    "009": "Diffusion and Flow-Based Models for Controllable Image Generation",
    "010": "Gaussian Splatting and Neural Rendering for Dynamic 3D Scene Reconstruction",
    "011": "Multi-Sensor Fusion for Autonomous Driving Perception",
    "012": "Continual Learning and Model Editing for Foundation Models",
    "013": "Offline and Preference-Based Reinforcement Learning for Robotics",
    "014": "Graph Neural Networks and Graph Foundation Models for Scientific Discovery",
    "015": "Efficient LLM Serving with KV Cache, Speculative Decoding, and Quantization",
    "016": "Vector Databases and Retrieval Systems for Large-Scale AI Applications",
    "017": "Privacy-Preserving Machine Learning with Federated Learning and Differential Privacy",
    "018": "AI Software Supply-Chain Security and Vulnerability Detection",
    "019": "Human-AI Collaboration in Scientific Writing and Research Workflows",
    "020": "Causal Representation Learning and Causal Discovery in Deep Learning",
    "021": "Large Language Models for Generative Recommendation and User Behavior Modeling",
    "022": "AI-Driven Protein Design with Diffusion and Language Models",
    "023": "Single-Cell Foundation Models for Cell Type Annotation and Perturbation Prediction",
    "024": "Radiomics and Deep Learning for Tumor Diagnosis and Prognosis",
    "025": "Machine Learning for Solid-State Battery Materials Discovery",
    "026": "Machine Learning for Electrocatalyst Discovery in Energy Conversion",
    "027": "Deep Learning for Extreme Weather Forecasting",
    "028": "Foundation Models for Satellite Earth Observation",
    "029": "Deep Learning for Financial Risk Modeling under Uncertainty",
    "030": "Bayesian Deep Learning for Uncertainty Quantification",
}

TOPIC_IDS = list(TOPIC_TITLES.keys())

TSQ_SUBMETRICS = [
    "Research-Space Coverage",
    "Taxonomy Clarity and Boundary Control",
    "Survey Organization and Functional Coherence",
    "Synthesis Insight and Gap Analysis",
]

HDQ_SUBMETRICS = [
    "Multi-Level Goal Alignment",
    "Paragraph Argument Progression",
    "Atomic Claim Specificity and Technical Concreteness",
    "Local Synthesis and Non-Enumerative Writing",
]

CSV_FIELDS = [
    "method",
    "topic_id",
    "topic",
    "api_mode",
    "input_type",
    "research_space_coverage",
    "taxonomy_clarity_and_boundary_control",
    "survey_organization_and_functional_coherence",
    "synthesis_insight_and_gap_analysis",
    "multi_level_goal_alignment",
    "paragraph_argument_progression",
    "atomic_claim_specificity_and_technical_concreteness",
    "local_synthesis_and_non_enumerative_writing",
    "tsq_raw_20",
    "hdq_raw_20",
    "json_path",
    "status",
]

CSV_SCORE_FIELD_BY_SUBMETRIC = {
    "Research-Space Coverage": "research_space_coverage",
    "Taxonomy Clarity and Boundary Control": "taxonomy_clarity_and_boundary_control",
    "Survey Organization and Functional Coherence": (
        "survey_organization_and_functional_coherence"
    ),
    "Synthesis Insight and Gap Analysis": "synthesis_insight_and_gap_analysis",
    "Multi-Level Goal Alignment": "multi_level_goal_alignment",
    "Paragraph Argument Progression": "paragraph_argument_progression",
    "Atomic Claim Specificity and Technical Concreteness": (
        "atomic_claim_specificity_and_technical_concreteness"
    ),
    "Local Synthesis and Non-Enumerative Writing": (
        "local_synthesis_and_non_enumerative_writing"
    ),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_root(root_arg: str) -> Path:
    root = Path(root_arg)
    if root_arg == "EVAL":
        return PROJECT_ROOT
    return root if root.is_absolute() else (Path.cwd() / root)

def resolve_data_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    parts = path.parts
    if parts and parts[0].lower() == root.name.lower():
        candidates.append(root.joinpath(*parts[1:]))

    candidates.extend([Path.cwd() / path, root / path, root.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if parts and parts[0].lower() == root.name.lower():
        return root.joinpath(*parts[1:])
    return root / path


def load_config(root: Path, config_path: str | None) -> dict[str, Any]:
    path = resolve_data_path(config_path, root) if config_path else root / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"config.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(root: Path) -> list[dict[str, str]]:
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_tsq_hdq_prompt(topic: str) -> str:
    return f"""You are an expert reviewer responsible for evaluating the quality of an anonymous academic survey.

You will be given one survey on the topic below. Evaluate only the final survey content. Do not infer the generating system and do not reward or penalize any method name.

Topic:
{topic}

Evaluate exactly two dimensions:
- Taxonomic Synthesis Quality (TSQ): global research-space coverage, taxonomy, organization, and high-level synthesis.
- Hierarchical Drafting Quality (HDQ): section, subsection, and paragraph-level alignment, progression, technical specificity, and local synthesis.

Do not evaluate citation correctness, citation-to-claim support, reference matching, publication-date validity, LaTeX compilation, figure/table formatting, or visual layout. The evaluation date is July 2026: references dated 2025 or 2026 are not future references. Never infer that a citation or technical claim is fabricated merely from its date, identifier, specificity, or unfamiliarity. Here, "verifiable claim" means linguistically specific enough to be checked; it does not ask you to verify truth. Do not reward document length, citation count, section count, fluent academic tone, or the mere presence of technical terms.

QWEN CALIBRATION VERSION: 20260726-v4-absolute-quality

ABSOLUTE-QUALITY CALIBRATION FOR ALL EIGHT SUBMETRICS
Use the full 1-5 scale and score the demonstrated function directly. Do not begin from a default score and then merely count defects: that approach systematically over-rates short documents because they offer fewer opportunities to observe weaknesses. Absence of a detected defect is never proof of coverage, taxonomy, synthesis, alignment, progression, or specificity.

5 = Exceptional, comprehensive, and close to submission-ready for this submetric. Strong positive evidence is distributed across the beginning, multiple substantive middle sections, and the ending; only negligible imperfections remain.
4 = Strong research-survey execution. The function is clearly and repeatedly demonstrated across the manuscript, with only localized minor weaknesses.
3 = Functional but substantially improvable. The intended function is present, but depth, breadth, comparative structure, or consistency is limited enough to require material revision.
2 = Weak. Important parts are missing, shallow, confused, or frequently ineffective.
1 = Failed or nearly absent.

CALIBRATION AGAINST THE SHORT-DOCUMENT BIAS
- Judge positive achievement, not defect density. A concise overview with fluent prose may have few visible errors yet still lack the substantive evidence required for 4 or 5.
- Do not reward or penalize page count, word count, citation count, section count, or table count by itself. Nevertheless, adequate substantive treatment is required: naming a major branch in one paragraph is not equivalent to reviewing its mechanisms, representative trajectories, trade-offs, evaluation, and limitations.
- For a broad technical topic, a document that mainly offers a short sequence of narrative topic summaries, while lacking a developed classification structure, cross-method comparison, evaluation or benchmark synthesis, and distributed gap analysis, normally cannot exceed 3 on Coverage, Taxonomy, Organization, or Insight. This is a functional-depth rule, not a length rule.
- A long reference list does not compensate for a thin analytical body. Judge how much of the literature is converted into taxonomy, comparison, evidence synthesis, and actionable conclusions.
- Conversely, do not penalize a comprehensive survey merely for being long or information-dense. Explicit scope and contribution statements, a survey pipeline, a well-explained taxonomy, comparison matrices, benchmark or metric synthesis, timelines, and technically grounded challenge analyses are positive scholarly evidence when they genuinely organize or compare the literature.

SUBMISSION-READINESS ANCHORS
- Typical 4-level TSQ evidence includes a defensible research-space map, explicit or unmistakable taxonomy, substantive treatment of major branches, comparative tables or equally effective textual comparisons, evaluation practice, limitations, and gaps derived from the reviewed evidence.
- Typical 3-level TSQ evidence includes a coherent and useful overview but incomplete research-space mapping, mostly narrative organization, uneven branch depth, limited comparison, or generic/concentrated gap analysis.
- Typical 2-level TSQ evidence includes an extended abstract, outline, annotated bibliography, or topic sequence that introduces directions without building a usable survey structure.
- For HDQ, a polished short essay is not automatically strong. Scores of 4 or 5 require representative evidence from several substantive sections that paragraphs build mechanisms, comparisons, consequences, and limitations rather than compressing each major topic into one self-contained paragraph.

Apply consistency caps only to verified manuscript defects:
- a severe systematic defect normally caps the directly affected submetric at 2;
- a moderate recurring defect normally caps it at 3;
- a mild recurring defect normally caps it at 4.
Do not mechanically deduct twice for the same defect across unrelated submetrics. Identical scores across methods are allowed when justified; never force a ranking.

NON-NEGOTIABLE NECESSARY CONDITIONS
A 5 is allowed only if all necessary conditions for that submetric are demonstrated:
- Coverage: the core technical space, evaluation practice, limitations, and relationships among major branches are treated with depth and topical discipline. Mentioning many topics is not sufficient.
- Taxonomy: the classification principle is explicit or unmistakable, categories use compatible abstraction levels, overlap is controlled or acknowledged, and no recurring duplicate structure exists.
- Organization: sections have distinct functions, form a manuscript-wide argument, and avoid systematic repetition or interchangeable compilations.
- Insight: comparison, trade-off, contradiction, limitation, and gap reasoning is technically grounded and distributed beyond the conclusion.
- Alignment: representative early, middle, and late sections fulfill their headings at section, subsection, and paragraph levels.
- Progression: paragraph order advances reasoning in multiple representative sections; topical adjacency and transition phrases alone are insufficient.
- Specificity: claims consistently state mechanisms, conditions, evaluation context, or bounded comparisons; names, citations, formulas, and numbers alone are insufficient.
- Local synthesis: explicit relationships among nearby works dominate representative passages; paper-by-paper enumeration is negligible.

DOCUMENT-FUNCTION HARD CAPS
The input may include a deterministic `markdown_functional_audit`. It measures document structure, not method identity or scientific correctness. Apply these caps when the named flag is true:
- `thin_overview_flag = true`: Research-Space Coverage, Taxonomy Boundary Control, Survey Organization, and Synthesis Insight are each at most 3. Each HDQ submetric is at most 4. This flag requires the combination of a very small analytical body, very few substantive headings, and no detected synthesis aids; it is not triggered by brevity alone.
- `reference_dominated_without_synthesis_aids = true`: the same TSQ caps of 3 apply because a long bibliography has not been converted into a developed survey body. HDQ submetrics are at most 4.
- `limited_functional_depth_flag = true` without either stronger flag: TSQ and HDQ submetrics are at most 4. A 4 still requires clear positive evidence.
- `unreproducible_corpus_claim_flag = true`: Research-Space Coverage and Taxonomy Boundary Control are each at most 4. This applies only when the manuscript explicitly claims a retrieved, curated, systematic, or statistically characterized survey corpus but lacks reproducible retrieval and screening information.
- `broad_survey_without_synthesis_aids_flag = true`: the four TSQ submetrics and Local Synthesis are each at most 4. A broad technical survey may use prose instead of visuals, but without any taxonomy figure, comparison table/matrix, benchmark table, timeline, or equivalent detected synthesis aid, there is insufficient evidence for an exceptional submission-ready score of 5.
- `formulaic_transition_pattern_flag = true`: Survey Organization and Paragraph Argument Progression are each at most 4. This flag requires repeated template transitions across a multi-section analytical body and cannot be triggered by one or two conventional phrases.
- These structural flags never justify a score below 3 by themselves. Scores of 1-2 still require verified semantic defects.
- Do not ignore a true flag because the available prose is fluent, locally coherent, or technically specific. Local polish cannot substitute for research-space mapping and manuscript-wide synthesis.

OBJECTIVE HARD CAPS
Apply every relevant cap and then use the lowest applicable maximum. The caps are safeguards, not instructions to manufacture defects.
- Systematic exact or near-duplicate headings, repeated section openings, or copy-like structural blocks across many sections: Survey Organization <= 2 and Paragraph Argument Progression <= 3. A localized repeated heading caps those dimensions at 4.
- Substantial, weakly justified scope drift into multiple adjacent domains: Research-Space Coverage <= 3, Taxonomy Boundary Control <= 2, and Multi-Level Goal Alignment <= 3. Dominant drift caps them at 2, 1, and 2 respectively.
- A table of contents or topic list without a defensible classification principle: Taxonomy Boundary Control <= 3. Recurring mixtures of incompatible axes such as methods, datasets, applications, and goals within the same claimed primary partition also cap it at 3. An isolated misplaced row or method is a localized defect and caps at 4, not 2. A survey may legitimately use several orthogonal analytical dimensions, and a method may appear in multiple dimensions when the text acknowledges its different roles. A comparison table may legitimately contain columns for different analytical attributes; this is not taxonomy-axis mixing by itself.
- Recurring paper-by-paper enumeration without explicit comparison in at least three distinct sections: Paragraph Argument Progression <= 3 and Local Synthesis <= 3. If independent paper summaries dominate the inspected prose across the manuscript, both are <= 2. Numbered lists, challenge lists, or repeated Discussion subsections are not paper-by-paper enumeration when each item synthesizes mechanisms, trade-offs, or multiple works.
- Recurring formulaic transitions, generic section templates, or interchangeable prose blocks: Survey Organization <= 3 and Paragraph Argument Progression <= 3.
- Synthesis or future-work claims that are mostly generic, aspirational, or disconnected from reviewed evidence: Synthesis Insight <= 3. If effective synthesis is nearly absent, it is <= 2.
- Repeated name-dropping, technical terms, equations, or numerical claims without mechanisms, conditions, or comparison context: Atomic Claim Specificity <= 3.
- Material internal contradictions about the manuscript's own taxonomy, scope, components, or conclusions cap the directly affected submetric at 3; repeated contradictions cap it at 2.
- Unfinished fragments, broken sentences, or obvious assembly artifacts that recur across sections cap Progression and Alignment at 3; widespread artifacts cap them at 2.

Survey-method reporting is relevant only when the manuscript presents itself as systematic, claims a retrieved corpus, reports corpus statistics, or relies on reproducible selection claims. In those cases, absent databases, search terms, cutoff dates, screening counts, or inclusion/exclusion criteria are direct negative evidence for boundary control and the credibility of coverage; they normally cap the affected score at 4, or at 3 when the unsubstantiated corpus claims are central. Do not penalize a clearly narrative survey merely for lacking a systematic-review protocol.

PREVALENCE, PARSING, AND FALSE-POSITIVE SAFEGUARDS
- "Moderate recurring" requires verified semantic evidence from at least three distinct substantive sections or a defect affecting a major manuscript component. "Severe" requires a systematic or dominant manuscript-wide pattern.
- Markdown is parsed from a PDF. Broken symbols, split words, apparent mid-sentence starts, misplaced caption text, line-order fragments, missing Markdown heading markers, and isolated orphan-like lines are presumptive extraction artifacts. They must not trigger a semantic defect, diagnostic, deduction, or hard cap unless coherent surrounding prose in multiple locations independently proves that the defect exists in the manuscript rather than the parser output.
- Do not score visual layout, typography, equation rendering, table rendering, or page assembly from Markdown. Those belong to MAR.
- Repeated generic labels such as "Discussion", "Summary", or "Challenges" under differently numbered method families are conventional parallel structure. They are not duplicate headings, template defects, or redundant sections unless the substantive content itself is duplicated or functionally interchangeable.
- A structured numbered challenge, trade-off, or failure-mode list is synthesis when each item explains a distinct mechanism, boundary, consequence, or research tension. It is not paper-by-paper enumeration.
- A comparison table or taxonomy table that places methods in rows and analytical properties in columns is a synthesis aid, not enumerative writing. Long method lists are negative only when neither surrounding prose nor table fields establish meaningful relationships, distinctions, or decision-relevant attributes.
- A multi-dimensional survey may analyze the same method under several orthogonal lenses. This is not taxonomy-axis mixing when each lens and the method's role are stated. Apply boundary deductions only when the manuscript falsely presents incompatible lenses as one mutually exclusive partition.
- Do not equate technical breadth with scope drift. Adjacent material is negative only when its relation to the stated topic is weakly justified and it occupies a material portion of the survey.
- Survey-method reporting is positive evidence when present. Its absence is relevant only when the manuscript claims a systematic/retrieved corpus or reports corpus statistics; even then, it normally limits Coverage or Boundary Control to 4. Use a cap of 3 only when unreproducible selection claims are central to the survey's coverage argument.
- A supplied pre-audit is a hypothesis, not an authority. Reject every pre-audit finding that violates these safeguards. Deterministic exact duplicate counts may be trusted only for the extracted Markdown strings; they do not prove duplicated visible headings or duplicated substantive content in the PDF.

MANDATORY AUDIT PROCEDURE
Before scoring, inspect the title/abstract/introduction, the global section structure, at least three representative substantive middle sections from different parts of the document, and the conclusion or future-work discussion. For HDQ, inspect representative paragraphs from early, middle, and late sections. For every submetric, identify both the strongest positive evidence and the strongest verified limitation, then compare them with the absolute anchors. Do not search only for defects and do not treat fewer observed defects as positive achievement. Scan the complete heading sequence for genuine semantic duplication and inspect whether the same topic reappears under incompatible branches, while applying all parsing safeguards. Also inspect whether the manuscript converts its literature into substantive taxonomy, comparison, evaluation synthesis, and grounded conclusions rather than merely mentioning each topic.

Each rationale must name concrete section titles, subsection topics, or representative passages and must state both the strongest supporting evidence and the most important limitation. It must also state the active cap when one applies. A 5 rationale must cite positive evidence from at least three distinct document regions, explicitly describe the strongest checked counterexample, and explain why no hard cap applies. Generic claims such as "consistently strong", "comprehensive", "no significant omissions", or "no material defects" are insufficient without those locations. Do not claim that every section or paragraph was checked unless the full input supports that claim. A contradiction between a rationale and its score must be resolved in favor of the lower score.

Score the following eight submetrics independently, each with an integer from 1 to 5.

TSQ — Taxonomic Synthesis Quality

1. Research-Space Coverage
Judge whether the survey covers the core problems, major technical branches, representative research trajectories, evaluation practices, and important limitations of the stated topic while maintaining topical focus.
5: The core research space and its relationships are comprehensively represented; only negligible omissions exist and adjacent topics remain clearly subordinate.
4: Coverage is strong, with a few localized omissions or mildly over-broad passages.
3: The central topic is covered, but several important directions, evaluation issues, or relationships are missing, uneven, or displaced by background.
2: Coverage is narrow, substantially off-topic, or dominated by only part of the research space.
1: No effective coverage of the stated research space is formed.
Deduct for topic drift, excessive generic background, breadth without depth, or missing core branches. A long list of topics is not comprehensive coverage by itself.

2. Taxonomy Clarity and Boundary Control
Judge whether the classification principle is explicit and stable, whether hierarchy is meaningful, and whether section/subsection boundaries are non-overlapping.
5: A coherent classification principle governs the hierarchy throughout; categories are well motivated, mutually distinguishable, and consistently used, with no material duplication.
4: The taxonomy is strong but contains minor overlap, a small number of duplicate or weakly placed topics, or a locally inconsistent boundary.
3: A usable taxonomy exists, but classification principles are partly implicit or mixed, and multiple boundaries or repeated topics are unclear.
2: Categories are frequently overlapping, enumerative, duplicated, or based on inconsistent principles.
1: No meaningful taxonomy; sections are essentially an arbitrary accumulation.
Duplicate headings, repeated coverage, parallel categories at incompatible abstraction levels, or a table of contents without an explained classification rationale are direct negative evidence.

3. Survey Organization and Functional Coherence
Judge whether sections have distinct functions and form substantive conceptual, problem-driven, methodological, historical, or evaluative progression.
5: Section order and functional division create a clear end-to-end argument, with substantive transitions and no material structural redundancy.
4: The overall progression is strong, with a few weak transitions or localized structural redundancies.
3: The document is readable and broadly ordered, but progression is partly template-driven, several sections function as stand-alone compilations, or transitions do not advance the argument.
2: Functional division and ordering are often confusing or repetitive.
1: Almost no coherent organizational thread is present.
Do not treat phrases such as 'the previous section' or 'the next section' as evidence of substantive organization by themselves.

4. Synthesis Insight and Gap Analysis
Judge whether the survey derives comparative insights, trends, trade-offs, methodological limitations, unresolved contradictions, research gaps, and actionable future directions from the reviewed literature.
5: Insight is technically specific and distributed throughout the survey; gaps and future directions follow from concrete trade-offs or limitations rather than generic claims.
4: Strong synthesis and gap analysis are present, but depth or distribution is uneven in a few places.
3: Some useful synthesis exists, yet many insights are generic, concentrated in the conclusion, or insufficiently connected to prior evidence.
2: The document mainly introduces existing work, with shallow or clichéd gap statements.
1: Effective synthesis and gap analysis are nearly absent.
Statements such as 'more research is needed', 'challenges remain', or generic calls for robustness, efficiency, or interpretability do not constitute strong insight without a specific mechanism, boundary, or trade-off.

HDQ — Hierarchical Drafting Quality

5. Multi-Level Goal Alignment
Judge whether each section serves the survey's global purpose, each subsection serves its parent section, and paragraphs remain aligned with their local heading.
5: Goals are explicit or readily inferable and consistently aligned across all levels, with no material drift in the inspected early, middle, and late sections.
4: Alignment is strong with a few localized digressions or overlong background passages.
3: Most content is relevant, but recurring generic paragraphs, weak subsection purposes, or local topic drift are present.
2: Many paragraphs or subsections have weak relationships to their parent goals.
1: Local content is substantially disconnected from the stated structure.

6. Paragraph Argument Progression
Judge whether paragraphs within subsections build comparison, contrast, causal explanation, problem progression, induction, or a staged argument rather than simply appearing next to one another.
5: Argument progression is consistently explicit and layered across representative subsections; paragraph order materially advances the reasoning.
4: Progression is strong in most inspected subsections, with a small number of loose or list-like sequences.
3: Paragraphs are topically related, but recurring passages are merely adjacent summaries and their order is often interchangeable.
2: Paragraphs frequently resemble stitched independent summaries or paper-by-paper descriptions.
1: Almost no paragraph-level progression is visible.
A sequence of coherent paper summaries is still enumerative unless relationships and consequences are explicitly synthesized.

7. Atomic Claim Specificity and Technical Concreteness
Judge whether claims are specific and potentially verifiable, with mechanisms, methodological boundaries, task conditions, evaluation context, or qualified comparisons where appropriate.
5: Concrete, bounded, technically meaningful claims dominate throughout the inspected sections, with negligible vague or unsupported evaluative language.
4: Most claims are specific, but some localized generalizations or missing conditions remain.
3: Technical content is present, yet many claims are broad, term-heavy, or lack mechanisms, conditions, and comparison context.
2: Generic evaluations and name-dropping dominate over technically bounded claims.
1: Specific and verifiable academic claims are nearly absent.
Technical vocabulary, model names, datasets, and citation density do not by themselves demonstrate concreteness.

8. Local Synthesis and Non-Enumerative Writing
Judge whether nearby works are grouped and compared through mechanisms, assumptions, use conditions, evidence, limitations, or trade-offs rather than introduced one by one.
5: Strong comparison and synthesis dominate representative local passages across the document; paper-by-paper enumeration is negligible.
4: Many passages synthesize effectively, with a few localized enumerative sequences.
3: Synthesis is present but inconsistent, and obvious paper-list writing recurs.
2: Most local text introduces works independently with weak comparison.
1: The document is almost entirely a list of papers or fragmented summaries.

OUTPUT REQUIREMENTS
Return only valid JSON, with no markdown or extra text. Write all rationales and overall_assessment in English. Scores must be integers from 1 to 5. Diagnostics are explanatory only and must use exactly one of: none, mild, moderate, severe.

Return this exact schema:
{{
  "TSQ": {{
    "Research-Space Coverage": {{"score": 0, "rationale": ""}},
    "Taxonomy Clarity and Boundary Control": {{"score": 0, "rationale": ""}},
    "Survey Organization and Functional Coherence": {{"score": 0, "rationale": ""}},
    "Synthesis Insight and Gap Analysis": {{"score": 0, "rationale": ""}}
  }},
  "HDQ": {{
    "Multi-Level Goal Alignment": {{"score": 0, "rationale": ""}},
    "Paragraph Argument Progression": {{"score": 0, "rationale": ""}},
    "Atomic Claim Specificity and Technical Concreteness": {{"score": 0, "rationale": ""}},
    "Local Synthesis and Non-Enumerative Writing": {{"score": 0, "rationale": ""}}
  }},
  "diagnostics": {{
    "topic_drift": "none | mild | moderate | severe",
    "template_style": "none | mild | moderate | severe",
    "enumerative_writing": "none | mild | moderate | severe",
    "redundancy": "none | mild | moderate | severe",
    "over_breadth": "none | mild | moderate | severe",
    "duplicate_heading_or_structure_pattern": "none | mild | moderate | severe",
    "taxonomy_axis_mixing": "none | mild | moderate | severe",
    "internal_inconsistency_or_assembly_artifact": "none | mild | moderate | severe",
    "survey_methodology_omission": "none | mild | moderate | severe"
  }},
  "applied_score_caps": ["submetric: cap and concrete trigger"],
  "overall_assessment": "State the three most consequential document-wide strengths/defects and every cap applied."
}}"""



def build_markdown_functional_audit(markdown: str) -> dict[str, Any]:
    """Deterministic readiness signals used to prevent reference-heavy short-overview bias."""
    lines = markdown.splitlines()
    heading_pattern = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
    headings: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        match = heading_pattern.match(line)
        if match:
            headings.append(
                {
                    "line": line_no,
                    "level": len(match.group(1)),
                    "text": match.group(2).strip(),
                }
            )

    reference_start = len(lines)
    for item in headings:
        if re.match(
            r"^(?:\d+(?:\.\d+)*\s+)?(?:references|bibliography)\b",
            item["text"],
            flags=re.I,
        ):
            reference_start = item["line"] - 1
            break

    body = "\n".join(lines[:reference_start])
    references = "\n".join(lines[reference_start:])
    body_words = len(re.findall(r"\b[\w'-]+\b", body))
    body_characters = len(body)
    reference_characters = len(references)

    substantive_headings = [
        item
        for item in headings
        if item["line"] - 1 < reference_start
        and not re.match(
            r"^(?:abstract|keywords?|references|bibliography)\b",
            item["text"],
            flags=re.I,
        )
    ]
    lower_body = body.lower()
    markdown_table_separators = sum(
        1
        for line in lines[:reference_start]
        if re.match(
            r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$",
            line,
        )
    )
    table_mentions = len(
        re.findall(r"(?i)\btable\s+(?:\d+|[ivxlcdm]+)\b", body)
    )
    figure_mentions = len(
        re.findall(r"(?i)\bfig(?:ure)?\.?\s*(?:\d+|[ivxlcdm]+)\b", body)
    )
    synthesis_aid_count = (
        markdown_table_separators + table_mentions + figure_mentions
    )
    transition_patterns = {
        "building_upon": r"\bbuilding upon\b",
        "preceding_section": r"\bpreceding section\b",
        "sets_the_stage": r"\bsets? the stage\b",
        "moving_forward": r"\bmoving forward\b",
    }
    template_transition_counts = {
        name: len(re.findall(pattern, lower_body))
        for name, pattern in transition_patterns.items()
    }
    template_transition_total = sum(template_transition_counts.values())
    has_explicit_taxonomy_signal = bool(
        re.search(
            r"\b(?:taxonomy|classification framework|research landscape|method comparison|comparative matrix)\b",
            lower_body,
        )
    )
    has_evaluation_signal = bool(
        re.search(
            r"\b(?:benchmark|evaluation metric|evaluation protocol|dataset comparison)\b",
            lower_body,
        )
    )
    has_gap_signal = bool(
        re.search(
            r"\b(?:research gap|open challenge|future direction|limitation|trade-off|tradeoff)\b",
            lower_body,
        )
    )
    claims_retrieved_or_systematic_corpus = bool(
        re.search(
            r"\b(?:retrieved|curated|screened|selected)\s+(?:survey\s+)?corpus\b"
            r"|\bsystematic\s+(?:literature\s+)?review\b"
            r"|\bcorpus\s+(?:contains|comprises|includes)\s+\d+"
            r"|\bannual\s+(?:paper|publication)\s+distribution\b",
            lower_body,
        )
    )
    methodology_signal_patterns = {
        "database_sources": (
            r"\b(?:scopus|web of science|semantic scholar|google scholar|"
            r"ieee xplore|acm digital library|pubmed)\b"
        ),
        "search_query_or_terms": (
            r"\b(?:search quer(?:y|ies)|search string|search terms?|"
            r"retrieval keywords?)\b"
        ),
        "screening_counts_or_flow": (
            r"\b(?:prisma|screening flow|records? screened|papers? screened|"
            r"after deduplication|duplicates? removed)\b"
        ),
        "inclusion_exclusion": (
            r"\b(?:inclusion criteria|exclusion criteria|eligibility criteria|"
            r"inclusion and exclusion)\b"
        ),
        "retrieval_cutoff": (
            r"\b(?:search cutoff|retrieval cutoff|cutoff date|search conducted|"
            r"last searched)\b"
        ),
    }
    methodology_signals = {
        name: bool(re.search(pattern, lower_body))
        for name, pattern in methodology_signal_patterns.items()
    }
    reproducible_methodology_signal_count = sum(methodology_signals.values())
    unreproducible_corpus_claim = (
        claims_retrieved_or_systematic_corpus
        and reproducible_methodology_signal_count < 3
    )

    very_small_body = body_words < 4000
    few_substantive_headings = len(substantive_headings) < 10
    no_synthesis_aids = synthesis_aid_count == 0
    reference_dominated = (
        reference_characters > body_characters and reference_characters > 0
    )
    thin_overview = (
        very_small_body and few_substantive_headings and no_synthesis_aids
    )
    reference_dominated_without_aids = (
        reference_dominated and very_small_body and no_synthesis_aids
    )
    limited_depth = (
        body_words < 7000
        and len(substantive_headings) < 16
        and synthesis_aid_count < 2
    )
    broad_survey_without_synthesis_aids = (
        body_words >= 7000
        and len(substantive_headings) >= 16
        and synthesis_aid_count == 0
    )
    formulaic_transition_pattern = (
        body_words >= 7000
        and len(substantive_headings) >= 20
        and template_transition_total >= 4
    )

    return {
        "body_word_count": body_words,
        "body_character_count": body_characters,
        "reference_character_count": reference_characters,
        "reference_to_body_character_ratio": round(
            reference_characters / body_characters, 4
        )
        if body_characters
        else None,
        "substantive_heading_count": len(substantive_headings),
        "substantive_heading_titles": [
            item["text"] for item in substantive_headings[:80]
        ],
        "markdown_table_separator_count": markdown_table_separators,
        "table_mention_count": table_mentions,
        "figure_mention_count": figure_mentions,
        "synthesis_aid_signal_count": synthesis_aid_count,
        "template_transition_counts": template_transition_counts,
        "template_transition_total": template_transition_total,
        "broad_survey_without_synthesis_aids_flag": (
            broad_survey_without_synthesis_aids
        ),
        "formulaic_transition_pattern_flag": formulaic_transition_pattern,
        "has_explicit_taxonomy_signal": has_explicit_taxonomy_signal,
        "has_evaluation_signal": has_evaluation_signal,
        "has_gap_or_tradeoff_signal": has_gap_signal,
        "claims_retrieved_or_systematic_corpus": (
            claims_retrieved_or_systematic_corpus
        ),
        "reproducible_methodology_signals": methodology_signals,
        "reproducible_methodology_signal_count": (
            reproducible_methodology_signal_count
        ),
        "unreproducible_corpus_claim_flag": unreproducible_corpus_claim,
        "thin_overview_flag": thin_overview,
        "reference_dominated_without_synthesis_aids": (
            reference_dominated_without_aids
        ),
        "limited_functional_depth_flag": limited_depth,
        "interpretation": (
            "Flags combine multiple structural conditions and are method-agnostic. "
            "They do not reward length by itself and do not evaluate visual quality."
        ),
    }


def apply_markdown_functional_caps(
    metric_result: dict[str, Any], audit: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Apply only the method-agnostic hard caps declared in the judge rubric."""
    cap_map: dict[tuple[str, str], int] = {}
    reasons: dict[tuple[str, str], list[str]] = {}

    def add_cap(group: str, dimension: str, cap: int, reason: str) -> None:
        key = (group, dimension)
        cap_map[key] = min(cap_map.get(key, 5), cap)
        reasons.setdefault(key, []).append(reason)

    thin = bool(audit.get("thin_overview_flag")) or bool(
        audit.get("reference_dominated_without_synthesis_aids")
    )
    if thin:
        for dimension in TSQ_SUBMETRICS:
            add_cap("TSQ", dimension, 3, "thin/reference-dominated overview")
        for dimension in HDQ_SUBMETRICS:
            add_cap("HDQ", dimension, 4, "thin/reference-dominated overview")
    elif audit.get("limited_functional_depth_flag"):
        for dimension in TSQ_SUBMETRICS:
            add_cap("TSQ", dimension, 4, "limited functional depth")
        for dimension in HDQ_SUBMETRICS:
            add_cap("HDQ", dimension, 4, "limited functional depth")

    if audit.get("unreproducible_corpus_claim_flag"):
        add_cap(
            "TSQ",
            "Research-Space Coverage",
            4,
            "unreproducible claimed survey corpus",
        )
        add_cap(
            "TSQ",
            "Taxonomy Clarity and Boundary Control",
            4,
            "unreproducible claimed survey corpus",
        )

    if audit.get("broad_survey_without_synthesis_aids_flag"):
        for dimension in TSQ_SUBMETRICS:
            add_cap("TSQ", dimension, 4, "broad survey without synthesis aids")
        add_cap(
            "HDQ",
            "Local Synthesis and Non-Enumerative Writing",
            4,
            "broad survey without synthesis aids",
        )

    if audit.get("formulaic_transition_pattern_flag"):
        add_cap(
            "TSQ",
            "Survey Organization and Functional Coherence",
            4,
            "recurring formulaic transitions",
        )
        add_cap(
            "HDQ",
            "Paragraph Argument Progression",
            4,
            "recurring formulaic transitions",
        )

    scores = metric_result.get("scores", {})
    applied: list[str] = []
    for (group, dimension), cap in cap_map.items():
        item = scores.get(group, {}).get(dimension)
        if not isinstance(item, dict):
            continue
        try:
            old_score = int(item.get("score"))
        except Exception:
            continue
        if old_score <= cap:
            continue
        item["score"] = cap
        reason = "; ".join(dict.fromkeys(reasons[(group, dimension)]))
        item["rationale"] = (
            str(item.get("rationale", "")).rstrip()
            + f" [Deterministic rubric cap applied: {cap} ({reason}).]"
        )
        applied.append(
            f"{group}.{dimension}: {old_score}->{cap} ({reason})"
        )

    for group in ("TSQ", "HDQ"):
        dimensions = TSQ_SUBMETRICS if group == "TSQ" else HDQ_SUBMETRICS
        subtotal = sum(
            int(scores.get(group, {}).get(dimension, {}).get("score", 3))
            for dimension in dimensions
        )
        if isinstance(scores.get(group), dict):
            scores[group]["subtotal"] = subtotal
        metric_result[f"{group.lower()}_raw_20"] = subtotal

    metric_result["deterministic_functional_caps_applied"] = applied
    return metric_result, applied

def encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _openai_client(api_key: str, base_url: str, timeout: int):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI Python SDK is required. Install with: pip install openai") from exc
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    choices = getattr(response, "choices", None)
    if choices:
        message = choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    chunks.append(str(item.get("text", "")))
                else:
                    chunks.append(str(getattr(item, "text", "")))
            return "".join(chunks)

    output = getattr(response, "output", None)
    if output:
        chunks = []
        for item in output:
            content = getattr(item, "content", None) or []
            for content_item in content:
                text = getattr(content_item, "text", None)
                if text:
                    chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)

    return str(response)


def call_responses_pdf_judge(
    pdf_path: Path, prompt: str, api_cfg: dict[str, Any]
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    api_key_env = api_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = str(api_cfg.get("api_key") or "").strip() or os.environ.get(api_key_env)
    if not api_key:
        return "", [f"api_on.api_key is empty and {api_key_env} is not set."]

    try:
        client = _openai_client(
            api_key=api_key,
            base_url=api_cfg["base_url"],
            timeout=int(api_cfg.get("timeout", 600)),
        )
    except Exception as exc:  # noqa: BLE001
        return "", [str(exc)]

    pdf_b64 = encode_file_base64(pdf_path)
    retries = int(api_cfg.get("retries", 3))
    sleep_base = float(api_cfg.get("retry_sleep_base_seconds", 5))

    for attempt in range(1, retries + 1):
        try:
            response = client.responses.create(
                model=api_cfg["model_name"],
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_file",
                                "filename": pdf_path.name,
                                "file_data": f"data:application/pdf;base64,{pdf_b64}",
                            },
                        ],
                    }
                ],
                temperature=api_cfg.get("temperature", 0.2),
                max_output_tokens=api_cfg.get("max_output_tokens", 4096),
            )
            return _extract_response_text(response), warnings
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Responses API attempt {attempt} failed: {exc}")
            if attempt < retries:
                time.sleep(sleep_base * attempt)

    return "", warnings


def get_img_dir_for_sample(row: dict[str, str], root: Path) -> Path:
    return root / "eval_cache" / row["method"] / "png" / row["topic_id"]


def display_img_dir_for_sample(root_label: str, row: dict[str, str]) -> str:
    root_label = root_label.rstrip("/\\") or "EVAL"
    return f"{root_label}/eval_cache/{row['method']}/png/{row['topic_id']}"


def list_page_images(img_dir: Path) -> list[Path]:
    if not img_dir.exists():
        return []
    def image_key(path: Path) -> tuple[int, str]:
        match = re.search(r"(\d+)", path.stem)
        return (int(match.group(1)) if match else 10**9, path.name)
    return sorted(
        [
            path
            for path in img_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=image_key,
    )

def render_pdf_to_images_if_needed(
    pdf_path: Path, img_dir: Path, api_cfg: dict[str, Any]
) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    existing_images = list_page_images(img_dir)
    if existing_images:
        return existing_images, warnings

    if not pdf_path.exists():
        return [], [f"Missing PDF and image directory: {img_dir}"]

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required to render PDF pages for --api off. "
            "Install with: pip install pymupdf"
        ) from exc

    img_dir.mkdir(parents=True, exist_ok=True)
    image_format = str(api_cfg.get("image_format", "png")).lower().lstrip(".") or "png"
    if image_format not in {"png", "jpg", "jpeg", "webp"}:
        warnings.append(f"Unsupported image_format={image_format!r}; using png.")
        image_format = "png"

    print(f"[RENDER] {pdf_path} -> {img_dir}/ page images")
    dpi = int(api_cfg.get("pdf_render_dpi", 160))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as document:
        for page_index in range(len(document)):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            output_path = img_dir / f"page_{page_index + 1:03d}.{image_format}"
            pixmap.save(output_path)

    return list_page_images(img_dir), warnings


def image_to_data_url(image_path: Path) -> str:
    mime_type = guess_type(image_path.name)[0] or "image/png"
    encoded = encode_file_base64(image_path)
    return f"data:{mime_type};base64,{encoded}"


def _limit_image_paths(
    image_paths: list[Path], api_cfg: dict[str, Any]
) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    max_image_pages = api_cfg.get("max_image_pages")
    if max_image_pages is None:
        return image_paths, warnings

    try:
        max_pages = int(max_image_pages)
    except (TypeError, ValueError):
        warnings.append(f"Invalid max_image_pages={max_image_pages!r}; all images were sent.")
        return image_paths, warnings

    if max_pages <= 0:
        warnings.append(f"Invalid max_image_pages={max_pages}; all images were sent.")
        return image_paths, warnings

    if len(image_paths) > max_pages:
        warnings.append(
            f"Image directory has {len(image_paths)} pages; only first {max_pages} pages were sent."
        )
        return image_paths[:max_pages], warnings
    return image_paths, warnings


def call_local_vllm_pdf_judge(
    image_paths: list[Path], prompt: str, api_cfg: dict[str, Any]
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if not image_paths:
        return "", warnings + ["No page images available for --api off evaluation."]

    try:
        client = _openai_client(
            api_key=api_cfg.get("api_key", "EMPTY"),
            base_url=api_cfg["base_url"],
            timeout=int(api_cfg.get("timeout", 600)),
        )
    except Exception as exc:  # noqa: BLE001
        return "", warnings + [str(exc)]

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
    messages = [{"role": "user", "content": content}]
    retries = int(api_cfg.get("retries", 3))
    sleep_base = float(api_cfg.get("retry_sleep_base_seconds", 5))

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=api_cfg["model_name"],
                messages=messages,
                max_tokens=api_cfg.get("max_tokens", 4096),
                temperature=api_cfg.get("temperature", 0.2),
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": api_cfg.get("enable_thinking", False)
                    }
                },
            )
            return _extract_response_text(response), warnings
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Local vLLM attempt {attempt} failed: {exc}")
            if attempt < retries:
                time.sleep(sleep_base * attempt)

    return "", warnings



def _json_unescape_string(value: str) -> str:
    try:
        return json.loads('"' + value + '"')
    except Exception:
        return value.replace('\\"', '"').replace('\\n', '\n')


def parse_tsq_hdq_loose_response(text: str) -> dict[str, Any]:
    """Recover fixed TSQ/HDQ scores from malformed local judge JSON.

    Local multimodal judges sometimes emit nearly valid JSON with a missing
    brace between TSQ and HDQ. Since the eight submetric names are fixed, we can
    safely recover score/rationale pairs without inventing any evaluation text.
    """
    recovered: dict[str, Any] = {"TSQ": {}, "HDQ": {}}
    for group_name, submetrics in (("TSQ", TSQ_SUBMETRICS), ("HDQ", HDQ_SUBMETRICS)):
        for submetric in submetrics:
            pattern = (
                r'"' + re.escape(submetric) + r'"\s*:\s*\{\s*'
                r'"score"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*'
                r'"rationale"\s*:\s*"((?:\\.|[^"\\])*)"'
            )
            match = re.search(pattern, text, flags=re.S)
            if not match:
                return {}
            recovered[group_name][submetric] = {
                "score": match.group(1),
                "rationale": _json_unescape_string(match.group(2)),
            }
    diagnostics_match = re.search(r'"diagnostics"\s*:\s*\{(.*?)\}', text, flags=re.S)
    if diagnostics_match:
        diagnostics: dict[str, str] = {}
        for key, value in re.findall(r'"([^"\\]+)"\s*:\s*"((?:\\.|[^"\\])*)"', diagnostics_match.group(1)):
            diagnostics[key] = _json_unescape_string(value)
        if diagnostics:
            recovered["diagnostics"] = diagnostics
    return recovered


def parse_llm_json(text: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    stripped = text.strip()
    if not stripped:
        return {}, ["LLM returned empty output."]

    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}, warnings
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.I)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1).strip())
            return parsed if isinstance(parsed, dict) else {}, warnings
        except json.JSONDecodeError as exc:
            warnings.append(f"Failed to parse fenced JSON: {exc}")
            loose = parse_tsq_hdq_loose_response(fenced_match.group(1).strip())
            if loose:
                warnings.append("Recovered malformed LLM JSON with fixed TSQ/HDQ submetric parser.")
                return loose, warnings

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}, warnings
        except json.JSONDecodeError as exc:
            warnings.append(f"Failed to parse braced JSON: {exc}")
            loose = parse_tsq_hdq_loose_response(stripped[start : end + 1])
            if loose:
                warnings.append("Recovered malformed LLM JSON with fixed TSQ/HDQ submetric parser.")
                return loose, warnings

    loose = parse_tsq_hdq_loose_response(stripped)
    if loose:
        warnings.append("Recovered malformed LLM JSON with fixed TSQ/HDQ submetric parser.")
        return loose, warnings

    warnings.append(f"Failed to parse LLM JSON. Raw output prefix: {stripped[:2000]}")
    return {}, warnings


def _coerce_score(value: Any, warnings: list[str], metric_name: str) -> int:
    try:
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if not match:
                raise ValueError("no numeric score")
            numeric = float(match.group(0))
        else:
            numeric = float(value)
    except Exception:  # noqa: BLE001
        warnings.append(f"{metric_name}: missing or invalid score; defaulted to 3.")
        return 3

    rounded = int(round(numeric))
    clamped = max(1, min(5, rounded))
    if numeric != clamped:
        warnings.append(f"{metric_name}: score {value!r} was coerced to {clamped}.")
    return clamped


def _normalize_metric_group(
    parsed_group: Any, submetrics: list[str], group_name: str, warnings: list[str]
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    source = parsed_group if isinstance(parsed_group, dict) else {}
    if not isinstance(parsed_group, dict):
        warnings.append(f"{group_name}: missing or invalid group; default scores were used.")

    subtotal = 0
    for submetric in submetrics:
        raw_item = source.get(submetric, {})
        if not isinstance(raw_item, dict):
            warnings.append(f"{group_name}.{submetric}: missing item; defaulted to 3.")
            raw_item = {}
        score = _coerce_score(raw_item.get("score"), warnings, f"{group_name}.{submetric}")
        rationale = raw_item.get("rationale", "")
        if rationale is None:
            rationale = ""
        normalized[submetric] = {"score": score, "rationale": str(rationale)}
        subtotal += score

    normalized["subtotal"] = subtotal
    return normalized


def normalize_scores(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    normalized = {
        "TSQ": _normalize_metric_group(parsed.get("TSQ"), TSQ_SUBMETRICS, "TSQ", warnings),
        "HDQ": _normalize_metric_group(parsed.get("HDQ"), HDQ_SUBMETRICS, "HDQ", warnings),
    }
    return normalized, warnings


def normalize_diagnostics(parsed: dict[str, Any]) -> dict[str, Any]:
    diagnostics = parsed.get("diagnostics", {})
    return diagnostics if isinstance(diagnostics, dict) else {}



def evaluate_tsq_hdq_sample(
    row: dict[str, str], root: Path, config: dict[str, Any], api_mode: str
) -> dict[str, Any]:
    warnings: list[str] = []
    llm_cfg = config["llm_api"][f"api_{api_mode}"]
    pdf_path = resolve_data_path(row["pdf_path"], root)
    topic = row.get("topic") or TOPIC_TITLES.get(row.get("topic_id", ""), EVALUATION_TOPIC)
    prompt = build_tsq_hdq_prompt(topic)
    input_type = "pdf" if api_mode == "on" else "images"
    img_dir: Path | None = None
    image_paths: list[Path] = []
    sent_image_count = 0

    if api_mode == "on":
        raw_text, call_warnings = call_responses_pdf_judge(pdf_path, prompt, llm_cfg)
    else:
        img_dir = get_img_dir_for_sample(row, root)
        try:
            image_paths, image_warnings = render_pdf_to_images_if_needed(pdf_path, img_dir, llm_cfg)
            warnings.extend(image_warnings)
        except Exception as exc:  # noqa: BLE001
            image_paths = []
            warnings.append(str(exc))
        limited_image_paths, limit_warnings = _limit_image_paths(image_paths, llm_cfg)
        sent_image_count = len(limited_image_paths)
        warnings.extend(limit_warnings)
        raw_text, call_warnings = call_local_vllm_pdf_judge(limited_image_paths, prompt, llm_cfg)
    warnings.extend(call_warnings)

    if not raw_text:
        return {
            "method": row["method"],
            "topic_id": row["topic_id"],
            "topic": topic,
            "pdf_path": row["pdf_path"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "api_mode": api_mode,
            "input_type": input_type,
            "model_name": llm_cfg.get("model_name", ""),
            "diagnostics": {},
            "scores": {"TSQ": {"subtotal": None}, "HDQ": {"subtotal": None}},
            "tsq_raw_20": None,
            "hdq_raw_20": None,
            "status": "error",
            "warnings": warnings + ["No LLM response text; scores were not generated."],
            **(
                {
                    "img_dir": f"EVAL/eval_cache/{row['method']}/png/{row['topic_id']}",
                    "num_images": sent_image_count,
                }
                if api_mode == "off"
                else {}
            ),
        }

    parsed, parse_warnings = parse_llm_json(raw_text)
    warnings.extend(parse_warnings)
    if not parsed:
        return {
            "method": row["method"],
            "topic_id": row["topic_id"],
            "topic": topic,
            "pdf_path": row["pdf_path"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "api_mode": api_mode,
            "input_type": input_type,
            "model_name": llm_cfg.get("model_name", ""),
            "diagnostics": {},
            "scores": {"TSQ": {"subtotal": None}, "HDQ": {"subtotal": None}},
            "tsq_raw_20": None,
            "hdq_raw_20": None,
            "status": "error",
            "warnings": warnings + ["LLM output could not be parsed as JSON; scores were not generated."],
            **(
                {
                    "img_dir": f"EVAL/eval_cache/{row['method']}/png/{row['topic_id']}",
                    "num_images": sent_image_count,
                }
                if api_mode == "off"
                else {}
            ),
        }

    scores, normalize_warnings = normalize_scores(parsed)
    warnings.extend(normalize_warnings)
    diagnostics = normalize_diagnostics(parsed)

    tsq_raw_20 = int(scores["TSQ"]["subtotal"])
    hdq_raw_20 = int(scores["HDQ"]["subtotal"])
    status = "done"

    result = {
        "method": row["method"],
        "topic_id": row["topic_id"],
        "topic": topic,
        "pdf_path": row["pdf_path"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "api_mode": api_mode,
        "input_type": input_type,
        "model_name": llm_cfg.get("model_name", ""),
        "diagnostics": diagnostics,
        "scores": scores,
        "tsq_raw_20": tsq_raw_20,
        "hdq_raw_20": hdq_raw_20,
        "status": status,
        "warnings": warnings,
    }
    if api_mode == "off":
        result["img_dir"] = f"EVAL/eval_cache/{row['method']}/png/{row['topic_id']}"
        result["num_images"] = sent_image_count
    return result


def write_json_result(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def result_to_csv_row(result: dict[str, Any], json_path: str) -> dict[str, Any]:
    row = {
        "method": result.get("method", ""),
        "topic_id": result.get("topic_id", ""),
        "topic": result.get("topic", ""),
        "api_mode": result.get("api_mode", ""),
        "input_type": result.get("input_type", ""),
        "tsq_raw_20": result.get("tsq_raw_20", ""),
        "hdq_raw_20": result.get("hdq_raw_20", ""),
        "json_path": json_path,
        "status": result.get("status", ""),
    }
    scores = result.get("scores", {})
    for group_name, submetrics in (("TSQ", TSQ_SUBMETRICS), ("HDQ", HDQ_SUBMETRICS)):
        group = scores.get(group_name, {})
        for submetric in submetrics:
            field_name = CSV_SCORE_FIELD_BY_SUBMETRIC[submetric]
            row[field_name] = group.get(submetric, {}).get("score", "")
    return row


def api_result_dir_name(api_mode: str) -> str:
    return f"api_{api_mode}"


def display_result_path(root_label: str, method: str, topic_id: str, api_mode: str) -> str:
    root_label = root_label.rstrip("/\\") or "EVAL"
    return f"{root_label}/results/{method}/tsq_hdq/{api_result_dir_name(api_mode)}/{topic_id}.json"


def update_tsq_hdq_scores_csv(root: Path, new_results: list[dict[str, Any]]) -> None:
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "tsq_hdq_scores.csv"

    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("method", ""), row.get("topic_id", ""), row.get("api_mode", ""))
                if all(key):
                    rows_by_key[key] = row

    for item in new_results:
        result = item["result"]
        key = (result.get("method", ""), result.get("topic_id", ""), result.get("api_mode", ""))
        if all(key):
            rows_by_key[key] = result_to_csv_row(result, item["json_path"])

    sorted_rows = sorted(
        rows_by_key.values(),
        key=lambda row: (row["topic_id"], row["method"], row.get("api_mode", "")),
    )
    filtered_rows = [{field: row.get(field, "") for field in CSV_FIELDS} for row in sorted_rows]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(filtered_rows)


def ensure_result_dirs(root: Path) -> None:
    for method in METHODS:
        (root / "results" / method / "tsq_hdq" / "api_on").mkdir(parents=True, exist_ok=True)
        (root / "results" / method / "tsq_hdq" / "api_off").mkdir(parents=True, exist_ok=True)
        (root / "eval_inputs" / method / "img").mkdir(parents=True, exist_ok=True)


def parse_methods(methods_arg: str | None) -> list[str]:
    if not methods_arg:
        return METHODS
    methods = [method.strip() for method in methods_arg.split(",") if method.strip()]
    return methods


def make_sample_row(method: str, topic_id: str) -> dict[str, str]:
    topic_id = str(topic_id).zfill(3)
    return {
        "method": method,
        "topic_id": topic_id,
        "topic": TOPIC_TITLES.get(topic_id, f"Topic {topic_id}"),
        "pdf_path": f"eval_inputs/{method}/{topic_id}.pdf",
    }


def select_methods(method: str | None, methods_arg: str | None) -> list[str]:
    if method:
        methods = [method]
    else:
        methods = parse_methods(methods_arg)
    return methods


def select_topic_ids(topic_id: str | None, all_topics: bool) -> list[str]:
    if all_topics or not topic_id:
        return TOPIC_IDS
    if str(topic_id).lower() == "all":
        return TOPIC_IDS
    return [str(topic_id).zfill(3)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TSQ and HDQ from fixed PDF inputs.")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Benchmark root directory.")
    parser.add_argument("--method", default=None, help="Single method name.")
    parser.add_argument("--topic-id", default=None, help="Single topic id such as 001, or all.")
    parser.add_argument("--all", action="store_true", help="Evaluate all topics for selected method(s).")
    parser.add_argument("--methods", default=None, help="Backward-compatible comma-separated method names.")
    parser.add_argument("--config", default=None, help="Path to config JSON. Default: <benchmark-root>/config.json")
    parser.add_argument("--api-mode", choices=("on", "off", "config"), default="off")
    parser.add_argument("--api", choices=("on", "off", "config"), default=None, help="Backward-compatible alias for --api-mode.")
    parser.add_argument("--model-name", default="", help="Optional model override. Empty uses the selected api config model.")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for interface consistency; TSQ/HDQ recomputes selected samples by default.")
    # Deprecated compatibility options. They no longer control skip behavior.
    parser.add_argument("--skip", choices=("on", "off"), default="off")
    parser.add_argument("--mode", choices=("on", "off"), default=None)
    args = parser.parse_args()

    root = resolve_root(args.root)
    config = load_config(root, args.config)
    api_mode = args.api if args.api is not None else args.api_mode
    if api_mode == "config":
        api_mode = str(config.get("llm_api", {}).get("api_mode", "off"))
    if args.model_name:
        config.setdefault("llm_api", {}).setdefault(f"api_{api_mode}", {})["model_name"] = args.model_name
    methods = select_methods(args.method, args.methods)
    topic_ids = select_topic_ids(args.topic_id, args.all)
    ensure_result_dirs(root)

    new_results: list[dict[str, Any]] = []
    evaluated = 0
    skipped_missing_input = 0
    failed = 0

    for method in methods:
        for topic_id in topic_ids:
            row = make_sample_row(method, topic_id)
            output_path = root / "results" / method / "tsq_hdq" / api_result_dir_name(api_mode) / f"{topic_id}.json"
            json_path_for_csv = display_result_path(str(root), method, topic_id, api_mode)
            pdf_path = resolve_data_path(row["pdf_path"], root)
            img_dir = get_img_dir_for_sample(row, root)
            if api_mode == "on":
                if not pdf_path.exists():
                    skipped_missing_input += 1
                    print(f"[SKIP] Missing PDF: {row['pdf_path']}")
                    continue
            else:
                if not list_page_images(img_dir) and not pdf_path.exists():
                    skipped_missing_input += 1
                    print(f"[SKIP] Missing PDF and image directory: {display_img_dir_for_sample(str(root), row)}")
                    continue
            try:
                result = evaluate_tsq_hdq_sample(row, root, config, api_mode)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                result = {
                    "method": method,
                    "topic_id": topic_id,
                    "topic": row.get("topic", ""),
                    "pdf_path": row["pdf_path"],
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    "api_mode": api_mode,
                    "model_name": config["llm_api"].get(f"api_{api_mode}", {}).get("model_name", ""),
                    "scores": {"TSQ": {"subtotal": None}, "HDQ": {"subtotal": None}},
                    "tsq_raw_20": None,
                    "hdq_raw_20": None,
                    "status": "error",
                    "warnings": [str(exc)],
                }
            write_json_result(result, output_path)
            new_results.append({"result": result, "json_path": json_path_for_csv})
            evaluated += 1
            if result.get("status") != "done":
                warning_preview = "; ".join(str(warning) for warning in result.get("warnings", [])[:2])
                print(f"[ERROR] {method}/{topic_id} api={api_mode} status=error warnings={warning_preview}")
            elif api_mode == "off":
                print(f"[DONE] {method}/{topic_id} api=off input=images pages={result.get('num_images', 0)} TSQ={result.get('tsq_raw_20')}/20 HDQ={result.get('hdq_raw_20')}/20")
            else:
                print(f"[DONE] {method}/{topic_id} api=on input=pdf TSQ={result.get('tsq_raw_20')}/20 HDQ={result.get('hdq_raw_20')}/20")

    update_tsq_hdq_scores_csv(root, new_results)
    print(
        "[SUMMARY] "
        f"evaluated={evaluated}, skipped_missing_input={skipped_missing_input}, failed={failed}, "
        f"csv={root / 'results' / 'tsq_hdq_scores.csv'}"
    )


if __name__ == "__main__":
    main()
