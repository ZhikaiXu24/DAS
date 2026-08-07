#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-judge evaluation for Manuscript Artifact Reliability (MAR).

This script evaluates visible manuscript artifact quality from page-level PNGs.
It reads PNG pages from eval_cache/{method}/png/{topic_id} and writes results
to results/{method}/mar/api_on|api_off/{topic_id}.json.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DIMENSIONS = [
    "Citation and Reference Presentation Integrity",
    "Figure/Table Quality and Textual Integration",
    "Layout and Formatting Professionalism",
    "Manuscript Component Completeness",
]

DIAGNOSTIC_FIELDS = [
    "citation_reference_presentation_errors",
    "figure_table_artifact_issues",
    "layout_formatting_defects",
    "missing_manuscript_components",
    "visual_readability_problems",
]

DIAGNOSTIC_LEVELS = {"none", "mild", "moderate", "severe"}

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

MAR_JUDGE_PROMPT = """You are an expert reviewer evaluating the visible artifact quality of an anonymous academic survey manuscript.

You will be given page-level PNG images rendered from one survey PDF. Evaluate only visible manuscript artifact quality. Do not evaluate scientific correctness, citation support, taxonomy quality, writing depth, novelty, or research-space coverage. Do not infer or reward the generating method.

Use only the supplied page images. Do not use external knowledge. Do not reward document length, page count, citation count, figure count, visual ornament, or a generally polished first impression.

Topic:
{topic}

QWEN CALIBRATION VERSION: 20260726-v3

STRICT EVIDENCE-GATED CALIBRATION
Use the full 1-5 scale for each dimension. Start at 3 when the visible function is competently present.
5 = Visually excellent and close to submission-ready for this dimension. At least roughly 90% of applicable inspected pages must show stable, professional, clear quality, all necessary components for the dimension must be present, and there must be no material or recurring defect. A few attractive pages, a polished first page, document length, or the mere presence of figures is never enough.
4 = Strong and professional overall, with identifiable minor or localized defects that do not substantially harm use.
3 = Adequate and readable, but recurring or material presentation problems require revision.
2 = Weak; serious defects substantially reduce readability, traceability, integration, or completeness.
1 = Failed, absent, or unusable for the dimension.

A 5 rationale must identify representative early, middle, late, visual, and ending/reference page regions as applicable, name the strongest checked counterexample, and explain why it is negligible. If the input does not permit that audit, do not award 5. When relevant, a severe diagnostic caps the corresponding score at 2, a moderate recurring diagnostic caps it at 3, and a mild recurring diagnostic caps it at 4. Scores and diagnostics must be consistent. Do not force score differences between manuscripts and never tune scores to an expected method ranking.

MANDATORY VISUAL AUDIT
Inspect the first page/front matter, representative early, middle, and late body pages, pages containing figures or tables, and the reference/ending pages when supplied. Look actively for counterexamples: unreadable text, clipping, overflow, duplicated or broken headings, inconsistent typography, malformed citations, low-resolution figures, cramped tables, excessive blank space, orphaned captions, missing components, and unfinished fragments. Each rationale must cite representative page numbers and state both the strongest evidence and the most important limitation. If assigning 5, state which page regions were checked and why any observed imperfection is negligible. If pages are sampled, judge only visible pages and state that limitation.

OBJECTIVE HARD CAPS
Apply every relevant cap and use the lowest applicable maximum:
- Citation/reference presentation: if many visible reference entries omit essential scholarly metadata such as authors and year or venue/link, or if recurring entries are title-only, the score is at most 3; when this dominates the reference list, it is at most 2. Repeated undefined markers, broken entries, or unreadable wrapping also cap it at 3 or 2 according to severity.
- Figure/table quality: for a substantial technical survey whose architectures, taxonomies, benchmarks, or method comparisons visibly call for synthesis aids, complete absence of meaningful figures and tables caps the score at 2. Decorative visuals or raw counts without comparative value do not satisfy the dimension. Recurring unreadable, clipped, orphaned, or unreferenced visuals cap it at 3; widespread defects cap it at 2.
- Layout: systematic duplicate headings, repeated title blocks, broken hierarchy, overflow, clipping, or assembly artifacts across many pages cap the score at 2. A recurring but less severe pattern caps it at 3; a localized issue caps it at 4.
- Component completeness: absence of one essential manuscript component such as an abstract/summary, substantive conclusion, or references caps the score at 3. Absence of multiple essential components, an abrupt ending, placeholders, or clearly incomplete front/back matter caps it at 2. Missing keywords alone is minor and caps at 4 only when keywords are expected by the visible format.
- Any stated diagnostic and score must agree. A severe defect cannot coexist with 4 or 5 in the affected dimension; a moderate recurring defect cannot coexist with 4 or 5. If the rationale says one essential component is missing, Component Completeness cannot exceed 3; if it says multiple essential components are missing, it cannot exceed 2.

NECESSARY CONDITIONS FOR 5
- Citation/reference integrity: body citations and representative entries throughout the reference pages are consistently readable, traceable, and visibly complete. Before awarding 4 or 5, transcribe visible author, year, and venue/link tokens from at least three representative entries; never infer that metadata is present from spacing or citation style.
- Figure/table quality: meaningful visuals are legible, captioned, numbered, referenced in nearby prose, and provide genuine synthesis or comparison.
- Layout: at least roughly 90% of inspected pages are visually stable, professional, and clear, with consistent hierarchy and no recurring artifact. Before awarding 4 or 5, transcribe representative heading number/title pairs from at least three early/middle/late pages and explicitly check whether a full numbered heading is printed twice.
- Component completeness: title/front matter, abstract or clearly separated summary, introduction, substantive body, conclusion or outlook, and references are all visibly present and finished. A title followed directly by Introduction is not an abstract. Name the pages on which the abstract/summary and conclusion/outlook are visibly present.

Evaluate these four dimensions independently with integer scores from 1 to 5.

1. Citation and Reference Presentation Integrity
Evaluate visible in-text citation presentation and the reference list for readability, consistency, completeness of visible entries, and traceability.
5: Citations and references are consistently readable and professionally formatted throughout the inspected body and reference pages, with no undefined markers, visibly broken entries, or material inconsistency.
4: Presentation is reliable overall, with a few localized spacing, style, wrapping, or consistency issues.
3: Citations and references remain usable but show recurring inconsistency, awkward wrapping, weak traceability, or several visibly incomplete entries.
2: Serious missing, undefined, broken, or unreadable citation/reference presentation is common.
1: Citations or references are absent, largely unreadable, or unusable.
Do not judge whether a citation scientifically supports its claim.

2. Figure/Table Quality and Textual Integration
Evaluate whether meaningful figures and tables are readable, captioned, numbered, placed appropriately, and visibly linked to nearby discussion.
5: Applicable figures/tables are consistently crisp, legible, well-sized, professionally captioned and numbered, and clearly integrated with the text; no material clipping, overlap, or orphaning is visible.
4: Strong overall, with a few localized readability, sizing, placement, caption, or linkage issues.
3: Figures/tables are useful but recurring problems in legibility, density, captioning, numbering, placement, or textual integration require revision.
2: Most visual elements are hard to read, poorly formatted, weakly connected, or substantially clipped.
1: Meaningful figures/tables are absent where the manuscript visibly depends on them, or supplied visual elements are unusable.
Do not award 5 solely because many figures or tables are present.

3. Layout and Formatting Professionalism
Evaluate page geometry, headings, spacing, typography, columns, equations, tables, alignment, overflow, consistency, and visual stability.
5: The inspected pages show stable, publication-quality academic layout with consistent typography and hierarchy, strong readability, and no material visual defect.
4: Layout is professional overall, with only minor localized defects such as uneven spacing, isolated dense pages, or small alignment inconsistencies.
3: Readable, but recurring crowding, sparse pages, unstable hierarchy, awkward page breaks, inconsistent formatting, or small text requires revision.
2: Layout defects substantially affect readability or visual coherence.
1: The manuscript is visually broken or not usable as an academic paper.

4. Manuscript Component Completeness
Evaluate visible presence and completion of expected survey components: title/front matter, abstract or summary, introduction, substantive body, conclusion or outlook, and references.
5: All major components are visibly present, internally complete, and professionally terminated; no unfinished placeholder, abrupt ending, or materially missing component is evident.
4: The manuscript is essentially complete, with a minor omission or localized incomplete presentation.
3: Several major components are present, but one material component is missing, thin, abruptly terminated, or visibly unfinished.
2: Multiple essential components are missing or the output resembles a partial draft.
1: Most essential components are absent.

OUTPUT REQUIREMENTS
Return only valid JSON with no markdown or extra explanation. Each rationale should be concise but evidence-based, include representative page numbers, state the strongest positive evidence and most important limitation, and identify any active cap. Scores must be integers from 1 to 5. mar_raw_20 must equal the sum of the four scores. Diagnostic values must be exactly one of: none, mild, moderate, severe.

Return this exact schema:
{
  "MAR": {
    "Citation and Reference Presentation Integrity": {"score": 0, "rationale": ""},
    "Figure/Table Quality and Textual Integration": {"score": 0, "rationale": ""},
    "Layout and Formatting Professionalism": {"score": 0, "rationale": ""},
    "Manuscript Component Completeness": {"score": 0, "rationale": ""}
  },
  "mar_raw_20": 0,
  "diagnostics": {
    "citation_reference_presentation_errors": "none | mild | moderate | severe",
    "figure_table_artifact_issues": "none | mild | moderate | severe",
    "layout_formatting_defects": "none | mild | moderate | severe",
    "missing_manuscript_components": "none | mild | moderate | severe",
    "visual_readability_problems": "none | mild | moderate | severe",
    "duplicate_heading_or_layout_pattern": "none | mild | moderate | severe",
    "incomplete_reference_metadata": "none | mild | moderate | severe"
  },
  "applied_score_caps": ["dimension: cap and visible trigger"],
  "overall_assessment": "State the most consequential visible strengths/defects and every cap applied."
}

PNG page images:
"""


class MARError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(read_text(path))


def selected_api_config(config: Dict[str, Any], mode_override: str = "off") -> Dict[str, Any]:
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


def load_topics(manifest_path: Path) -> Dict[str, str]:
    if not manifest_path.exists():
        return {}
    topics: Dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            topic_id = str(row.get("topic_id", "")).zfill(3)
            topic = str(row.get("topic", "")).strip()
            if topic_id and topic and topic_id not in topics:
                topics[topic_id] = topic
    return topics


def png_dir_for(args: argparse.Namespace, topic_id: str) -> Path:
    return Path(args.cache_root) / args.method / "png" / topic_id


def api_result_dir_name(api_mode: str) -> str:
    mode = str(api_mode or "off")
    if mode not in {"on", "off"}:
        mode = "off"
    return f"api_{mode}"


def result_path_for(args: argparse.Namespace, topic_id: str) -> Path:
    return Path(args.results_root) / args.method / "mar" / api_result_dir_name(args.api_mode) / f"{topic_id}.json"


def discover_topic_ids(args: argparse.Namespace) -> List[str]:
    png_root = Path(args.cache_root) / args.method / "png"
    if not png_root.exists():
        raise FileNotFoundError(f"PNG root not found: {png_root}")
    topic_ids = []
    for path in sorted(png_root.iterdir()):
        if path.is_dir() and re.fullmatch(r"\d{3}", path.name):
            if any(path.glob("*.png")):
                topic_ids.append(path.name)
    return topic_ids


def list_png_pages(png_dir: Path) -> List[Path]:
    if not png_dir.exists():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")
    pages = sorted(p for p in png_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not pages:
        raise FileNotFoundError(f"No PNG pages found in: {png_dir}")
    return pages


def sample_pages_for_mar(
    pages: List[Any],
    max_pages: int = 50,
    strategy: str = "uniform",
) -> Tuple[List[Any], Dict[str, Any]]:
    if max_pages <= 0:
        raise ValueError("--max-pages must be positive")
    if strategy != "uniform":
        raise ValueError(f"Unsupported page sampling strategy: {strategy}")

    num_pages = len(pages)
    front_pages = int(max_pages * 0.20)
    back_pages = int(max_pages * 0.10)
    middle_pages = max_pages - front_pages - back_pages

    if num_pages <= max_pages:
        selected_indices = list(range(num_pages))
        return list(pages), {
            "total_pages": num_pages,
            "selected_pages": len(selected_indices),
            "max_pages": max_pages,
            "front_ratio": 0.20,
            "back_ratio": 0.10,
            "front_pages": front_pages,
            "back_pages": back_pages,
            "middle_pages": middle_pages,
            "strategy": strategy,
            "sampled": False,
            "selected_page_numbers_1based": [i + 1 for i in selected_indices],
        }

    front = list(range(min(front_pages, num_pages)))
    back_start = max(0, num_pages - back_pages)
    back = list(range(back_start, num_pages))
    fixed = set(front) | set(back)
    middle_pool = [i for i in range(num_pages) if i not in fixed]

    if middle_pages <= 0:
        middle = []
    elif len(middle_pool) <= middle_pages:
        middle = list(middle_pool)
    elif middle_pages == 1:
        middle = [middle_pool[round((len(middle_pool) - 1) / 2)]]
    else:
        positions = [round(i * (len(middle_pool) - 1) / (middle_pages - 1)) for i in range(middle_pages)]
        middle = [middle_pool[pos] for pos in positions]

    selected_indices = sorted(set(front + middle + back))
    if len(selected_indices) > max_pages:
        selected_indices = selected_indices[:max_pages]
    selected_pages = [pages[i] for i in selected_indices]
    return selected_pages, {
        "total_pages": num_pages,
        "selected_pages": len(selected_pages),
        "max_pages": max_pages,
        "front_ratio": 0.20,
        "back_ratio": 0.10,
        "front_pages": front_pages,
        "back_pages": back_pages,
        "middle_pages": middle_pages,
        "strategy": strategy,
        "sampled": True,
        "selected_page_numbers_1based": [i + 1 for i in selected_indices],
    }

def encode_png_data_url(path: Path) -> str:
    # Normalize image bytes in memory before sending them to vLLM. Some PNGs
    # produced by PDF renderers are valid for PIL but trigger backend decoder
    # failures in multimodal serving. Re-encoding preserves pixels and avoids
    # mutating the cached PNG files on disk.
    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            raw = buffer.getvalue()
    except Exception:
        raw = path.read_bytes()
    data = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_mar_prompt(topic: str) -> str:
    return MAR_JUDGE_PROMPT.replace("{topic}", topic)


def extract_response_text_from_dict(data: Dict[str, Any]) -> str:
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


def http_post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: int) -> Dict[str, Any]:
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
        raise MARError(f"HTTP {exc.code} from LLM API: {raw}") from exc
    return json.loads(raw)


def build_multimodal_content(prompt: str, pages: List[Path]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for page in pages:
        content.append({"type": "text", "text": f"Page image: {page.name}"})
        content.append({"type": "image_url", "image_url": {"url": encode_png_data_url(page)}})
    return content


def call_llm_judge(prompt: str, pages: List[Path], config: Dict[str, Any], api_mode_override: str = "off", model_name_override: str = "") -> Tuple[str, str, str]:
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
    content = build_multimodal_content(prompt, pages)
    for attempt in range(1, retries + 1):
        try:
            if provider == "openai_responses":
                responses_content = []
                for item in content:
                    if item["type"] == "text":
                        responses_content.append({"type": "input_text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        responses_content.append({"type": "input_image", "image_url": item["image_url"]["url"]})
                payload = {
                    "model": model,
                    "input": [{"role": "user", "content": responses_content}],
                    "temperature": float(api_cfg.get("temperature", 0.2)),
                    "max_output_tokens": int(api_cfg.get("max_output_tokens", api_cfg.get("max_tokens", 4096))),
                }
                data = http_post_json(f"{base_url}/responses", payload, api_key, timeout)
            else:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": float(api_cfg.get("temperature", 0.2)),
                    "max_tokens": int(api_cfg.get("max_tokens", api_cfg.get("max_output_tokens", 4096))),
                }
                if "enable_thinking" in api_cfg:
                    payload["chat_template_kwargs"] = {"enable_thinking": bool(api_cfg.get("enable_thinking"))}
                data = http_post_json(f"{base_url}/chat/completions", payload, api_key, timeout)
            return extract_response_text_from_dict(data), model, api_cfg.get("api_config_key", "")
        except Exception as exc:  # noqa: BLE001 - preserve retry details.
            last_exc = exc
            if attempt < retries:
                time.sleep(sleep_base * attempt)
    raise MARError(f"LLM judge failed after {retries} attempts: {last_exc}")


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



def _json_unescape_string(value: str) -> str:
    try:
        return json.loads('"' + value + '"')
    except Exception:
        return value.replace('\\"', '"').replace('\\n', '\n')


def parse_mar_loose_response(response: str) -> Optional[Dict[str, Any]]:
    """Recover common MAR JSON mistakes from local multimodal judges.

    Some models place mar_raw_20/diagnostics inside the MAR object or miss one
    closing brace. For MAR we can safely recover because dimension names are
    fixed and scores are scalar integers.
    """
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    mar: Dict[str, Dict[str, Any]] = {}
    for dim in DIMENSIONS:
        pattern = (
            r'"' + re.escape(dim) + r'"\s*:\s*\{\s*'
            r'"score"\s*:\s*(\d+)\s*,\s*'
            r'"rationale"\s*:\s*"((?:\\.|[^"\\])*)"'
        )
        match = re.search(pattern, text, flags=re.S)
        if not match:
            return None
        mar[dim] = {
            "score": int(match.group(1)),
            "rationale": _json_unescape_string(match.group(2)),
        }
    diagnostics: Dict[str, str] = {}
    for field in DIAGNOSTIC_FIELDS:
        match = re.search(r'"' + re.escape(field) + r'"\s*:\s*"([^"]+)"', text)
        if match:
            diagnostics[field] = match.group(1)
    raw_match = re.search(r'"mar_raw_20"\s*:\s*(\d+)', text)
    overall_match = re.search(r'"overall_assessment"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.S)
    return {
        "MAR": mar,
        "mar_raw_20": int(raw_match.group(1)) if raw_match else sum(item["score"] for item in mar.values()),
        "diagnostics": diagnostics,
        "overall_assessment": _json_unescape_string(overall_match.group(1)) if overall_match else "",
    }

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
                loose = parse_mar_loose_response(response)
                if loose is not None:
                    return loose, None
                return None, f"JSON parse failed: {exc1}; extracted object parse failed: {exc2}"
        loose = parse_mar_loose_response(response)
        if loose is not None:
            return loose, None
        return None, f"JSON parse failed: {exc1}"


def validate_scores(parsed: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    mar = parsed.get("MAR")
    if not isinstance(mar, dict):
        raise MARError("LLM output missing MAR object")
    total = 0
    normalized_mar: Dict[str, Dict[str, Any]] = {}
    for dim in DIMENSIONS:
        item = mar.get(dim, {})
        if not isinstance(item, dict):
            item = {}
        score = item.get("score", 1)
        try:
            score_int = int(score)
        except Exception:
            score_int = 1
            warnings.append(f"Invalid score for {dim}; coerced to 1")
        if score_int < 1 or score_int > 5:
            warnings.append(f"Score for {dim} out of range; clipped to 1-5")
            score_int = max(1, min(5, score_int))
        normalized_mar[dim] = {"score": score_int, "rationale": str(item.get("rationale", ""))}
        total += score_int
    parsed["MAR"] = normalized_mar
    if parsed.get("mar_raw_20") != total:
        warnings.append("mar_raw_20 did not equal sum of dimension scores; recomputed locally")
    parsed["mar_raw_20"] = total
    diagnostics = parsed.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    normalized_diag = {}
    for field in DIAGNOSTIC_FIELDS:
        value = str(diagnostics.get(field, "none")).strip().lower()
        if value not in DIAGNOSTIC_LEVELS:
            warnings.append(f"Invalid diagnostic level for {field}; set to none")
            value = "none"
        normalized_diag[field] = value
    parsed["diagnostics"] = normalized_diag
    parsed["overall_assessment"] = str(parsed.get("overall_assessment", ""))
    return parsed, warnings


def make_result_base(args: argparse.Namespace, topic_id: str, topic: str, png_dir: Path, pages: List[Path], page_sampling: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
    return {
        "method": args.method,
        "topic_id": topic_id,
        "topic": topic,
        "metric": "MAR",
        "evaluated_at": now_iso(),
        "input_png_dir": str(png_dir),
        "num_pages": page_sampling.get("total_pages", len(pages)),
        "num_pages_sent_to_llm": page_sampling.get("selected_pages", len(pages)),
        "page_sampling": page_sampling,
        "model_name": model_name,
        "api_mode": args.api_mode,
        "scores": {},
        "mar_raw_20": None,
        "diagnostics": {},
        "overall_assessment": "",
        "warnings": [],
        "status": "unknown",
    }


def evaluate_topic(args: argparse.Namespace, topic_id: str, topics: Dict[str, str]) -> Dict[str, Any]:
    topic_id = str(topic_id).zfill(3)
    topic = topics.get(topic_id, f"Topic {topic_id}")
    png_dir = png_dir_for(args, topic_id)
    result_path = result_path_for(args, topic_id)
    all_pages = list_png_pages(png_dir)
    pages, page_sampling_info = sample_pages_for_mar(
        all_pages,
        max_pages=args.max_pages,
        strategy="uniform",
    )
    config = load_config(Path(args.config))
    result = make_result_base(args, topic_id, topic, png_dir, pages, page_sampling_info)
    try:
        raw_response, model_name, api_config_key = call_llm_judge(build_mar_prompt(topic), pages, config, args.api_mode, args.model_name)
        parsed, parse_error = parse_llm_json_response(raw_response)
        result["model_name"] = model_name
        result["api_config_key"] = api_config_key
        if parsed is None:
            result.update({"status": "parse_error", "parse_error": parse_error, "raw_response": raw_response})
            write_json(result_path, result)
            return result
        parsed, warnings = validate_scores(parsed)
        result.update(
            {
                "scores": parsed["MAR"],
                "mar_raw_20": parsed["mar_raw_20"],
                "diagnostics": parsed["diagnostics"],
                "overall_assessment": parsed.get("overall_assessment", ""),
                "warnings": warnings,
                "status": "success",
            }
        )
        write_json(result_path, result)
        return result
    except Exception as exc:  # noqa: BLE001 - write traceable failure record.
        result.update({"status": "failed", "error": str(exc)})
        write_json(result_path, result)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAR from page-level PNG images with an LLM judge.")
    parser.add_argument("--method", default="submission")
    parser.add_argument("--topic-id", default="001", help="Topic id such as 001, or 'all'.")
    parser.add_argument("--all", action="store_true", help="Evaluate all PNG topic directories for the method.")
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "eval_cache"))
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--manifest", default=None, help="Deprecated and ignored; topic titles are built in.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.json"))
    parser.add_argument("--model-name", default="", help="Optional model override. Empty uses the selected api config model.")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum PNG pages sent to the judge. Sampling keeps 20%% front pages, 10%% back pages, and uniformly samples the rest from the middle.")
    parser.add_argument(
        "--api",
        choices=["on", "off", "config"],
        default=None,
        help="LLM API selector. 'on' uses config llm_api.api_on (gpt); 'off' uses api_off (Qwen).",
    )
    parser.add_argument(
        "--api-mode",
        choices=["on", "off", "config"],
        default="off",
        help="Backward-compatible alias for --api. Default 'off' uses config llm_api.api_off, currently Qwen3.6-35B-A3B-FP8.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.api is not None:
        args.api_mode = args.api
    if args.max_pages <= 0:
        raise ValueError("--max-pages must be positive")
    return args

def main() -> None:
    args = parse_args()
    if args.api_mode == "config":
        config_mode = str(load_config(Path(args.config)).get("llm_api", {}).get("api_mode", "off"))
        args.api_mode = config_mode if config_mode in {"on", "off"} else "off"
    topics = dict(TOPIC_TITLES)
    if args.all or str(args.topic_id).lower() == "all":
        topic_ids = discover_topic_ids(args)
    else:
        topic_ids = [str(args.topic_id).zfill(3)]
    outputs = []
    for topic_id in topic_ids:
        result = evaluate_topic(args, topic_id, topics)
        outputs.append(
            {
                "topic_id": topic_id,
                "status": result.get("status"),
                "result": str(result_path_for(args, topic_id)),
                "model_name": result.get("model_name"),
                "mar_raw_20": result.get("mar_raw_20"),
            }
        )
    print(json.dumps(outputs[0] if len(outputs) == 1 else outputs, ensure_ascii=False))


if __name__ == "__main__":
    main()
