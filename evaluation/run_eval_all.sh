#!/usr/bin/env bash

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

METHODS=(
  "submission"
)

TOPIC_IDS=(
  "001" "002" "003" "004" "005" "006" "007" "008" "009" "010"
  "011" "012" "013" "014" "015" "016" "017" "018" "019" "020"
  "021" "022" "023" "024" "025" "026" "027" "028" "029" "030"
)

TOPICS=(
  "Tool Learning and Function Calling for LLM Agents"
  "Memory and Long-Context Mechanisms for Long-Horizon LLM Agents"
  "Retrieval-Augmented Generation for Large Language Models"
  "Planning and Self-Reflection in Large Language Model Reasoning"
  "Prompt Injection and Tool-Use Security in LLM Agents"
  "Program Repair and Automated Debugging with Code LLMs"
  "Multimodal Retrieval-Augmented Generation for Chart and Document Understanding"
  "Vision-Language Models for Embodied Reasoning"
  "Diffusion and Flow-Based Models for Controllable Image Generation"
  "Gaussian Splatting and Neural Rendering for Dynamic 3D Scene Reconstruction"
  "Multi-Sensor Fusion for Autonomous Driving Perception"
  "Continual Learning and Model Editing for Foundation Models"
  "Offline and Preference-Based Reinforcement Learning for Robotics"
  "Graph Neural Networks and Graph Foundation Models for Scientific Discovery"
  "Efficient LLM Serving with KV Cache, Speculative Decoding, and Quantization"
  "Vector Databases and Retrieval Systems for Large-Scale AI Applications"
  "Privacy-Preserving Machine Learning with Federated Learning and Differential Privacy"
  "AI Software Supply-Chain Security and Vulnerability Detection"
  "Human-AI Collaboration in Scientific Writing and Research Workflows"
  "Causal Representation Learning and Causal Discovery in Deep Learning"
  "Large Language Models for Generative Recommendation and User Behavior Modeling"
  "AI-Driven Protein Design with Diffusion and Language Models"
  "Single-Cell Foundation Models for Cell Type Annotation and Perturbation Prediction"
  "Radiomics and Deep Learning for Tumor Diagnosis and Prognosis"
  "Machine Learning for Solid-State Battery Materials Discovery"
  "Machine Learning for Electrocatalyst Discovery in Energy Conversion"
  "Deep Learning for Extreme Weather Forecasting"
  "Foundation Models for Satellite Earth Observation"
  "Deep Learning for Financial Risk Modeling under Uncertainty"
  "Bayesian Deep Learning for Uncertainty Quantification"
)

METHOD_FILTER=""
TOPIC_FILTER=""
OVERWRITE=0
BSC_API_MODE="off"
MAR_API_MODE="off"
TSQ_HDQ_API_MODE="off"

usage() {
  cat <<USAGE
Usage: bash $0 [--method METHOD] [--topic-id 001] [--overwrite] \
  [--bsc-api-mode on|off] [--mar-api-mode on|off] [--tsq-hdq-api-mode on|off]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      METHOD_FILTER="$2"; shift 2 ;;
    --topic-id)
      TOPIC_FILTER="$2"; shift 2 ;;
    --overwrite)
      OVERWRITE=1; shift ;;
    --bsc-api-mode)
      BSC_API_MODE="$2"; shift 2 ;;
    --mar-api-mode)
      MAR_API_MODE="$2"; shift 2 ;;
    --tsq-hdq-api-mode)
      TSQ_HDQ_API_MODE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ ${#TOPIC_IDS[@]} -ne ${#TOPICS[@]} ]]; then
  echo "[ERROR] TOPIC_IDS and TOPICS length mismatch" >&2
  exit 2
fi

for mode in "$BSC_API_MODE" "$MAR_API_MODE" "$TSQ_HDQ_API_MODE"; do
  if [[ "$mode" != "on" && "$mode" != "off" ]]; then
    echo "[ERROR] api mode must be on or off, got: $mode" >&2
    exit 2
  fi
done

SELECTED_METHODS=()
if [[ -n "$METHOD_FILTER" ]]; then
  SELECTED_METHODS=("$METHOD_FILTER")
else
  SELECTED_METHODS=("${METHODS[@]}")
fi

SELECTED_TOPIC_INDEXES=()
if [[ -n "$TOPIC_FILTER" ]]; then
  normalized_topic=$(printf "%03d" "$((10#$TOPIC_FILTER))")
  found=0
  for i in "${!TOPIC_IDS[@]}"; do
    if [[ "${TOPIC_IDS[$i]}" == "$normalized_topic" ]]; then
      SELECTED_TOPIC_INDEXES+=("$i")
      found=1
      break
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    echo "[ERROR] unknown topic-id: $TOPIC_FILTER" >&2
    exit 2
  fi
else
  for i in "${!TOPIC_IDS[@]}"; do
    SELECTED_TOPIC_INDEXES+=("$i")
  done
fi

OVERWRITE_FLAG=()
if [[ "$OVERWRITE" -eq 1 ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

topic_id_list=""
for idx in "${SELECTED_TOPIC_INDEXES[@]}"; do
  topic_id_list+="${TOPIC_IDS[$idx]} "
done

printf '[CONFIG] EVAL_ROOT=%s\n' "$EVAL_ROOT"
printf '[CONFIG] METHODS=%s\n' "${SELECTED_METHODS[*]}"
printf '[CONFIG] TOPIC_IDS=%s\n' "$topic_id_list"
printf '[CONFIG] BSC_API_MODE=%s\n' "$BSC_API_MODE"
printf '[CONFIG] MAR_API_MODE=%s\n' "$MAR_API_MODE"
printf '[CONFIG] TSQ_HDQ_API_MODE=%s\n' "$TSQ_HDQ_API_MODE"
printf '[CONFIG] PYTHON_BIN=%s\n' "$PYTHON_BIN"
printf '[CONFIG] OVERWRITE=%s\n' "$OVERWRITE"

run_step() {
  local label="$1"
  shift
  echo "[STEP] $label: $*"
  if "$@"; then
    echo "[OK] $label"
  else
    local code=$?
    echo "[WARN] $label failed with exit_code=$code; continuing" >&2
  fi
}

for method in "${SELECTED_METHODS[@]}"; do
  for idx in "${SELECTED_TOPIC_INDEXES[@]}"; do
    topic_id="${TOPIC_IDS[$idx]}"
    topic="${TOPICS[$idx]}"
    pdf_path="$EVAL_ROOT/eval_inputs/$method/$topic_id.pdf"
    if [[ ! -f "$pdf_path" ]]; then
      echo "[SKIP] missing PDF: method=$method topic_id=$topic_id"
      continue
    fi

    echo "[RUN] method=$method topic_id=$topic_id topic=$topic"

    run_step "prepare $method/$topic_id" \
      "$PYTHON_BIN" "$EVAL_ROOT/evaluation/eval_prepare.py" \
        --method "$method" \
        --topic-id "$topic_id"

    run_step "BSC $method/$topic_id" \
      "$PYTHON_BIN" "$EVAL_ROOT/evaluation/eval_bsc.py" \
        --method "$method" \
        --topic-id "$topic_id" \
        --api-mode "$BSC_API_MODE" \
        "${OVERWRITE_FLAG[@]}"

    run_step "MAR $method/$topic_id" \
      "$PYTHON_BIN" "$EVAL_ROOT/evaluation/eval_mar.py" \
        --method "$method" \
        --topic-id "$topic_id" \
        --api-mode "$MAR_API_MODE" \
        "${OVERWRITE_FLAG[@]}"

    run_step "TSQ/HDQ $method/$topic_id" \
      "$PYTHON_BIN" "$EVAL_ROOT/evaluation/eval_tsq_hdq.py" \
        --method "$method" \
        --topic-id "$topic_id" \
        --api-mode "$TSQ_HDQ_API_MODE" \
        "${OVERWRITE_FLAG[@]}"
  done
done
