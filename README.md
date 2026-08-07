# DAS-Bench

This repository releases the data specification and evaluation protocol for DAS-Bench, a benchmark of automatically generated academic surveys.

The current release does not include survey-generation code, evaluator implementation code, submission-format details, private experiment artifacts, or model outputs.

## Benchmark task

Each benchmark instance provides a research topic and a fixed pool of 300 candidate-paper metadata records. The benchmark contains 30 topics spanning language models, computer vision, robotics, scientific discovery, security, medicine, climate, remote sensing, finance, and uncertainty estimation.

The frozen topic list is available in `benchmark/topics.json`. Detailed task boundaries are documented in `benchmark/task_specification.md`.

## Evaluation dimensions

The benchmark defines four complementary metric families:

- Balanced Scholarly Citation Quality (BSC)
- Manuscript Artifact Reliability (MAR)
- Taxonomic Synthesis Quality (TSQ)
- Hierarchical Drafting Quality (HDQ)

Each family contains four dimensions scored from 1 to 5. The benchmark reports the four subtotals independently and does not introduce an undocumented combined score. See `benchmark/evaluation_protocol.md` for the metric definitions.

## Compared systems

The following tables link to the official papers, product pages, source repositories, and datasets associated with the evaluated systems. A dash indicates that the corresponding resource is not publicly available.

### General

| Method | Reference | Code | Data or resources |
|---|---|---|---|
| Naive RAG | [Lewis et al. (2020)](https://arxiv.org/abs/2005.11401) | — | — |
| Gemini Deep Research | [Official product page](https://blog.google/products-and-platforms/products/gemini/google-gemini-deep-research/) | — | — |
| GPT Deep Research | [Official product page](https://openai.com/index/introducing-deep-research/) | — | — |
| Codex | [Official product page](https://openai.com/codex/get-started/) | [openai/codex](https://github.com/openai/codex) | — |

### Auto Survey Generation

| Method | Reference | Code | Data or resources |
|---|---|---|---|
| AutoSurvey | [Wang et al. (2024)](https://arxiv.org/abs/2406.10252) | [AutoSurveys/AutoSurvey](https://github.com/AutoSurveys/AutoSurvey) | [Paper database](https://1drv.ms/u/c/8761b6d10f143944/EaqWZ4_YMLJIjGsEB_qtoHsBoExJ8bdppyBc1uxgijfZBw?e=2EIzti) |
| SurveyForge | [Yan et al. (2025)](https://aclanthology.org/2025.acl-long.609/) | [InternScience/SurveyForge](https://github.com/InternScience/SurveyForge) | [SurveyBench](https://huggingface.co/datasets/InternScience/SurveyBench) · [Paper database](https://huggingface.co/datasets/InternScience/SurveyForge_database) |
| SurveyX | [Liang et al. (2025)](https://arxiv.org/abs/2502.14776) | [IAAR-Shanghai/SurveyX](https://github.com/IAAR-Shanghai/SurveyX) | [Generated examples](https://www.surveyx.cn/) |
| IterSurvey | [Zhang et al. (2025)](https://arxiv.org/abs/2510.21900) | [HancCui/IterSurvey_Autosurveyv2](https://github.com/HancCui/IterSurvey_Autosurveyv2) | [Evaluation resources](https://github.com/HancCui/IterSurvey_Autosurveyv2/tree/main/evaluation) |
| DAS | Forthcoming | Not released | DAS-Bench metadata on Hugging Face (forthcoming) |

## Topics and metadata

The paper metadata will be distributed separately through Hugging Face and is intentionally not duplicated in this Git repository. The public metadata schema is provided in `benchmark/metadata_schema.json`; dataset provenance and download instructions will be added to `data/README.md` when the dataset release is approved.

## Current release scope

This staged release includes:

- the frozen 30-topic list;
- the benchmark task definition;
- BSC, MAR, TSQ, and HDQ metric definitions;
- the public metadata schema; and
- metadata release notes.

Evaluator code and the formal submission interface are retained privately and will be considered for a later version.

## License and citation

The license and citation entry will be added after release approval. Until a license is added, the repository contents should not be treated as granting reuse rights.
