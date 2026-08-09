<div align="center">

<img src="assets/das_title.svg" width="100%" alt="Deep Academic Survey: Stateful Agentic Closed-Loop Paradigm for Academic Survey Automation" />

[Zhikai Xu](https://github.com/ZhikaiXu24)<sup>1,&#42;</sup> · [Zhucun Xue](https://scholar.google.com/citations?hl=zh-CN&user=m3KDreEAAAAJ)<sup>1,&#42;</sup> · [Teng Hu](https://sjtuplayer.github.io/)<sup>2</sup> · [Yabiao Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=xiK4nFUAAAAJ)<sup>1</sup> · [Yong Liu](https://scholar.google.com/citations?user=qYcgBbEAAAAJ)<sup>1</sup> · [Jiangning Zhang](https://zhangzjn.github.io/)<sup>1,†</sup>

<sup>1</sup>Zhejiang University &nbsp;&nbsp; <sup>2</sup>Shanghai Jiao Tong University<br />
<sup>&#42;</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding author

<img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv&logoColor=white" alt="arXiv Paper" />
<img src="https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-DAS--2M-FF950D" alt="Hugging Face DAS-2M" />
<img src="https://img.shields.io/badge/%F0%9F%8C%90_DAS-Website-B750F0" alt="DAS Website" />

[Overview](#-overview) · [Release](#-release) · [Method](#-method) · [DAS-2M](#-das-2m) · [DAS-Bench](#-das-bench) · [Installation](#-installation) · [Citation](#-citation)

</div>

## 🔥 Continuous Updates

- ✅ **[2026-08-08]** DAS-2M was released.
- ✅ **[2026-08-08]** DAS-Bench and its evaluation toolkit were released.
- ⏳ **DAS method code:** To be released.

The resource badges above are placeholders and will be activated as the corresponding public pages become available.

## 🔭 Overview

Academic survey generation requires more than retrieving papers and drafting long-form text. **Deep Academic Survey (DAS)** is the first framework for **publication-oriented academic survey generation**, designed to construct **complete, evidence-grounded survey manuscripts**. A publication-oriented survey must ensure **coherent literature organization**, **traceable claim-level support**, **integrated figures and tables**, and **manuscript-level structural completeness**. To achieve this, DAS formulates survey generation as **stateful manuscript construction**, separating reusable paper understanding from topic-specific **organization, drafting, review, and finalization**.

<p align="center">
  <img src="assets/das_teaser.png" width="700" alt="Comparison between existing research systems and DAS" />
</p>

## 💡 Highlights

- **DAS-2M:** a dynamically updated literature metadata lake with survey-oriented representations of approximately two million papers.
- **Stateful agentic construction:** explicit literature, organization, writing, and finalization states connect candidate discovery to the final manuscript.
- **Scoped closed-loop review:** semantic feedback reactivates the affected writing stage, while deterministic checks protect structural integrity and compilability.
- **DAS-Bench and DAS-Eval:** a 30-topic benchmark and a 16-criterion evaluation suite for scholarly citation quality, taxonomic synthesis, hierarchical discourse, and manuscript reliability.

## 📦 Release

| Component | Status | Contents |
|---|---|---|
| [`DAS/`](DAS/) | ⏳ To be released | Core DAS method implementation and generation pipeline. |
| [`DAS-Bench/`](DAS-Bench/) | ✅ Released | Benchmark specification, topics, evaluation code, results, and qualitative materials. |
| DAS-2M | ✅ Released | Survey-oriented scholarly metadata distributed separately through Hugging Face. |

Large metadata records are hosted separately and are not duplicated in this Git repository.

## 🧭 Method

DAS maintains a shared manuscript state across four coordinated stages: literature discovery, candidate-grounded taxonomy and paper routing, hierarchical claim-and-citation planning with drafting, and scoped review followed by artifact finalization.

<p align="center">
  <img src="assets/das_overview.png" width="100%" alt="Overview of the DAS framework" />
</p>

1. **Evolving literature metadata lake:** full papers are converted into reusable survey-oriented representations and indexed for lexical and semantic discovery.
2. **Candidate-grounded taxonomy and routing:** candidate papers jointly inform the taxonomy, and papers are routed to sections according to substantive support.
3. **Hierarchical planning and drafting:** section objectives are refined into paragraph responsibilities, technical claims, and citation groups before prose generation.
4. **Review and finalization:** semantic critiques trigger scoped rollback to the affected section or paragraph, where planning or drafting is revised and then reviewed again. This closed loop continues until the content is accepted; deterministic checks then validate the artifacts before the manuscript is compiled with its references and visual elements.

## 🗃️ DAS-2M

DAS-2M contains survey-oriented representations of approximately two million arXiv papers submitted between January 2020 and June 2026. Each representation organizes bibliographic metadata, topical categorization, technical configuration, resources, methods, datasets, findings, empirical results, and limitations. The same processing and indexing pipeline supports continued updates as new papers become available.

The Hugging Face dataset page and download command will be linked from the badge at the top of this page.

## 📊 DAS-Bench

DAS-Bench evaluates publication-oriented academic surveys on 30 topics: 21 core computer science topics and nine non-CS topics. DAS-Eval contains four metric families, each with four criteria scored from 1 to 5:

- **BSC:** Balanced Scholarly Citation Quality;
- **TSQ:** Taxonomic Synthesis Quality;
- **HDQ:** Hierarchical Discourse Quality; and
- **MAR:** Manuscript Assembly Reliability.

### Main comparison

| Method | BSC Avg. ↑ | TSQ Avg. ↑ | HDQ Avg. ↑ | MAR Avg. ↑ | Total Avg. ↑ |
|---|---:|---:|---:|---:|---:|
| Human | 3.84 | 4.29 | 4.24 | 5.00 | 4.34 |
| Codex | 2.80 | 2.99 | 2.75 | 4.19 | 3.18 |
| GPT Deep Research | 3.32 | 3.48 | 3.76 | 4.14 | 3.68 |
| Gemini Deep Research | 2.95 | 3.82 | 4.07 | 4.84 | 3.92 |
| Naive RAG | 3.73 | 4.06 | 4.22 | 4.09 | 4.03 |
| AutoSurvey | 3.81 | 3.74 | 3.69 | 3.67 | 3.73 |
| SurveyForge | 3.73 | 3.81 | 3.74 | 3.83 | 3.78 |
| LiRA | 3.63 | 3.72 | 4.06 | 3.07 | 3.62 |
| InteractiveSurvey | 2.75 | 3.88 | 3.93 | 4.68 | 3.81 |
| **DAS** | **3.85** | **4.22** | **4.28** | **5.00** | **4.34** |

The main comparison follows each system's supported topic coverage. See the [DAS-Bench documentation](DAS-Bench/) for the matched 21-topic CS comparison, criterion-level CSV files, complete metric definitions, and evaluation protocol.

### Capability comparison

| System | Corpus | PRep | GTax | LRoute | DPlan | RLoop | DCheck | VInt | CAVis | PDF |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AutoSurvey | 530K | × | × | × | × | × | × | × | × | × |
| SurveyForge | 600K + 20K | × | × | × | × | × | × | × | × | × |
| SurveyX | 2.63M* + online | × | ✓ | × | × | × | × | ✓ | × | ✓ |
| InteractiveSurvey | online + uploads | × | ✓ | × | × | × | × | ✓ | × | ✓ |
| LiRA | provided references | × | × | × | × | ✓ | × | × | × | × |
| DeepSurvey | online | × | ✓ | ✓ | × | ✓ | × | × | × | × |
| **DAS** | **2M** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

A check mark denotes an explicitly documented capability; a cross denotes an absent or undocumented capability. *SurveyX reports an unreleased 2.63M-paper corpus. Definitions and sources are provided in the [benchmark documentation](DAS-Bench/README.md#system-capability-comparison).

### Qualitative manuscript comparison

Representative first and interior pages from Codex, GPT Deep Research, AutoSurvey, InteractiveSurvey, and DAS complement the aggregate scores with manuscript-level evidence.

<p align="center">
  <a href="DAS-Bench/assets/qualitative_case_study.pdf">
    <img src="DAS-Bench/assets/qualitative_case_study.png" width="100%" alt="Qualitative comparison of generated survey manuscripts" />
  </a>
</p>

## 🔨 Installation

The current public release provides DAS-Bench and its evaluation toolkit. The DAS generation method remains marked **To be released**.

```bash
git clone https://github.com/ZhikaiXu24/DAS.git
cd DAS/DAS-Bench
python -m pip install -r requirements.txt
```

Place generated survey PDFs under `eval_inputs/<method>/<topic_id>.pdf`, configure the OpenAI-compatible judges in `config.json`, and export credentials through environment variables. Never write API keys into committed files.

For a one-topic evaluation:

```bash
export OPENAI_API_KEY="your_api_key"
export PDF_EXTRACT_KIT_ROOT="/path/to/PDF-Extract-Kit"

bash evaluation/run_eval_all.sh \
  --method submission \
  --topic-id 001 \
  --bsc-api-mode on \
  --mar-api-mode on \
  --tsq-hdq-api-mode on
```

See [`DAS-Bench/README.md`](DAS-Bench/README.md) for input conventions, judge configuration, metric definitions, and complete evaluation instructions.

## ✒️ Citation

If DAS, DAS-2M, or DAS-Bench is useful for your research, please consider citing:

```bibtex
@article{xu2026deepacademicsurvey,
  title   = {Deep Academic Survey: Stateful Agentic Closed-Loop Paradigm for Academic Survey Automation},
  author  = {Xu, Zhikai and Xue, Zhucun and Hu, Teng and Wang, Yabiao and Liu, Yong and Zhang, Jiangning},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## 📄 License

The code and documentation in this repository are released under the [Apache License 2.0](LICENSE). DAS-2M is distributed separately and is subject to the terms published on its Hugging Face dataset page.

## ✉️ Contact

For questions about DAS, please contact `186368@zju.edu.cn`.
