# Evaluation protocol

The benchmark evaluates each system-topic pair with four metric families. Every dimension receives an integer score from 1 to 5. Each family therefore has a raw subtotal from 4 to 20.

## Balanced Scholarly Citation Quality (BSC)

BSC evaluates the manuscript Markdown, its reference map, and evidence cards derived from the released paper metadata.

1. Claim-Level Citation Support
2. Reference Faithfulness and Attribution Accuracy
3. Multi-Reference Synthesis Coverage and Quality
4. Citation Distribution Balance and Non-Redundancy

The evaluator separates missing evaluator-side evidence from demonstrated manuscript defects. Citation support is sampled with fixed limits defined by the evaluator command-line arguments. The released protocol filters citation-map arXiv identifiers to publication years 2020 through 2026.

## Manuscript Artifact Reliability (MAR)

MAR evaluates rendered page images and only considers visible artifact quality.

1. Citation and Reference Presentation Integrity
2. Figure/Table Quality and Textual Integration
3. Layout and Formatting Professionalism
4. Manuscript Component Completeness

## Taxonomic Synthesis Quality (TSQ)

TSQ evaluates research-space organization and synthesis at manuscript scale.

1. Research-Space Coverage
2. Taxonomy Clarity and Boundary Control
3. Survey Organization and Functional Coherence
4. Synthesis Insight and Gap Analysis

## Hierarchical Drafting Quality (HDQ)

HDQ evaluates alignment and argument quality across document levels.

1. Multi-Level Goal Alignment
2. Paragraph Argument Progression
3. Atomic Claim Specificity and Technical Concreteness
4. Local Synthesis and Non-Enumerative Writing

## Judge configuration

The evaluator supports a hosted Responses-style API and a local OpenAI-compatible chat endpoint. API credentials must be supplied through environment variables. Exact model identifiers, endpoint type, temperature, token limits, page-rendering settings, retry policy, and any repeated-run aggregation must accompany reported results.

The evaluator prompts are frozen for the reported experiments but are not part of the current public release.

## Aggregation and reporting

For each system, scores are first averaged across the evaluated topics at the criterion level. Each metric-family average is the arithmetic mean of its four criterion scores:

- BSC Avg., MAR Avg., TSQ Avg., and HDQ Avg. each range from 1 to 5.
- Total Avg. is the unweighted arithmetic mean of all 16 criteria and also ranges from 1 to 5. It is equivalent to the arithmetic mean of the four family averages when computed before display rounding.

Reports must include all four family averages whenever Total Avg. is reported. The evaluated topic set and number of topics must be stated explicitly. Human results, when provided, serve only as a reference and are excluded from system ranking.
