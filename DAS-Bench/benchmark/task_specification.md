# Task specification

## Input

Each instance consists of:

- a topic identifier from `001` to `030`;
- a natural-language research topic; and
- a fixed pool of 300 candidate-paper metadata records.

## System output

A participating system generates an academic survey for each selected topic. A system may use any internal generation process, but the evaluated artifact must remain anonymous and self-contained. The formal submission interface and auxiliary-file format are not part of the current public release.

## Evaluation unit

One system-topic pair is one evaluation sample. Metric availability depends on whether the submitted artifact contains the evidence required by the corresponding evaluation family. Exact validation and failure-handling rules will be released with the evaluator implementation.

## Candidate pool

The released metadata pool is the benchmark input. Whether external papers are permitted must be fixed before public release and reported with every result. The initial paper results use the frozen pools associated with the 30 topic identifiers.
