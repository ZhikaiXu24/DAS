# Metadata

DAS-2M is distributed separately through Hugging Face and is not bundled in this Git repository.

After downloading it, place JSON records in either of these supported layouts:

```text
data/metadata/<arxiv_id>.json
```

or:

```text
data/metadata/<topic_id>/<rank>_<arxiv_id>.json
```

Each record should follow `benchmark/metadata_schema.json`. Do not include private filesystem paths in `basic_info.file_path`. Parsed paper Markdown is optional and may be placed under `data/mineru/` using the layout distributed with DAS-2M.

Hugging Face datasets:

- [DAS-Bench](https://huggingface.co/datasets/ZhikaiXu24/DAS-Bench) provides directly loadable topics and public benchmark results.
- [DAS-2M](https://huggingface.co/datasets/ZhikaiXu24/DAS-2M) provides the literature metadata lake used with the benchmark schema.
