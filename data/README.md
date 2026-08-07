# Metadata

The metadata release will be hosted on Hugging Face and is not bundled in this Git repository.

After downloading it, place JSON records in either of these supported layouts:

```text
data/metadata/<arxiv_id>.json
```

or:

```text
data/metadata/<topic_id>/<rank>_<arxiv_id>.json
```

Each record should follow `benchmark/metadata_schema.json`. Do not include private filesystem paths in `basic_info.file_path`. Parsed paper Markdown is optional and may be placed under `data/mineru/` using the layout documented with the future dataset release.

Hugging Face dataset: `TO_BE_ADDED_AFTER_RELEASE_APPROVAL`

