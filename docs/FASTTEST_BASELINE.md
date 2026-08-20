# Five-keyframe regression baseline

Task 03 freezes a structural regression gate around the existing five-keyframe
fasttest. The gate reads generated artifacts but never commits or rewrites
them. All configured paths are repository-relative.

Run the structural Task05A gate without an external backend:

```powershell
python scripts/indexing/verify_fasttest_baseline.py
```

When the Task05B Elasticsearch smoke manifest has been generated, require it:

```powershell
python scripts/indexing/verify_fasttest_baseline.py --require-lexical
```

The frozen acceptance counts are:

| Check | Expected |
|---|---:|
| Canonical mapping rows | 5 |
| Unique canonical `keyframe_id` values | 5 |
| Task05A documents | 5 |
| Task05A `invalid_schema` | 0 |
| Missing/extra/duplicate identities | 0 |
| Task05B indexed documents, when required | 5 |
| Task05B failed/missing/extra/duplicate documents | 0 |

The validator checks every processed document with the existing Task05A
schema validator and compares identities as sets keyed by `keyframe_id`. It
does not infer identity from filenames or row position. A missing Task05B
manifest is a documented skip for the structural command and a failure under
`--require-lexical`.

The local acceptance run used the ignored `artifacts/fasttest/` tree and
returned all counts above. This is a reproducibility/structure baseline over
five keyframes, not evidence of retrieval quality. Model weights, embeddings,
indexes, raw OCR output, and generated documents remain outside Git.
