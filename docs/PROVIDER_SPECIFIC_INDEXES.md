# Provider-specific semantic indexes

Task05C builds one exact FAISS index per embedding provider. A collection name
is always `<environment>__<dataset>__<model_id>`; vectors cannot be routed to a
collection whose provider suffix differs from their manifest `model_id`.

Example:

```powershell
python scripts/indexing/build_provider_indexes.py `
  --embeddings-root artifacts/fasttest/features/embeddings `
  --processed-root artifacts/fasttest/task05a `
  --output-root artifacts/fasttest/provider_indexes `
  --providers beit3,metaclip2,perception_encoder `
  --environment local `
  --dataset fasttest `
  --policy create
```

`create` is idempotent only when the existing index manifest has the same
provider, collection, and source embedding-manifest hash. `rebuild` is the only
policy that explicitly replaces an existing collection. `append` retains the
existing collection and reports duplicate IDs as failed inserts. Never use
`rebuild` against a production output root without reviewing the exact target.

Each provider directory records `index_manifest.json` with source provenance,
dimension, metric, normalized state, backend, collection, indexed/failed
counts, and failed IDs. Generated FAISS files, vectors, and metadata remain
under ignored `artifacts/` paths and must not be committed.
