# Provider query encoders

`QueryEncoderRegistry.encode_query(model_id, text, index_contract=...)` encodes
the same text independently in each provider's matching text space. The
registry lazily creates and caches one adapter per provider; raw query text is
not logged or included in errors.

The adapters follow the locked provider contracts:

- BEiT-3 uses the paired retrieval model, local `beit3.spm`, BOS/EOS, a
  64-token padded sequence, and the official padding mask convention.
- MetaCLIP-2 uses the processor/model from the same pinned Hugging Face
  revision and `get_text_features` with maximum length 77.
- PE Core uses `get_text_tokenizer(model.context_length)` and
  `encode_text(..., normalize=True)` from the paired PE Core checkpoint.

BEiT-3 and PE require explicit local checkpoint paths. MetaCLIP-2 remains
local/cache-only unless model download is explicitly enabled. A missing or
failed provider raises a typed error and never invokes another adapter.

When an index contract is supplied, model ID and dimension are checked before
model initialization or encoding. Therefore a BEiT-3 query vector cannot be
routed to MetaCLIP-2 or PE indexes.
