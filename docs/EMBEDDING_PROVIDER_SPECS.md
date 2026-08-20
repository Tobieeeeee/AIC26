# Locked embedding provider specifications

Versioned runtime data lives in `configs/embedding_providers.json`. That file,
not a model-family name, is the source of truth for model construction and
artifact manifests. Runtime code may override device and batch size, but not a
checkpoint, revision, preprocessing rule, tokenizer, dimension, normalization,
or metric without updating this contract and its evidence.

## Locked matrix

| Provider | Exact checkpoint | Revision | Dim | Image/text pair | Metric | License/access |
|---|---|---:|---:|---|---|---|
| `beit3` | `beit3_base_itc_patch16_224.pth` | No upstream checkpoint revision; source `69961945…` | 768 | Official `BEiT3ForRetrieval` vision/language heads | L2 + inner product | MIT code; public Microsoft files |
| `metaclip2` | `facebook/metaclip-2-worldwide-b16` | `8cc8591f…` | 512 | One pinned `MetaClip2Model` image/text checkpoint | L2 + inner product | CC-BY-NC-4.0; public/ungated |
| `perception_encoder` | `facebook/PE-Core-L14-336` | `ee1fdb46…`, SHA256 `0cdab5b3…` | 1024 | Official PE Core `CLIP` image/text towers | L2 + inner product | Apache-2.0; public/ungated |

The dimensions are provider-specific. A matching text encoder may query only
the image index created from the same row and revision.

## BEiT-3

Microsoft explicitly recommends the ITC checkpoints for image-text retrieval,
publishes the base architecture as hidden size 768, and publishes the
SentencePiece tokenizer. The locked implementation is the official retrieval
class, whose vision and language heads both L2-normalize their outputs. The
official eval transform resizes to 224 × 224 with bicubic interpolation and
normalizes by mean/std 0.5. Evidence: [official model table and tokenizer](https://github.com/microsoft/unilm/blob/69961945a2fd283f2beed74d1f3c543265c9930c/beit3/README.md),
[paired retrieval implementation](https://github.com/microsoft/unilm/blob/69961945a2fd283f2beed74d1f3c543265c9930c/beit3/modeling_finetune.py),
and [official preprocessing](https://github.com/microsoft/unilm/blob/69961945a2fd283f2beed74d1f3c543265c9930c/beit3/datasets.py).

Microsoft's blob distribution does not expose an immutable checkpoint revision
or published SHA256, and its requirements leave several packages unpinned.
Therefore Task 06 must use an isolated compatibility environment, hash the
downloaded checkpoint, and record the resolved environment before accepting a
real artifact. This is an explicit reproducibility risk, not permission to
replace BEiT-3 with another checkpoint.

## MetaCLIP-2

The smaller distilled worldwide B/16 checkpoint is selected to keep inference
practical while remaining an official MetaCLIP-2 release. The pinned repository
contains image and text towers with projection dimension 512, the XLM-R
tokenizer assets, and the official 224-pixel CLIP processor. Evidence:
[official MetaCLIP-2 table](https://github.com/facebookresearch/MetaCLIP/blob/f47f7841f6a91cc5676729a3d125519393d87d1e/README.md),
[pinned model repository](https://huggingface.co/facebook/metaclip-2-worldwide-b16/tree/8cc8591f32b43a809f3325192d3b77aa56968ddb),
[model config](https://huggingface.co/facebook/metaclip-2-worldwide-b16/blob/8cc8591f32b43a809f3325192d3b77aa56968ddb/config.json),
and [processor config](https://huggingface.co/facebook/metaclip-2-worldwide-b16/blob/8cc8591f32b43a809f3325192d3b77aa56968ddb/preprocessor_config.json).

The model is CC-BY-NC-4.0. The team must preserve attribution and confirm that
its competition/deployment use remains within the non-commercial terms.

## Perception Encoder

PE Core L/14 336 is the official paired CLIP checkpoint used in Meta's example.
The official config gives both towers output dimension 1024 and text context
length 32. The transform squashes to 336 × 336 with bilinear interpolation,
converts RGB, then uses mean/std 0.5. Both encoders must be called with
`normalize=True`. Evidence: [official PE documentation](https://github.com/facebookresearch/perception_models/blob/65a4a17a587b6a89eaa0dc08f231c2cb1c285f9f/apps/pe/README.md),
[paired encoder implementation](https://github.com/facebookresearch/perception_models/blob/65a4a17a587b6a89eaa0dc08f231c2cb1c285f9f/core/vision_encoder/pe.py),
[official transforms](https://github.com/facebookresearch/perception_models/blob/65a4a17a587b6a89eaa0dc08f231c2cb1c285f9f/core/vision_encoder/transforms.py),
and [pinned checkpoint](https://huggingface.co/facebook/PE-Core-L14-336/blob/ee1fdb467ab42497fa2e60e94a79e9d7383c1cbd/PE-Core-L14-336.pt).

The model and PE code are Apache-2.0. The checkpoint is approximately 2.68 GB;
Task 08 must not download it implicitly in unit tests.

## Failure policy

- No provider may fall back to another provider or checkpoint.
- A missing revision, license acceptance, dependency, checkpoint, or matching
  text tower is a typed provider failure.
- Unit tests parse only the JSON contract and never access the network.
- Actual checkpoint download, checksum verification, GPU compatibility, and
  five-keyframe inference belong to Tasks 06–08 and remain separately audited.
