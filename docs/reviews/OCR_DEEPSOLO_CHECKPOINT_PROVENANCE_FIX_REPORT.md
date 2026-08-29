# OCR DeepSolo Checkpoint Provenance Fix Report

Date: 2026-08-22

Branch: `fix/ocr-deepsolo-checkpoint-provenance`

Starting commit: `2159148`

## Executive summary

The repository has a proven DeepSolo artifact-identity mismatch. The lock file names the destination `deepsolo_r50_totaltext_textocr.pth`, but its URL names `deepsolo_res50_finetune_synth.pth`. The fetch script writes the URL response directly to the configured destination, trusts an existing destination without hashing it, and has no expected SHA256. The runtime then loads the destination name. Consequently, bytes from a differently named source can silently reach inference under the expected-looking runtime path.

Repository configuration and the official DeepSolo model table identify the intended detector as the ResNet-50 Total-Text fine-tuned checkpoint trained with `Synth150K+MLT17+IC13+IC15+TextOCR`. However, the official checkpoint source does not publish an SHA256, and neither the official OneDrive artifact nor the current Hugging Face mirror was anonymously retrievable in this environment. An expected digest therefore cannot be established reliably.

The authoritative task says to stop and report instead of inventing an expected SHA256. No lock, fetch, runtime, or test implementation was changed. The requested integrity implementation remains blocked until a trusted maintainer supplies the checkpoint bytes or an authoritative digest.

## 1. Pre-edit mismatch proof

The audit was completed before any edit to the OCR implementation.

| Concern | Current evidence | Result |
|---|---|---|
| Lock destination | `docker/ocr/model_sources.lock.json` sets `deepsolo.filename` to `deepsolo_r50_totaltext_textocr.pth` | Expected-looking Total-Text/TextOCR name |
| Lock source | The same entry points to `.../deepsolo_res50_finetune_synth.pth` | Source basename differs from destination identity |
| Fetch behavior | `scripts/ocr/fetch_ocr_models.ps1` downloads directly to `Join-Path $dir $filename` | Source bytes are renamed without identity or integrity validation |
| Existing-file behavior | The fetch script skips any existing destination | Existing bytes are trusted without hashing |
| Runtime path | `src/mvr/enrichment/ocr/docker_ocr.py` loads `/models/deepsolo/deepsolo_r50_totaltext_textocr.pth` | Runtime trusts the renamed destination |
| Worker validation | `docker/ocr/worker/deepsolo_worker.py` checks existence, non-empty size, and obvious placeholder/HTML headers | No model identity or SHA256 enforcement |

Git history shows that the inconsistent lock URL/destination and runtime path entered together in commit `4cfe5bb` (`feat: complete OCR enrichment pipeline`). No expected model digest was found in repository history or review snapshots.

This proves the unsafe identity flow. It does **not** prove which checkpoint is contained in any current local file; treating an observed local digest as authoritative would repeat the provenance error.

## 2. Intended DeepSolo checkpoint

The intended logical model is:

- architecture: DeepSolo with a ResNet-50 backbone;
- benchmark/fine-tune target: Total-Text;
- external training data: `Synth150K+MLT17+IC13+IC15+TextOCR`;
- repository runtime configuration: `DeepSolo/configs/R_50/TotalText/finetune_150k_tt_mlt_13_15_textocr.yaml`;
- official upstream model row: Total-Text, Res-50, with the same external-training set;
- official upstream weight reference: the fine-tuned Total-Text Res-50 link associated with that row, not the separately listed generic synthetic/pre-trained checkpoint.

Official evidence:

- DeepSolo repository: <https://github.com/ViTAE-Transformer/DeepSolo>
- Audited upstream revision: `dbadae995035246bad3376c7a44c015c69e9b313`
- Revision-pinned README: <https://github.com/ViTAE-Transformer/DeepSolo/blob/dbadae995035246bad3376c7a44c015c69e9b313/DeepSolo/README.md>
- Official fine-tuned checkpoint share linked from the Total-Text Res-50 row: <https://1drv.ms/u/s!AimBgYV7JjTlgcd138p8HaXViFk-tw?e=r15pMR>

The official README distinguishes this fine-tuned Total-Text checkpoint from other pre-trained/synthetic checkpoints. That distinction makes the current `deepsolo_res50_finetune_synth.pth` mirror name insufficient proof that it is the intended artifact.

## 3. SHA256 and source resolution

### DeepSolo

- Expected SHA256: **unresolved**.
- The official repository does not publish a checksum for the linked checkpoint.
- The official OneDrive link resolved to the expected distinct share item, but artifact download was denied with HTTP 403 in this environment.
- The configured Hugging Face URL returned HTTP 401 and uses the mutable `main` revision.
- No repository-trusted manifest, release asset, checksum file, or historical lock digest was found.

An ignored local file exists at `artifacts/_ocr_models/deepsolo/deepsolo_r50_totaltext_textocr.pth`. Its observed SHA256 is:

`6342945f34781ea90340d11b089c091327115d381fceb62c87b5cd2a82efeed3`

That value is recorded only as audit evidence. Because the file was acquired through the unverified fetch path, it is not an authoritative expected digest and was not copied into the lock.

### PARSeq-Vietnamese

The same weakness exists for PARSeq-Vietnamese:

- lock destination: `best-parseq.ckpt`;
- source basename: `parseq_vi.ckpt`;
- source revision: mutable `main`;
- expected SHA256: absent and unresolved;
- fetch behavior: direct write and blind reuse;
- runtime path: `/models/parseq_vi/best-parseq.ckpt`;
- worker validation: no checksum enforcement;
- configured Hugging Face source: HTTP 401 in this environment.

An ignored local file exists at `artifacts/_ocr_models/parseq_vi/best-parseq.ckpt`. Its observed SHA256 is:

`8089b13c5ad115a96a608c6401eaab36b081393ea0a8323537b29a2dc80168f5`

This digest is likewise observational only and must not be promoted to the lock without trusted provenance.

## 4. Required fix status

The complete required implementation cannot be made safely until trusted expected digests are available for both artifacts. In particular, the repository still needs:

1. immutable artifact identities and authoritative SHA256 values in `model_sources.lock.json`;
2. a fetch path that hashes existing files before reuse and fails closed on mismatch;
3. download to a same-directory temporary file, SHA256 verification, and atomic publication;
4. cleanup of temporary files on every failure path;
5. runtime preflight using the same lock contract before either model is passed to inference;
6. focused synthetic tests for success, corrupt existing files, corrupt downloads, missing/malformed hashes, temp cleanup, and runtime rejection.

Implementing checksum plumbing with guessed or locally observed expected hashes would create a false trust anchor. Implementing it while leaving hashes optional would not meet the fail-closed requirement. The task's explicit provenance stop condition therefore applies.

## 5. Changes made

- Added this required audit/fix report only.
- Did not change OCR architecture, lock data, fetch behavior, runtime loaders, Docker images, or unrelated pipeline code.
- Did not download, add, stage, or commit model weights.
- Did not merge or push.

## 6. Verification

Focused existing synthetic tests were run without downloading models:

```text
py -3.12 -m pytest -q tests/unit/enrichment/ocr/test_weight_validation.py tests/unit/enrichment/ocr/test_docker_ocr.py
.....                                                                    [100%]
5 passed in 4.40s
```

These tests confirm the pre-existing placeholder/header validation behavior; they do not provide the missing SHA256 enforcement or satisfy the new integrity-test requirements.

Final repository hygiene results after this report was added:

- `git diff --check`: passed with no output;
- `git status --short`: this report appears under the untracked `docs/reviews/` directory alongside the pre-existing audit report; the command also lists unrelated pre-existing untracked user files and directories;
- `git diff --cached --name-only`: no output, confirming no files or model weights are staged;
- active branch: `fix/ocr-deepsolo-checkpoint-provenance`.

## 7. Unblock requirements

A trusted maintainer must provide, for both the intended DeepSolo fine-tuned checkpoint and PARSeq-Vietnamese checkpoint, at least one of:

- the authoritative checkpoint bytes through a verified transfer, from which an agreed SHA256 can be computed and independently confirmed; or
- an SHA256 published by the model owner or another explicitly approved trust root, together with an immutable source/revision.

Once those values are available, the lock, fetch verifier, runtime preflight, and synthetic tests can be implemented as one fail-closed change.

## Final verdict

`BLOCKED_BY_MODEL_PROVENANCE`
