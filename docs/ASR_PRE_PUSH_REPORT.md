# ASR pre-push verification report

Date: 2026-08-20
Branch: `feat/keyframe-asr`
Target branch: `origin/main`
Verdict: **LOCAL GO; PR/CI/REVIEW REQUIRED**

## 1. Scope

This report records the checks performed before proposing the per-keyframe ASR
feature for integration into `main`. Normal feature work must be pushed to the
feature branch and merged through a reviewed Pull Request; it must not be
pushed directly to `main`.

Verified ASR flow:

```text
authoritative keyframe mapping
  -> independent audio window [t - 3 s, t + 3 s]
  -> Silero VAD
  -> ChunkFormer Vietnamese ASR only for detected speech
  -> one canonical JSON artifact per keyframe
  -> manifest.json + errors.json
```

## 2. Git and synchronization checks

| Check | Result | Evidence |
|---|---|---|
| Feature branch, not `main` | PASS | `feat/keyframe-asr` |
| Working tree clean before report creation | PASS | `git status -sb` contained only the branch name |
| Latest remote state fetched | PASS | `git fetch --prune origin` completed |
| Up to date with `origin/main` | PASS | behind `0`, ahead `2` |
| Merge conflicts | PASS | no unmerged paths or conflict markers |
| HEAD | INFO | `6088d3a97e10c44e64f8260fa50bac0f75773675` |
| Compared `origin/main` | INFO | `07651bec52e268467ba20fd877d2670989e65a36` |

The feature diff against `origin/main` contains 24 files with 1,801 insertions
and 8 deletions. Its scope is limited to ASR source, CLI/configuration, Docker,
documentation, tests, and ignore rules.

## 3. Repository safety checks

| Check | Result |
|---|---|
| `git diff --check origin/main...HEAD` | PASS |
| Conflict-marker scan | PASS |
| Model/checkpoint files in feature diff | none |
| Videos, datasets, ZIP files, or generated ASR output in feature diff | none |
| FAISS/index/database artifacts in feature diff | none |
| Credentials, tokens, or private machine paths | none |
| Tracked files violating model/output ignore rules | none |
| Largest changed file | 28,917 bytes |

Model weights, datasets, and generated outputs remain outside Git. The
repository includes `.dockerignore` and `.gitignore` protection for these
runtime files.

## 4. Automated regression tests

The complete suite was executed in an isolated Python 3.11 container with the
CI dependency groups and a CPU-compatible Torch/Torchvision pair.

```text
253 passed, 6 skipped, 0 failed
```

The six skips are opt-in real-runtime tests and do not represent failures.
Real ASR inference is verified separately below.

## 5. Docker verification

Build command:

```powershell
docker build -f docker/asr/Dockerfile.cpu -t mvr-asr:prepush .
```

| Check | Result |
|---|---|
| Clean image build from current source | PASS |
| Python package compilation inside image | PASS |
| CLI startup | PASS |
| `build-asr-features` command present | PASS |
| Merged `enrich-ocr` command retained | PASS |
| Dataset/model/output included in build context | no |

Built image identity:

```text
tag: mvr-asr:prepush
digest: sha256:e79e59338cb2e4bad783b5112f0ad7d17282e8c6a81370ef36bb8e5c736cc30b
python: 3.11.10
device: cpu
```

The image is a local verification artifact and must not be committed.

## 6. Real five-keyframe smoke test

The cleanly built image processed keyframes 1-5 from `L21_V001` using the
read-only model cache and a new output directory.

```text
output inside container: /output/asr-prepush-5
```

| Metric | Result |
|---|---:|
| Keyframes total | 5 |
| Keyframes with speech | 4 |
| Keyframes without speech | 1 |
| Speech segments | 4 |
| ASR calls | 4 |
| Failed keyframes | 0 |
| Processing errors | 0 |
| Runtime warnings in `errors.json` | 0 |
| Schema/mapping/window violations | 0 |
| Artifacts matching verified baseline SHA-256 | 5/5 |
| Pipeline elapsed time | 87.854 seconds |

The no-speech keyframe correctly produced:

```json
{
  "text": "",
  "segments": []
}
```

The smoke test verifies execution, deterministic artifacts, mapping identity,
absolute segment timestamps, and VAD gating. It does not establish ASR quality
or word-error rate.

## 7. Non-blocking runtime messages

- `torch_npu` is not installed. This is expected for the verified CPU runtime.
- ChunkFormer currently emits an upstream `torch.load(weights_only=False)`
  future warning. It did not affect checkpoint loading or inference.

## 8. Remaining gates

- [x] Feature branch is synchronized with current `origin/main`.
- [x] Working tree was clean before generating this report.
- [x] Diff scope and repository safety checks pass.
- [x] Full regression suite passes.
- [x] Docker builds from current source.
- [x] Real five-keyframe ASR smoke test passes.
- [ ] Commit this report and the updated progress line.
- [ ] Push `feat/keyframe-asr` to the remote feature branch.
- [ ] Open a Pull Request targeting `main`.
- [ ] Wait for GitHub CI to pass.
- [ ] Obtain required review and merge through the repository workflow.

GPU inference, full-dataset production inference, ASR retrieval/indexing, and
quality benchmarking remain separate gates or future tasks. They are not
required to accept this per-keyframe artifact-generation feature.

## 9. Recommended next commands

```powershell
git add docs/ASR_PRE_PUSH_REPORT.md docs/CODEX_PROGRESS.md
git commit -m "docs: add ASR pre-push verification report"
git push -u origin feat/keyframe-asr
```

After pushing, open a Pull Request from `feat/keyframe-asr` to `main`. Do not
push the feature commit directly to `main`.
