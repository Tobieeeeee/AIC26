# Scripts

Nơi đặt các entrypoint chạy offline như:

- `build_manifest.py`
- `extract_keyframes.py`
- `enrich_keyframes.py`
- `rebuild_index.py`
- `run_benchmark.py`
- `start_stack.py`

## Multi-embedding release commands

- `verification/audit_multi_embedding_release.py`: audits provider routing,
  dimensions, RRF boundaries, Task05B fields, and Git hygiene.
- `verification/run_multi_embedding_ablation.py`: runs the seven configured
  structural ablations.
- `operations/run_full_corpus.py --dry-run`: validates and prints the
  approval-gated full-corpus plan without launching extraction or index builds.

See `docs/MULTI_EMBEDDING_FINAL_RELEASE_REPORT.md` for exact commands,
limitations, and rollback guidance.

Mục tiêu cuối: corpus có thể rebuild bằng script, không cần mở notebook và chạy cell thủ công.
