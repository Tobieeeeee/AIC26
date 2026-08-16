# 06 — Final Verification, Push, and Pull Request

## Goal

Verify the reused MVR baseline, clean the Git state, push the feature branch, and create a Pull Request to `main`.

Do not implement new features in this phase.

## Required preflight

Read:

```text
AGENTS.md
codex/00_MASTER_PLAN.md
codex/06_VERIFY_AND_PUBLISH.md
docs/LAB_REUSE_MAP.md
docs/CODEX_PROGRESS.md
README.md
```

## Full repository audit

Run:

```bash
git rev-parse --show-toplevel
git status -sb
git branch --show-current
git log --oneline --decorate -10
git remote -v
```

The expected feature branch is:

```text
feat/reuse-lab-baseline
```

Do not force-switch or overwrite unrelated work.

## Check ignored/generated files

Verify that the following are not staged or tracked accidentally:

- virtual environments
- raw videos
- datasets
- generated keyframes
- generated embeddings
- FAISS indexes
- Elasticsearch runtime data
- model weights
- `.env`
- secrets
- caches
- temporary output

Do not delete local ignored files.

## Test

Run the lightweight suite:

```bash
pytest -q
```

If additional verified project commands exist, run them as appropriate.

Do not download huge models only for final verification.

Report heavy/integration tests honestly as:

- passed
- failed
- skipped
- not run

with reasons.

## Verify documented commands

Check that README commands correspond to actual code.

At minimum, verify the available form of:

```text
environment setup
ingestion
index building
semantic search
API startup
tests
```

Do not leave README claiming commands work when they do not.

## GitHub CLI

Run:

```bash
gh --version
gh auth status
```

If GitHub CLI is unavailable or unauthenticated:

- stop publication;
- do not request a plaintext token;
- report exactly what must be configured.

Local verification/commits may still be completed.

## Remote safety

Run:

```bash
git remote -v
```

If `origin` exists, verify it is the intended MVR repository.

Do not overwrite an unrelated remote.

If no intended GitHub repository exists and creation is required, create:

```text
mvr-competition
```

as a **PRIVATE** repository.

Do not make it public without explicit approval.

## Push

Push the feature branch:

```bash
git push -u origin feat/reuse-lab-baseline
```

Never force push.

Do not push feature implementation directly to `main`.

## Pull Request

Create a PR:

```text
feat/reuse-lab-baseline
        ↓
main
```

Suggested title:

```text
feat: build runnable MVR baseline from existing labs
```

PR description should include:

```markdown
## Goal

Reuse the team's existing lab implementations to create the first runnable MVR baseline.

## Reused Sources

- Shot Boundary Lab
- Lab 3
- Lab 4
- Lab 6
- FastAPI Lab

## Implemented

List only components that actually work.

## Reuse Policy

Algorithms and default behavior were preserved wherever practical.
No model upgrades or optimization were performed during this phase.

## Verification

List tests run, passed, failed, skipped, CLI commands verified, and API commands verified.

## Known Limitations

List unfinished or optional components.

## Next Phase

Refactor and benchmark modules only after this baseline is accepted.
```

Do not merge the PR automatically.

## Final verification

Run:

```bash
git status -sb
git branch -vv
git remote -v
git log --oneline --decorate -10
```

Verify the remote feature branch and PR exist.

## Final report

Report:

### Source analysis
- labs found
- labs reused

### File mapping
- major source → destination mappings

### Runnable features
- what now works end-to-end

### Commands
- setup
- ingestion
- build index
- search
- API
- tests

### Tests
- passed
- failed
- skipped
- not run

### Git
- branch
- commit hashes
- commit messages
- remote URL
- push status
- PR URL
- working tree status

### Deviations
- every meaningful change from original lab behavior

### Remaining work
- clearly separated future work

## Definition of done

This phase is complete only when:

- repository state is audited;
- tests are reported honestly;
- generated/secret files are not committed;
- feature branch is pushed when GitHub access is available;
- PR to `main` is created;
- PR is not auto-merged;
- final report is complete.
