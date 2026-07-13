---
name: release
description: >
  Automates release work for Fast-MIA based on semantic versioning.
  Use this skill for any request about cutting a release or bumping the version,
  such as "release it", "bump the version", "release as v0.5.0", or "create a release".
  It infers patch/minor/major from the changes, then updates the version, opens the
  version-bump PR, and after the merge tags the release and creates the GitHub release
  (with release notes) in one flow.
---

# Release (Semantic Versioning automation)

## Overview

This skill automates release work for Fast-MIA following semantic versioning (semver).
It analyzes the changes to decide how much to bump the version, then runs the
following steps in one flow:

1. Version decision (patch / minor / major)
2. Update the `version` field in `pyproject.toml`
3. Open a version-bump PR (the `main` branch is protected — see below)
4. Merge the PR once CI is green and it has an approval
5. Tag the merged commit on `main` and push the tag
6. Create the GitHub release (with release notes)

**Important**: `main` is protected by repository rules. Direct pushes are rejected
(`GH013: Changes must be made through a pull request`), and PRs additionally require a
review approval before they can be merged. The version bump therefore always goes
through a PR, and the tag is only created **after** that PR is merged.

## Semantic versioning rules

Analyze the commit and PR history since the previous tag and decide the bump using
the criteria below. If the user explicitly specifies a version (e.g. "release as
v0.5.0"), that takes precedence.

### MAJOR (x.0.0)

When any of the following applies:

- A commit containing `BREAKING CHANGE:`
- A commit with a `!` marker such as `feat!:` or `fix!:`
- A change that breaks backward compatibility (e.g. changing the config YAML schema,
  changing a `BaseMethod` interface such as `process_output()` / `run()`, or removing
  a config key)

### MINOR (0.x.0)

When any of the following applies:

- A commit starting with `feat:` (new functionality)
- A new MIA method added under `src/methods/` (e.g. the Ref or Neighbour methods)
- A new data loader, script, or user-facing capability
- A vLLM version upgrade that unlocks new behavior

### PATCH (0.0.x)

When none of the above applies:

- `fix:` — bug fix (e.g. error handling in a method)
- `chore:` — maintenance work
- `docs:` — documentation updates
- `refactor:` — refactoring
- `build(deps):` — dependency bumps (e.g. Dependabot updates)
- Other small changes

### Notes

- When multiple kinds of change are mixed, apply the highest bump (e.g. feat + fix →
  minor).
- During the `0.x.y` development phase it is common to bump even breaking changes as a
  minor. Fast-MIA is currently on `0.x`, so follow that convention.

## Procedure

### 1. Check the current state

```bash
# Get the latest tag (fetch tags first so the local repo is up to date)
git fetch --tags
git describe --tags --abbrev=0

# List commits since the latest tag
git log <latest-tag>..HEAD --oneline

# Confirm the working tree is clean and in sync with the remote
git status
git log origin/main..HEAD --oneline

# Confirm CI on main is green
gh run list --branch main --limit 3
```

**Important**: If there are uncommitted changes, ask the user whether to commit them
before releasing.

Note: a previous release's version-bump commit may show up in this list (for v0.4.0 the
tag was cut before its bump PR merged). Version-bump PRs are release plumbing — always
exclude them from the release notes.

### 2. Decide the version

Analyze the commit and PR history and determine the next version using the rules
above. Present the decision and reasoning to the user and get confirmation.

Example:
```
Previous: v0.3.0
Changes:
  - [Feature] Ref method (new MIA method)
  - [Feature] Neighbour method (new MIA method)
  - upgrade vLLM to 0.23.0
  - docs: quantization guide
→ minor change present → proposing v0.4.0
```

### 3. Update the version

Update the `version` field in `pyproject.toml`.

### 4. Open the version-bump PR

`main` is protected: never `git push` the bump commit straight to `main` (it is rejected
with `GH013`). Always go through a release branch and a PR.

```bash
git switch -c release/v<new-version>
git add pyproject.toml
git commit -m "chore: bump version to <new-version>"
git push -u origin release/v<new-version>

gh pr create --base main --head release/v<new-version> \
  --title "chore: bump version to <new-version>" \
  --body "<summary of the changes since the previous tag>"
```

### 5. Merge the PR, then tag

Wait for CI, then merge. The base branch policy requires a **review approval**, so
`gh pr merge` fails with `the base branch policy prohibits the merge` until someone
approves. Do not bypass this on your own: report the block to the user and let them
decide (get a reviewer, use `--auto` for auto-merge on approval, or merge with
`--admin` if they have the rights).

```bash
# Wait for CI to finish
gh pr checks <pr-number> --watch

# Merge once approved (add --auto to merge automatically after approval)
gh pr merge <pr-number> --squash --delete-branch
```

Only after the PR is merged, tag the resulting commit on `main`:

```bash
git switch main && git pull
git tag v<new-version>
git push origin v<new-version>
```

### 6. Create the GitHub release

Generate release notes from the merged PRs since the previous tag and create the
release with `gh release create`.

```bash
gh release create v<new-version> --title "v<new-version>" --notes "<release-notes>"
```

Fast-MIA release notes group PRs into categorized sections. Each entry follows the
GitHub-generated style `* <PR title> by @<author> in <PR-URL>`, and the notes end with
a `**Full Changelog**` compare link. Use these section headings (include only the
sections that have changes). Order them as listed below, from most impactful to least:

- **Breaking Changes** — backward-incompatible changes that drive a major bump
  (config YAML schema changes, `BaseMethod` interface changes, removed config keys).
  Include this first when present, and call out the required migration.
- **Methods** — new or updated MIA methods under `src/methods/`
- **Data & Evaluation** — new data loaders / dataset support (WikiMIA, Mimir, etc.),
  and changes to the evaluator, metrics (AUROC, FPR@95, TPR@5), or visualizations
- **Bug Fixes** — `fix:` changes such as error handling in a method
- **Dependency versions** — vLLM upgrades, Dependabot bumps, and other dependency
  changes
- **Utils** — tooling, scripts, skills, CI, and infrastructure (e.g. Google Cloud
  support)
- **Documentation** — docs and README updates

Anything that does not fit the above can go under a generic **Other Changes** section
at the end.

Format (ref: v0.4.0 release):
```markdown
## What's Changed

### Methods

* [Feature] Ref Method by @hiromu166 in https://github.com/Nikkei/fast-mia/pull/43
* [Feature] Neighbour method by @hiromu166 in https://github.com/Nikkei/fast-mia/pull/54

### Dependency versions

* upgrade vLLM to 0.23.0 by @hiromu166 in https://github.com/Nikkei/fast-mia/pull/58
* build(deps): bump astral-sh/setup-uv from 8.1.0 to 8.2.0 by @dependabot[bot] in https://github.com/Nikkei/fast-mia/pull/55

### Utils

* Add running on Google Cloud option by @upura in https://github.com/Nikkei/fast-mia/pull/44
* skills: add-mia-method by @hiromu166 in https://github.com/Nikkei/fast-mia/pull/53

### Documentation

* [Docs] How to add methods by @hiromu166 in https://github.com/Nikkei/fast-mia/pull/48

**Full Changelog**: https://github.com/Nikkei/fast-mia/compare/v0.3.0...v0.4.0
```

Tip: `gh release create --generate-notes` produces the base `* <title> by @<author> in
<PR-URL>` list and the `**Full Changelog**` link automatically. Group those entries
into the sections above before publishing.

## Cautions

- Confirm CI is green before releasing (the GitHub Actions workflow).
- If the user explicitly specifies a version, skip the semver inference and use that
  version.
- Push, PR creation, merge, and release creation affect the remote: run them only after
  the user confirms.
- Never push to `main` directly and never create the tag before the bump PR is merged —
  a tag on an unmerged commit has to be deleted and recreated.
- Exclude version-bump PRs from the release notes.
- `uv.lock` also carries a `fast-mia` version entry, and it has drifted from
  `pyproject.toml` in past releases. Check it, and mention it to the user if it is out
  of sync (`uv lock` refreshes it).
