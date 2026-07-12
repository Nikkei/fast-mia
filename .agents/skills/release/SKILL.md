---
name: release
description: >
  Automates release work for Fast-MIA based on semantic versioning.
  Use this skill for any request about cutting a release or bumping the version,
  such as "release it", "bump the version", "release as v0.5.0", or "create a release".
  It infers patch/minor/major from the changes, then updates the version, commits,
  tags, pushes, and creates the GitHub release (with release notes) in one flow.
---

# Release (Semantic Versioning automation)

## Overview

This skill automates release work for Fast-MIA following semantic versioning (semver).
It analyzes the changes to decide how much to bump the version, then runs the
following steps in one flow:

1. Version decision (patch / minor / major)
2. Update the `version` field in `pyproject.toml`
3. Create the commit and tag
4. Push to the remote
5. Create the GitHub release (with release notes)

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
# Get the latest tag
git describe --tags --abbrev=0

# List commits since the latest tag
git log <latest-tag>..HEAD --oneline --no-merges

# Confirm the working tree is clean
git status
```

**Important**: If there are uncommitted changes, ask the user whether to commit them
before releasing.

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

### 4. Commit, tag, and push

```bash
# Commit
git add pyproject.toml
git commit -m "chore: bump version to <new-version>"

# Create the tag
git tag v<new-version>

# Push (both the commit and the tag)
git push && git push origin v<new-version>
```

### 5. Create the GitHub release

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
- Run the push and release creation only after the user confirms (they affect the
  remote).
