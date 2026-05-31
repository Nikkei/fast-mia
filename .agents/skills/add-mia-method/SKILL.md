---
name: add-mia-method
description: Use when adding a new membership inference attack (MIA) method to Fast-MIA from a research paper or GitHub repository
---

# Add MIA Method

## Overview

Implement a new MIA method in Fast-MIA from a paper URL (required) and optional GitHub reference implementation. Follows a 5-phase process with human approval at each gate before proceeding.

## Invocation

```
/add-mia-method <paper-url> [github-url]
```

- `paper-url` — required (any URL: arXiv, ACL Anthology, NeurIPS, ICML, PDF direct link, etc.)
- `github-url` — optional reference implementation

## Phase 1: Read and Summarize

**Download the paper.** The source depends on the host. arXiv provides an HTML
version for most papers (but not all — newly submitted papers, or TeX sources
that fail HTML conversion, only have a PDF), so prefer HTML there and fall back
to the PDF. Other hosts (ACL Anthology, OpenReview, NeurIPS/ICML/PMLR, direct
PDF links, etc.) are downloaded as PDF. The Read tool parses PDFs natively, so a
PDF is always a valid outcome.

```bash
mkdir -p references/papers

PAPER_URL="<paper-url>"

# Try to extract an arXiv ID (e.g. 2301.12345 or 2301.12345v2) from the URL.
ARXIV_ID=$(python3 -c "
import re, sys
m = re.search(r'([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)', sys.argv[1])
print(m.group(1) if m else '')
" "$PAPER_URL")

if [ -n "$ARXIV_ID" ]; then
    # --- arXiv: try the HTML version first, fall back to the PDF ---
    if curl -fsSL "https://arxiv.org/html/${ARXIV_ID}" \
           | uvx html2text > "references/papers/${ARXIV_ID}.md" 2>/dev/null \
       && [ -s "references/papers/${ARXIV_ID}.md" ]; then
        echo "Fetched HTML version -> references/papers/${ARXIV_ID}.md"
    else
        # HTML unavailable (404) or empty: fall back to the PDF.
        rm -f "references/papers/${ARXIV_ID}.md"
        curl -fsSL "https://arxiv.org/pdf/${ARXIV_ID}" \
            -o "references/papers/${ARXIV_ID}.pdf"
        echo "HTML unavailable; saved PDF -> references/papers/${ARXIV_ID}.pdf"
    fi
else
    # --- Non-arXiv: resolve a direct PDF URL from the page URL ---
    case "$PAPER_URL" in
        *.pdf)              PDF_URL="$PAPER_URL" ;;            # already a PDF link
        *aclanthology.org*) PDF_URL="${PAPER_URL%/}.pdf" ;;   # .../2023.acl-long.1.pdf
        *openreview.net*)   PDF_URL="${PAPER_URL/forum/pdf}" ;; # forum?id=.. -> pdf?id=..
        *)                  PDF_URL="$PAPER_URL" ;;            # last resort: try as-is
    esac

    # Build a filesystem-safe slug for the output filename.
    SLUG=$(python3 -c "
import re, sys
seg = sys.argv[1].rstrip('/').split('/')[-1]
seg = re.sub(r'\.pdf$', '', seg, flags=re.I)
print(re.sub(r'[^A-Za-z0-9._-]+', '-', seg) or 'paper')
" "$PDF_URL")

    if curl -fsSL "$PDF_URL" -o "references/papers/${SLUG}.pdf" \
       && [ -s "references/papers/${SLUG}.pdf" ]; then
        echo "Saved PDF -> references/papers/${SLUG}.pdf"
    else
        rm -f "references/papers/${SLUG}.pdf"
        echo "Could not download a PDF from ${PDF_URL}"
    fi
fi
```

**Read whichever file was produced:**

- If a `.md` file was produced (arXiv HTML), read it with the Read tool.
- If a `.pdf` file was produced, read it with the Read tool directly — the Read
  tool parses PDFs natively. Use the `pages` argument for papers over 10 pages
  (max 20 pages per call; page through the whole document).

If no file was downloaded (the `curl` step failed or produced an empty file —
e.g. the host is behind a login wall, or the PDF lives at a non-obvious URL),
inform the user and ask them to provide a direct PDF URL you can download, or to
save the PDF into `references/papers/` themselves.

**Clone the GitHub reference** (if provided):

```bash
git clone --depth 1 <github-url> references/repos/<repo-name>
```

Then read relevant files (e.g. the main scoring module) with the Read tool. Prefer tracing imports to locate helper functions rather than reading every file.

Produce a summary containing:

1. **Method name** — proposed `snake_case` identifier used as the config `type` and module name
2. **Core scoring formula** — what quantity is computed per input text?
3. **Required inputs beyond `texts` and `model`** — does the method need `labels`, `tokenizer`, or neither?
4. **Configurable parameters** — name, type, default value, and whether required
5. **`run()` override needed?** — yes only when the method needs extra models, augmented prompts, or multi-step inference; state the reason explicitly
6. **Ambiguities** — for each unclear algorithmic detail, state what is unclear and which interpretation you chose

**Present this summary and wait for explicit user approval before proceeding to Phase 2.**
If the user corrects an interpretation, revise the summary and confirm again.

## Phase 2: Design

Map the algorithm to `BaseMethod`. Present the following design:

1. **Class name** — CamelCase, e.g. `MinKMethod`
2. **Module name** — snake_case matching config `type`, e.g. `mink`
3. **`process_output()` logic** — exact formula as Python pseudocode
4. **`run()` override** — yes/no and the reason (from Phase 1)
5. **Requirement flags** — state which of the three flags differ from their defaults:
   - `requires_labels = False` (default)
   - `requires_tokenizer = False` (default)
   - `requires_sampling_params = True` (default)
6. **Config parameters** — for each: name, type, default, required (yes/no)
7. **Difference from existing methods** — one sentence comparing to the closest existing method

Closest existing methods for reference:
- `loss` — mean log-prob
- `zlib` — loss divided by zlib compression size
- `lower` — loss on lowercased text
- `mink` — min-k% token log-probs
- `ref` — loss minus reference model loss
- `recall` — few-shot perplexity ratio
- `conrecall` — contrastive recall
- `samia` — self-adversarial MIA
- `dcpdd` — DC-PDD tokenizer-based scoring
- `pac` — PAC-based scoring

**Present the design and wait for explicit user approval before proceeding to Phase 3.**

## Phase 3: Implement

Make the following three changes. **For each sub-step, show the complete diff or file content to the user before writing.**

### 3a. Create `src/methods/<module_name>.py`

Use this template (replace placeholders):

```python
# Copyright (c) 2025 Nikkei Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

# Add other imports as needed (numpy, vllm, etc.)
from vllm.outputs import RequestOutput

from .base import BaseMethod


class <ClassName>(BaseMethod):
    """<One-line description of the method>"""

    def __init__(self, method_config: dict[str, Any] = None) -> None:
        super().__init__("<module_name>", method_config)
        # Extract config params; raise ValueError for missing required ones:
        # self.param = self.method_config.get("param")
        # if self.param is None:
        #     raise ValueError("<ClassName> requires 'param' in params.")

    def process_output(self, output: RequestOutput) -> float:
        token_log_probs = self._extract_token_log_probs(output)
        # Implement the scoring formula here
        ...
```

Rules:
- Line length ≤ 88 characters (ruff enforces this)
- Only import what is used
- If `run()` override is needed, add it after `process_output()` following the signature in `BaseMethod`
- Use `self.get_outputs(...)` inside `run()`, never `model.generate(...)` directly
- If loading a temporary model inside `run()`, call `self.cleanup_model(model)` at the end

### 3b. Register in `src/methods/factory.py`

Add one import and one entry. The existing `METHOD_BUILDERS` dict is defined in this file. Add:

```python
from .<module_name> import <ClassName>
```

to the import block, and:

```python
"<module_name>": <ClassName>,
```

to `METHOD_BUILDERS`.

### 3c. Export from `src/methods/__init__.py`

Add:

```python
from .<module_name> import <ClassName>
```

to the import block, and `"<ClassName>"` to `__all__`.

**Wait for user approval on all three changes before writing any files.**

## Phase 4: Test

### 4a. Add factory test

In `tests/unit/test_factory.py`, add the class import at the top:

```python
from src.methods.<module_name> import <ClassName>
```

Add a parametrize entry to `TestMethodFactory.test_create_method`:

```python
("<module_name>", <ClassName>, {<minimal_valid_params_dict>}),
```

Use the smallest set of params that allows `__init__()` to succeed. For methods with no required params, use `{}`.

### 4b. Run tests

```bash
make test
```

Expected: all tests pass. If any test fails, diagnose the failure, fix the implementation, and re-run before reporting.

### 4c. Run linting

```bash
make lint
```

Expected: no errors. Fix any ruff errors before reporting.

**Report all results to the user.**

## Phase 5: Report

Summarize what was implemented:

- Files created/modified (with paths)
- Method config `type` identifier
- Example YAML config snippet:

```yaml
methods:
  - type: "<module_name>"
    params:
      # list params with their defaults
```

- Test and lint status

Then prompt:

> Next steps: run `/commit-push-pr` when ready to commit and open a PR.
> For contributions, also update `README.md`, `docs/docs/index.md`, and `docs/docs/how-to-use.md` per `docs/docs/adding-methods.md`.

## Judgment Rules

- Surface every algorithmic ambiguity in Phase 1 — never silently pick an interpretation.
- Do not override `run()` unless Phase 1 analysis shows a clear need. The default in `BaseMethod` covers the common case.
- Validate all required config params in `__init__()` with a `ValueError` and a descriptive message.
- If the GitHub reference uses Hugging Face or PyTorch directly, adapt the algorithm to vLLM's `RequestOutput` interface — do not add new framework dependencies.
- Ruff enforces 88-character line length. Keep lines short.
