# Contributing to coco-pipe

Thank you for your interest in contributing to coco-pipe.

We welcome contributions of all kinds, including:

* Bug reports
* Feature requests
* Documentation improvements
* Examples and tutorials
* Tests and infrastructure improvements
* New analysis modules and pipelines
* Performance optimizations
* Research and benchmarking contributions

We follow the Scientific Python development model:

* Development occurs on feature branches.
* Changes are reviewed through pull requests.
* All code must pass automated checks before merging.
* New functionality should include tests whenever practical.
* Documentation should be updated when user-facing behavior changes.

---

## Getting Started

### 1. Fork and Clone

Fork the repository on GitHub and clone it locally:

```bash
git clone https://github.com/your-username/coco-pipe.git
cd coco-pipe
```

### 2. Create a Virtual Environment

We recommend using a dedicated virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Development Dependencies

For most contributors:

```bash
pip install -e .[dev,test]
```

If you plan to work on documentation:

```bash
pip install -e .[dev,test,docs]
```

If you are working on optional modules that require additional dependencies (e.g., foundation models, dimensionality reduction, MOABB, or EEG pipelines):

```bash
pip install -e .[full,test]
```

---

## Verify Your Installation

After installation, verify that your development environment is working correctly:

```bash
pytest -q
pre-commit run --all-files
```

These commands should complete successfully before you begin development.

---

## Development Workflow

### 1. Create a Branch

Always work on a dedicated branch:

```bash
git checkout -b feature/my-new-feature
```

### 2. Make Your Changes

Implement your changes and add or update tests where appropriate.

Contributions should:

* Include tests when applicable
* Maintain backward compatibility whenever possible
* Include documentation updates when behavior changes
* Keep pull requests focused and reviewable

Large changes should be split into smaller pull requests whenever practical.

---

## Code Style

coco-pipe uses automated tooling to maintain a consistent codebase.

### Install Pre-commit

```bash
pre-commit install
```

### Run Checks Manually

```bash
pre-commit run --all-files
```

The pre-commit pipeline includes:

* Ruff linting
* Ruff formatting
* Import sorting
* YAML/TOML/JSON validation
* Merge-conflict detection
* Notebook cleanup
* Repository consistency checks

All checks are automatically executed in CI.

---

## Running Tests

### Standard Test Suite

The standard test suite is the same one executed by GitHub Actions on every pull request.

Run:

```bash
pytest
```

Heavy foundation-model tests are excluded by default through:

```toml
addopts = "-m 'not real_checkpoints'"
```

No network access or Hugging Face token is required for the standard test suite.

### Running Heavy Foundation Tests

Some tests validate integration with real pretrained checkpoints.

These tests:

* Download real models from Hugging Face
* Require network access
* May take considerably longer to execute

Enable them explicitly:

```bash
export HF_TOKEN=hf_xxx
export COCO_PIPE_RUN_REAL_FOUNDATION=1

pytest -m real_checkpoints
```

The real-training tests additionally require:

```bash
export COCO_PIPE_RUN_REAL_FOUNDATION_TRAINING=1
```

and a CUDA-capable GPU.

Never commit authentication tokens to the repository.

For additional details, see the README section on foundation-model testing.

---

## Building Documentation

Install documentation dependencies:

```bash
pip install -e .[docs]
```

Build the documentation locally:

```bash
sphinx-build -b html docs/source docs/build/html
```

The generated site will be available in:

```text
docs/build/html
```

Documentation builds are automatically validated in CI.

---

## Changelog Entries

We use Towncrier to generate the project changelog.

Most pull requests should include a changelog fragment inside:

```text
changelog.d/
```

The filename should follow the pattern:

```text
<PR_NUMBER>.<TYPE>
```

Examples:

```text
123.feature
456.bugfix
789.doc
999.misc
```

Use the following fragment types:

| Type    | Purpose                                                  |
| ------- | -------------------------------------------------------- |
| feature | New functionality                                        |
| bugfix  | Bug fixes                                                |
| doc     | Documentation improvements                               |
| misc    | Refactoring, CI, maintenance, and other internal changes |

Example fragment:

```text
Added support for REVE foundation-model embeddings.
```

The CI workflow automatically validates changelog fragments.

For very small maintenance changes (e.g., typo fixes or comment-only updates), maintainers may choose to skip changelog requirements.

---

## Continuous Integration

Every pull request automatically runs:

* Code quality checks
* Unit tests
* Documentation builds
* Dependency review
* Security analysis

All required checks must pass before a pull request can be merged.

---

## Pull Requests

Before opening a pull request, ensure:

* [ ] All tests pass
* [ ] Documentation builds successfully
* [ ] Pre-commit checks pass
* [ ] New functionality includes tests when appropriate
* [ ] A changelog fragment has been added when required

Push your branch:

```bash
git push origin feature/my-new-feature
```

Then open a pull request against the `main` branch.

Please include:

* A clear description of the change
* The motivation for the change
* Related issue numbers
* Performance or benchmark results when relevant

---

## Reporting Bugs

Before opening a bug report:

1. Search existing issues to avoid duplicates.
2. Check the documentation and README.
3. Verify the issue on the latest version of coco-pipe.
4. Provide a minimal reproducible example whenever possible.

Useful information includes:

* Operating system
* Python version
* coco-pipe version
* Complete traceback
* Minimal code example

---

## Commit Messages

We recommend using Conventional Commit-style messages.

Examples:

```text
feat: add support for REVE embeddings
fix: resolve MOABB loader crash
docs: update dimensionality reduction tutorial
test: improve foundation model coverage
ci: update GitHub Actions workflow
refactor: simplify dataset loading pipeline
```

---

## New Contributors

If you are looking for a place to start, consider issues labeled:

* good first issue
* documentation
* help wanted

We are happy to provide guidance for first-time contributors.

---

## Releases

Releases are managed by maintainers through GitHub Actions.

Contributors do not need to perform any release-related actions.

---

## Questions

If you have questions, suggestions, or would like feedback before implementing a large change, please open a GitHub issue or start a discussion.

We appreciate all contributions to coco-pipe.
