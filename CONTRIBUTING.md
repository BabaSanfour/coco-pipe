# Contributing to CoCo Pipe

Thank you for your interest in contributing to CoCo Pipe! We welcome contributions from the community to help improve this project.

## Getting Started

### 1. Fork and Clone
Fork the repository on GitHub and clone it to your local machine:
```bash
git clone https://github.com/your-username/coco-pipe.git
cd coco-pipe
```

### 2. Set Up Environment
We recommend using a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the package in editable mode with development dependencies:
```bash
pip install -e .[dev]
```

### 3. Install Pre-commit Hooks
This project uses `pre-commit` to ensure code quality. Install the hooks to run automatically before every commit:
```bash
pre-commit install
```
This will check for formatting (Black), import sorting (isort), and linting (Ruff).

## Development Workflow

1.  **Create a Branch**: Always work on a new branch for your feature or fix.
    ```bash
    git checkout -b feature/my-new-feature
    ```

2.  **Make Changes**: Write your code and tests.

3.  **Run Tests**: Ensure all tests pass before submitting.
    ```bash
    pytest
    ```

4.  **Build Docs**: If you changed documentation, verify it builds.
    ```bash
    cd docs
    make html
    ```

5.  **Commit and Push**:
    ```bash
    git add .
    git commit -m "feat: description of changes"
    git push origin feature/my-new-feature
    ```

6.  **Open a Pull Request**: Submit a PR to the `main` branch.

## Code Style
- **Formatting**: We use [Black](https://github.com/psf/black).
- **Imports**: We use [isort](https://github.com/PyCQA/isort).
- **Linting**: We use [Ruff](https://github.com/astral-sh/ruff).

These are enforced via pre-commit hooks.

## Releasing
Releases are automated via GitHub Actions when a tag starting with `v` is pushed (e.g., `v0.1.0`).
