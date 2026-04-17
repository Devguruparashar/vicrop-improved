This folder is packaged as a clean starting point for a new repository.

Contents:

- Main project files: copied from the improved `mllms_know` workspace version.
- Colab support files: `COLAB.md`, `colab_requirements.txt`, `colab_run_all.py`, `mllms_know_colab.ipynb`.
- Progress artifacts: stored in `progress_artifacts/`.

Recommended next steps:

1. Create a new GitHub repository.
2. Upload the contents of this folder as the repo root.
3. Run smoke tests in a clean environment, starting with `textvqa` for `llava` and `blip`.

Notes:

- `.git` was intentionally excluded.
- This package includes the local `transformers/` directory used by the project.
