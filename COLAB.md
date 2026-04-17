# Google Colab

Use [mllms_know_colab.ipynb](./mllms_know_colab.ipynb) to run this repository in Google Colab.

What the notebook does:

- clones the repo into `/content`
- installs a minimal Colab-friendly dependency set
- installs the repo's modified `transformers` package
- optionally logs into Hugging Face for gated models
- prepares TextVQA in the repo's expected JSON format
- runs the evaluation sequentially on a single GPU with `colab_run_all.py`
- computes scores with `get_score.py`

Notes:

- `qwen2_5` is the most practical default for Colab.
- `llava` and `blip` are much heavier and generally need a higher-memory GPU runtime.
- The original `run_all.sh` is multi-GPU oriented; `colab_run_all.py` is the single-GPU Colab replacement.
