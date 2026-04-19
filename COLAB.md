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
- For a RAM-aware LLaVA smoke test on TextVQA, use `smoke_test_llava_textvqa_rel_att.py`. It writes a tiny on-disk subset into `data/textvqa/data.json`, runs 4-bit LLaVA with a chosen method such as `rel_att` or `grad_att`, scores the result, and restores the full dataset file afterward.
- For the same RAM-aware workflow on BLIP, use `smoke_test_blip_textvqa.py`. It follows the same subset/restore pattern and runs BLIP in 4-bit mode for Colab-friendly TextVQA smoke tests.
- For prepared datasets beyond TextVQA, use `smoke_test_subset.py`. It works for both `llava` and `blip`, writes a small subset into the task's `data.json`, runs the selected method, scores the results, and restores the original file afterward.
