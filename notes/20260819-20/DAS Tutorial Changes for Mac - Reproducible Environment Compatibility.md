
**Source (original):** Atticus Geiger's official DAS coding tutorial (`stanfordnlp/pyvene`) 
**Source (mine):** Same tutorial, after ~2 days debugging to get it running reproducibly on macOS (Apple Silicon) .

---

## Summary

Diffed both notebooks cell-by-cell (ignoring comments/whitespace) — **8 cells changed**, all falling into 3 categories: (1) CUDA→CPU device fixes, (2) a `transformers` API rename, (3) minor environment/debug additions. No changes to the actual DAS logic, math, or model architecture — every change is purely about making the _identical_ method run correctly on different hardware/library versions.

---

## 1. CUDA → CPU device fixes (6 cells)

**The core issue:** the original notebook was written assuming an Nvidia GPU (`"cuda"` / `"cuda:0"`). Macs have no CUDA support at all (Apple Silicon uses Metal/MPS instead) — every hardcoded `"cuda"` reference had to be changed to `"cpu"`(MPS was tried too, but hit a separate blocking issue — see §3).

**Why CPU and not MPS:** attempted MPS first — hit a confirmed, still-open `pyvene` bug (GitHub issue #67 https://github.com/stanfordnlp/pyvene/issues/67): `torch.matrix_exp`, used internally by the orthogonal-parametrization of the rotation matrix R, is not implemented for Apple's MPS backend (`NotImplementedError: aten::linalg_matrix_exp`). A `PYTORCH_ENABLE_MPS_FALLBACK=1` env var workaround exists, but going all-CPU was simpler and fully avoided the device-mismatch bugs encountered along the way (model on one device, data tensors on another — a separate, confusing failure mode hit mid-debugging).

## 2. `transformers` API rename (cell 56)

```diff
- evaluation_strategy="epoch",
+ eval_strategy="epoch",
```

`evaluation_strategy` was renamed to `eval_strategy` in a recent `transformers` release. The original notebook (Feb 2025) predates the rename; running against a newer `transformers` install threw `TypeError: unexpected keyword argument 'evaluation_strategy'`.
[Check https://github.com/huggingface/transformers/commit/60d5f8f9f04026cb801d0dc5158bf4531e250072]
## 3. Minor additions/tweaks

- **Cell 4** — replaced the original's `try/except ModuleNotFoundError: !pip install ...` block with an explicit version-check print (`torch.__version__`, `transformers.__version__`, `pyvene.__file__`) — used throughout debugging to confirm exactly which environment/kernel was active, since kernel-mismatch issues (wrong Python interpreter selected in VS Code) were a recurring, confusing source of "identical code, different results" during debugging.
- **Cell 6** — seed changed from `42` (original) to `0` — arbitrary; used to confirm results were genuinely reproducible/seed-sensitive (different seeds → legitimately different final accuracy, as expected for DAS's non-convex optimization) rather than being a sign anything was broken.
- **Cell 56** — `num_train_epochs` bumped from `3` → `4`; [testing purposes]

---

## What this diff does _not_ show (resolved before reaching final code state)
- A HuggingFace label-format bug (`problem_type` silently auto-routing to `multi_label_classification` given float one-hot labels instead of integer class indices) caused the base MLP to train to exactly `ln(2)` loss (frozen, no learning at all) for a full day of debugging — turned out to be **version drift** (the pip-installed `transformers`/`torch`/`pyvene` versions differed from what the tutorial was originally built/tested against in Feb 2025), not a code bug per se. Pinning versions to match the tutorial's original release date resolved it entirely — the label-format/`problem_type` workaround wasn't actually needed once versions matched.
- Several rounds of "identical results across different fixes" were eventually traced to a **stale/non-restarting Jupyter kernel** in VS Code (compounded by a "Python 3.9 no longer supported" warning) — not the code changes being wrong.
- Root-caused the CUDA/MPS issue by finding the exact upstream GitHub issue (`pyvene` #67) confirming the `matrix_exp`/MPS gap is a known, unresolved library limitation — not something fixable from the notebook side alone.