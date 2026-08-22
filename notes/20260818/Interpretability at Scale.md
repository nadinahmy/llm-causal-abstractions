Identifying Causal Mechanisms in Alpaca

---
- Introduces **Boundless DAS**, a scaled-up version of Distributed Alignment Search (DAS) that ==removes the need for brute-force search over subspace dimensionality by learning it directly via gradient descent==. This makes causal-abstraction-based interpretability tractable for billion-parameter LLMs for the first time.

Some background: DAS → Boundless DAS
- **DAS** learns a rotation matrix aligning a subspace of neural activations with an interpretable causal variable, but still requires manually searching over *how many dimensions* (k) to allocate — infeasible at scale (e.g., testing all k from 1–4096).
- **Boundless DAS** replaces this manual search with a learnable *boundary index* parameter per variable, using a sigmoid-based soft mask (annealed toward binary during training) to automatically learn subspace size.
- Complexity drops from O(n×m) → O(m), where n = representation dimensionality, m = number of causal variables.

Boundary Mask Mechanism:
- Plain DAS needs to know *how many dimensions* of a d-dimensional rotated representation space Y to allocate to each causal variable — previously found via manual brute-force search (infeasible at scale, e.g. d=4096 for Alpaca).
With boundless DAS ---> Learn the **boundaries between variable-subspaces** directly via gradient descent, instead of searching over sizes by hand.

How they do this:
**1. Boundary index `bⱼ`**
- A learnable, continuous cut point along the d dimensions (like marking a chunk boundary at position 50.3 out of 4096).
- Ordering constraint: `b₀ = 0 < b₁ < b₂ < ... < d`
- Segment for variable `Zⱼ` = dimensions between `bⱼ` and `bⱼ₊₁`

**2. Boundary mask `Mⱼ`** — a soft "which dims belong to me" indicator:

$$(M_j)_k = \text{sigmoid}\left(\frac{k - b_j}{\beta}\right) \times \text{sigmoid}\left(\frac{b_{j+1} - k}{\beta}\right)$$

| Term | Role |
|---|---|
| `sigmoid((k−bⱼ)/β)` | smooth "turn on after `bⱼ`" switch |
| `sigmoid((bⱼ₊₁−k)/β)` | smooth "turn off after `bⱼ₊₁`" switch |
| product | ≈1 only when `bⱼ ≤ k ≤ bⱼ₊₁`, ≈0 elsewhere → smooth rectangular window |

- ==A hard step function (0/1 cutoff) isn't differentiable → can't train `bⱼ` with gradients.==
- Sigmoid product = **differentiable approximation** of a step function.

**3. Temperature `β` (annealed during training)**
- ==**High β** → soft/blurry edges → gradients flow → `bⱼ` can move and find good boundaries.==
- ==**β → 0** → edges sharpen into a hard cutoff → clean binary partition at eval time.==
- Strategy: *soft during training (learnable) → hard at convergence (interpretable)*.
- It is mentioned in the paper that this is inspired by neural PDE learning methods (Wu et al. [57]).

At convergence, masks `M₁, M₂, ...` ==partition Y into **non-overlapping, contiguous dimension blocks**, one per causal variable, plus a leftover "unaligned" residual block — an orthogonal decomposition of the representation space, learned end-to-end instead of searched by hand.==

Why is this important for DAS?
- Turns subspace-size search from **O(n×m)** (test many candidate sizes × variables) into **O(m)** (one learnable boundary per variable) — ==the key enabler for scaling DAS to billion-parameter LLMs.==

Weighted Distributed Interchange Intervention (Equation 4)
Why we need this at all?
- A **hard** interchange intervention (from earlier in the paper) swaps a variable's subspace in the base input's representation for the value that subspace takes in a **source** input — a clean yes/no replacement. But hard swaps aren't differentiable, and Boundless DAS needs to back-prop through the boundary index `bⱼ` to learn it. So instead of "replace or don't," ==Weighted DII **blends** base and source per-dimension, in proportion to how much the mask says that dimension belongs to a given variable.==

Equation 4

$$F^*_N(b) = R^{-1}\Big((1 - \sum_{j=1}^k M_j) \circ R(F_N(b)) + \sum_{j=1}^k \big(M_j \circ R(F_N(s_j))\big)\Big)$$

- `R(F_N(b))`, `R(F_N(sⱼ))`: rotate the base's and each source's representations into Y-space so masks apply meaningfully.
- `Σⱼ Mⱼ`: **sum the mask vectors across all variables**, dimension by dimension — ==tells you, per dimension, how much of it is claimed by _some_ variable in total (regardless of which one).==
- `1 − Σⱼ Mⱼ`: the _unclaimed_ portion of each dimension → multiplied by the base's own value → "<mark style="background: #FF5582A6;">leave unclaimed dimensions alone</mark>."
- `Σⱼ (Mⱼ ∘ R(F_N(sⱼ)))`: for each variable j, take its own source input's representation, keep only the part in Zⱼ's masked region, sum across variables → "in claimed dimensions, blend in the appropriate source's value."
- `R⁻¹(...)`: rotate the blended result back to the original space to continue the forward pass.

(example, d=6, 2 variables):

```
dim:  1    2    3    4    5    6
M1:  0.9  0.8  0.1  0.0  0.0  0.0
M2:  0.0  0.0  0.1  0.9  0.8  0.0
```

- Dim 1: `Σⱼ Mⱼ = 0.9` → new value = `0.1×base[1] + 0.9×source1[1]` — mostly replaced by source1 (since Z₁ "owns" this dim).
- Dim 6: `Σⱼ Mⱼ = 0` → new value = `base[6]` — left completely untouched (unclaimed by anyone).
- Dim 3: weakly claimed by both (`0.1` each) → mostly base, small nudge toward both sources — this in-between blending only happens mid-training, before boundaries sharpen.

Why source1 specifically for Z₁?
- An intervention on a causal variable means: "what if Z₁ had taken a _different_ value?" That different value isn't invented — it's **borrowed** from a real second example (the source), where Z₁ naturally took on a different value. E.g., base amount=1.50 (Z₁=False) vs. source1 amount=3.00 (Z�1=True) — intervening on Z₁ means importing the True value from source1. Since dims 1–2 are hypothesized to encode Z₁, their intervention values are pulled specifically from source1 (not source2, which is reserved for Z₂'s intervention in a multi-variable setup).

Equation 4 is a **differentiable relaxation of a hard swap** — a smooth blend that lets gradient descent simultaneously learn _both_ the rotation matrix (where a variable lives, direction-wise) _and_ the boundary width (how big its subspace is), while converging to a clean hard swap as training progresses (β → 0).

---

Refresher on Tokens, Token Positions, and Internal Activations:
Tokens
- Language models don't process raw text — they first split it into small pieces called **tokens** (whole words, sub-words, punctuation, or individual characters, depending on the tokenizer). E.g., `"1.50"` might tokenize into 4 separate tokens: `"1"`, `"."`, `"5"`, `"0"`.

Token position
- The whole prompt becomes an ordered sequence of tokens, each occupying a numbered **position** (its index/slot) in that sequence. E.g.:

```
Position: 1        2      3      ...    9    10   11   12
Token:   "Please" "say" "yes"   ...   "1"  "."  "5"  "0"
```

Internal activations
- At **every layer** of the transformer, the network computes a separate high-dimensional vector (the "hidden representation"/"activation") **for each token position**. For Alpaca: 32 layers × 4096-dim vector, per token position. So "the activation at position 9, layer 10" = a specific 4096-number vector — the network's evolving internal state for that token, at that depth.

Why start scanning from the first input digit?
- The causal variables being tested (Z₁: "amount > lower bound?") can only be computed **after** the model has seen the input amount. Before that token position, there's nothing yet to compare, so it wouldn't make sense to look for the variable there. Scanning starts from the first digit token and continues to the end of the prompt.

How this Connects to Figure 4's heat map:
- x-axis = token position (labeled with the actual token + its index, e.g. `'X' (70)`), y-axis = layer index. Each cell = "does the representation _at this token, at this layer_ encode the hypothesized variable?"

---
Alignment Process Details

**Quote from paper :"train Boundless DAS with test query token representations... 7 layers... control condition... single source, multiple variables."**

Which tokens/layers are probed
- **"Test query token representations, starting from the first input digit to the last token"**: scanning begins at the input amount's first digit (since the comparison can't be computed before the model sees the number) and continues to the prompt's end.
- **7 layers {0, 5, 10, 15, 20, 25, 30}**: out of Alpaca's 32 layers, they sample every 5th as checkpoints — cheaper than testing all 32, while still covering the full depth of the network.

Control condition
- **"The token before the first digit, where nothing should be expected to be aligned"** — a sanity/negative check. At that position, the model hasn't yet seen the query number, so it _cannot_ have computed the comparison. If Boundless DAS reported high IIA there anyway, that would signal the method is finding spurious/fake structure. (In the heatmaps, this column does show near-baseline scores, as expected.)

Single source, multiple variables
- **"Interchange with a single source"**: rather than using a separate source example per variable (source1 for Z₁, source2 for Z₂), they use just **one** source example per training example, and pull _both_ Z₁'s and Z₂'s counterfactual values from that same source.
- **"...while allowing multiple causal variables to be aligned across examples"**: across the full training set, both Z₁ and Z₂ still get learned/aligned — just not by drawing from two separate sources within a single training example. Likely done for implementation simplicity / computational cost.

---
Equal-Size Constraint on Variable Subspaces
**Q: "We enforce the interval between two boundary indices to be the same" — does this mean each high-level variable gets mapped to a neural subspace of the same size?**
- **Yes.** Normally each `bⱼ` is learned independently, so Z₁ could end up with a much bigger or smaller subspace than Z₂. This constraint forces `(b₁−b₀) = (b₂−b₁)` — every variable's segment must be the same width. The _shared_ width is still learned (gradient descent picks it), but no variable is allowed more "room" than another.

**Why do this instead of one source for each variable targeted?:**
1. Fewer free parameters → easier/more stable optimization.
2. Symmetric hypothesis: Z₁ and Z₂ are structurally analogous boolean comparisons (no a priori reason to expect asymmetric capacity).
3. Cleaner comparison across variables in results.

**Limitation:** explicitly a simplification — if the network really did devote unequal space to Z₁ vs Z₂, this constraint would hide that asymmetry, finding only the best equal-width compromise instead.

---
Task Restriction: Bracket Width [2.50, 7.50]

**Q: "We restrict the absolute difference between the lower bound and the upper bound to be [2.50, 7.50] due to model errors outside these values"
- Bracket width = upper bound − lower bound (how wide the "yes" range is). Alpaca's raw accuracy **drops** for brackets that are too narrow (hard to judge precisely) or too wide (too little "no" region, possible other biases). Since the whole method assumes reliable, explainable behavior (_"we need behavior to explain"_ — echoing the paper's earlier stated principle of only studying tasks with high model performance), they restrict experiments to the width range **[2.50, 7.50]**, where Alpaca performs reliably (their reported 85% task accuracy applies within this range). Outside it, errors are a separate, unexplained phenomenon that a clean causal model can't account for.
---
Baseline IIA and Label-Flip Distribution

**Q: "The latter two baseline IIAs are higher than chance because they are conditioned on the distribution of our output labels (i.e., how many times the original label gets to be flipped due to the intervention)" — what does this mean?**
Setup:
Baseline IIA = score obtained using a **random** (meaningless) rotation matrix instead of the learned one — a "chance" reference point.
- Left Boundary / Left and Right Boundary → baseline ≈ 0.50
- Mid-point Distance / Bracket Identity → baseline ≈ 0.60

Why is there a difference?
Not every intervention actually **flips** the output — sometimes swapping in the source's value for a variable leaves the causal model's predicted output unchanged from the base's original answer. ==A **random/broken** rotation matrix doesn't perform any real semantic swap — it effectively does nothing, so it "accidentally" gets credit whenever the _correct_ answer happened to be "no flip" (output stays the same).==
- For Left Boundary / Left and Right Boundary: interventions flip the label roughly 50/50 → random baseline ≈ 0.50.
- For Mid-point Distance / Bracket Identity: due to how these variables/interventions are structured, a larger share of trials happen to require "no flip" → a do-nothing random method gets these right more often by accident → inflated baseline ≈ 0.60.

You can't compare raw IIA scores across models directly — each has a different chance floor. This is why Figure 4's heat map coloring uses each model's own baseline as the lower bound and task performance as the upper bound, for fair comparison.

---
The "Bottom-Left / Upper-Right" Heatmap Pattern

Heatmap layout recap
Rows = layer index (30 at top, 0 at bottom, since the y-axis is printed descending); Columns = token position (early positions near input digits on the left, final token on the right).
- **Bottom-left** (low layers ~5–10, positions right after the digits start): high scores, ~0.7–0.9.
- **Upper-right** (high layers ~20–30, but _only_ the very last token column): high scores, ~0.86–0.89 — while the rest of that row (same layer, earlier positions) stays near baseline (~0.48–0.55).
- **Everywhere else**: near-baseline, ~0.48–0.55 — no real structure detected.

So visually: two "bright" patches in opposite corners, connected by a dim diagonal region in between — not a smooth gradient, but a **structured, localized** pattern.

The explanation?
1. **Bottom-left**: right after the model reads the input digits, it computes the comparison (Z₁/Z₂) and this shows up clearly at that token's own position, in the lower layers.
2. **Upper-right**: as depth increases, that already-computed value gets _copied forward_ into the **final token's**representation (since that's the position the model reads from to generate its answer) — visible only in later layers, only at the last token.
3. **Dim middle**: the variable isn't present at high layers _at the digit's own position_ (already handed off by then) nor at low layers _at the final token's position_ (hasn't been received yet).

Contrast with the inaccurate models?
- Mid-point Distance / Bracket Identity heatmaps don't show this same sharp two-corner structure — instead scores are spread diffusely with non-zero values everywhere, lacking the clean localized signature. This contrast (structured vs. diffuse) is used as extra evidence that the accurate models reflect a real, localized mechanism.

---
Table 1 Walkthrough
- **Columns:** Experiment | Task Acc. (Alpaca's raw behavioral accuracy) | IIAmax (best alignment score found) | Correlation (how closely this experiment's full IIA pattern matches the main/base experiment, marked ♣ = Left Boundary or ♥ = Left and Right Boundary).

Block 1 — the four main hypotheses
- All tested on the same inputs (Task Acc. = 0.85 throughout). Left Boundary (IIAmax 0.90) and Left and Right Boundary (0.86) are at/above task performance → accurate explanations. Mid-point Distance (0.70) and Bracket Identity (0.72) fall well short → inaccurate explanations. Correlation = 1.00 trivially (each is compared to itself as the reference).

Block 2 — correct vs. incorrect model outputs (Left and Right Boundary)
- **Correct Only**: Task Acc. defined as 1.00 (by construction). IIAmax rises to 0.88 (even cleaner than main result) — correlation 0.99.
- **Incorrect Only**: Task Acc. = 0.00 (by construction). IIAmax drops to 0.71 but stays well above the ~0.50 random baseline, correlation 0.84 — suggesting Alpaca uses the _same mechanism_ even when it gets the final answer wrong; it's narrowly missing, not using a different broken process.

Block 3 — robustness/generalization checks
- **New Bracket (Seen)**: retrained on one fixed bracket, Task Acc. 0.94, IIAmax 0.94 (matches, as expected — trained and tested on the same bracket).
- **New Bracket (Unseen)**: rotation matrix frozen, tested on a completely different bracket never seen in training — Task Acc. 0.95, IIAmax 0.95 — essentially no drop → generalizes to unseen brackets.
- **Irrelevant Contexts**: random GPT-4-generated prefix sentences added — Task Acc. drops to 0.84, IIAmax to 0.83 (only ~2–3% drop from main result), correlation 0.99 → alignment robust to distracting text.
- **Sibling Instructions**: tested on a logically equivalent but differently-worded instruction ("Say True/False" vs "Say yes/no") — Task Acc. 0.84, IIAmax 0.83, but correlation only 0.87 (some pattern shift).
- **+ exclude top right**: recomputing correlation while excluding the top-right heatmap region (high layers, final token — output-formatting zone) raises correlation to 0.92 — confirms the instruction-wording mismatch is localized specifically to the final output-fusion step, not the core boundary-check computation.

The causal explanation holds up across: correct/incorrect outputs, unseen price brackets, irrelevant distracting context, and different output vocabularies — with divergence only in the very last layers near the output token.

---
Boundary Learning Dynamics — Success vs. Failure Groups
Quote from paper : "We sample 100 experiment runs... success group where the boundary does not shrink to 0... failure group where the boundary does shrink to 0."

Each variable gets a learned boundary width (how many dimensions it's assigned).
- **Success group**: training settles on a meaningfully nonzero width → Boundless DAS found real dimensions/genuine alignment for that variable.
- **Failure group**: width shrinks all the way to 0 → the method effectively assigned no dimensions, meaning it found no real structure supporting that hypothesis at that spot.

(Related finding from Figure 6: <mark style="background: #ABF7F7A6;">in success cases, each aligned variable only needs ~5–10% of the representation space, and IIA stays stable despite the shrinking dimensionality — suggesting a small fraction of the space is sufficient to encode a causal variable.</mark>)

---
Section 4.7 — Metric Calibration
Addresses a reviewer concern: _is IIA actually trustworthy, or could it show fake structure?_ Two additional checks:

Check 1 — Random rotation matrix, position-specific
- At the key spot (layer 10, token position 75, "Left and Right Boundary" model), IIA drops from **0.83** (learned rotation) to **0.53** (random rotation). Other positions drop similarly. The two "control" causal models (Mid-point Distance, Bracket Identity) hit ~0.60 with a random rotation — consistent with the inflated baseline (label-flip distribution), not real structure.
- **Conclusion:** the high score found isn't a metric artifact — a meaningless rotation gets nowhere near it.

Check 2 — Boundless DAS on a randomly initialized (untrained) LLaMA-7B
- Task performance on the random model: **near 0%** (expected — random weights have no real language ability).
- Yet running Boundless DAS on its first layer still found alignments up to **69% IIA** for some token representations — comparable to a most-frequent-label dummy baseline (66%). It even found directions shifting probability toward unrelated tokens like "dog"/"give" instead of "yes"/"no" — pure noise being fit by a powerful optimizer, not real understanding.
- **Critically**, this spurious structure **collapsed to 0% IIA** under every robustness check (new brackets, different phrasing, etc.), and also collapsed to 0% on a different random seed's model.

Takeaway of 4.7:
1. Random-rotation comparison confirms the main finding is far above chance — real structure.
2. A powerful method like Boundless DAS _can_ occasionally "discover" fake structure in a meaningless network.
3. The distinguishing factor is **robustness**: real structure (Alpaca) survives all generalization checks; spurious structure (random model) fails all of them. This is exactly why the robustness checks in Section 4.5 / Table 1 matter — they're the real evidence, not the raw IIA number alone.