Atticus Geiger

---
The Causal Model Cell — Building the Symbolic Algorithm
- This cell constructs the **high-level causal model** (not a neural network) that solves hierarchical equality — the ground-truth algorithm DAS will later try to find inside a trained network.
```python
def randvec(n=50, lower=-1, upper=1):
    return np.array([round(random.uniform(lower, upper), 2) for i in range(n)])

embedding_dim = 2
number_of_entities = 20
variables = ["W", "X", "Y", "Z", "WX", "YZ", "O"]
reps = [randvec(embedding_dim, lower=-1, upper=1) for _ in range(number_of_entities)]
values = {variable: reps for variable in ["W", "X", "Y", "Z"]}
values["WX"] = [True, False]
values["YZ"] = [True, False]
values["O"] = [True, False]

parents = {
    "W": [], "X": [], "Y": [], "Z": [],
    "WX": ["W", "X"], "YZ": ["Y", "Z"], "O": ["WX", "YZ"],
}

functions = {
    "W": FILLER, "X": FILLER, "Y": FILLER, "Z": FILLER,
    "WX": lambda x, y: np.array_equal(x, y),
    "YZ": lambda x, y: np.array_equal(x, y),
    "O": lambda x, y: x == y,
}

equality_model = CausalModel(variables, values, parents, functions, pos=pos)
```
- **`randvec`**: generates a random n-dim vector, standing in for an arbitrary "entity" (same trick as word embeddings for discrete symbols).
- **`embedding_dim=2`**: each entity (W, X, Y, Z) is a 2D vector.
- **`reps`**: 20 candidate entity embeddings — W/X/Y/Z can each independently be any of these 20.
- **`variables`**: the 7 nodes in the causal graph — 4 raw inputs (W,X,Y,Z), 2 intermediates (WX, YZ), 1 output (O).
- **`parents`**: defines the DAG structure — WX depends on {W,X}, YZ depends on {Y,Z}, O depends on {WX,YZ}. W/X/Y/Z have no parents (roots).
- **`functions`**: the actual mechanism at each node — `WX = (W==X)` via `np.array_equal`, `YZ = (Y==Z)`, `O = (WX==YZ)`.
- **`FILLER()`**: placeholder for root variables — ==never really invoked, since root values are supplied directly as inputs when running the model.==
- **`equality_model`**: bundles all of this into a runnable `pyvene.CausalModel` object — supports `run_forward`, `run_interchange`, etc.

==**Connects to notes on Definitions 1–4 (DAS paper):** this is the symbolic model **B**/**H**, now instantiated as runnable code — same `V1=(w=x)`, `V2=(y=z)`, `V3=V1∧V2` structure.==

---
2. Handcrafted MLP — Why Build One At All

> "Before we train a network to solve the hierarchical equality task, first consider an analytical solution where we define a neural network to have weights that are handcrafted to solve the task."

**Purpose:** a validation/pedagogical step. Before trusting DAS to _discover_ an alignment inside an opaquely-trained network, first build a network where the correct alignment is _known by construction_ — giving a ground-truth check for DAS's later output.
```python
config = MLPConfig(
    h_dim=embedding_dim * 4,
    activation_function="relu",
    n_layer=2,
    num_classes=2,
    pdrop=0.0,
)
```

| Field                        | Meaning                                                                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `h_dim=embedding_dim*4=8`    | Hidden width. Input = 4 entities × 2 dims concatenated = 8-dim; hidden layers match this width.                                                                          |
| `activation_function="relu"` | Needed specifically because the handcrafted solution computes ==**absolute difference** via ReLU: `\|d\| = ReLU(d) + ReLU(-d)`.==                                        |
| `n_layer=2`                  | Two hidden layers, ==mirroring the algorithm's two-step tree structure (pairwise diffs → pair-vs-pair comparison)==. A third linear "score" layer produces final logits. |
| `num_classes=2`              | Binary output (True/False for O) → 2 logits.                                                                                                                             |
| `pdrop=0.0`                  | No dropout — weights are set by hand to exact deterministic values; dropout would corrupt this.                                                                          |

---
Layer 1 — Computing Absolute Differences
Formula : 
$ReLU(W_1[a,b,c,d])=[max⁡(a−b,0), max⁡(b−a,0), max⁡(c−d,0), max⁡(d−c,0)]$
- where `a=W, b=X, c=Y, d=Z` (each 2D vectors), input = `[a,b,c,d]` concatenated (8-dim).

The actual W1 matrix:
```
Row 1: [ 1,  0, -1,  0,  0,  0,  0,  0]   → a1 − b1
Row 2: [ 0,  1,  0, -1,  0,  0,  0,  0]   → a2 − b2
Row 3: [-1,  0,  1,  0,  0,  0,  0,  0]   → b1 − a1
Row 4: [ 0, -1,  0,  1,  0,  0,  0,  0]   → b2 − a2
Row 5: [ 0,  0,  0,  0,  1,  0, -1,  0]   → c1 − d1
Row 6: [ 0,  0,  0,  0,  0,  1,  0, -1]   → c2 − d2
Row 7: [ 0,  0,  0,  0, -1,  0,  1,  0]   → d1 − c1
Row 8: [ 0,  0,  0,  0,  0, -1,  0,  1]   → d2 − c2
```

Each row isolates one coordinate-wise difference (or its negation); the zeros ensure no mixing between unrelated coordinates.

For any two numbers, exactly one of `(a−b)` or `(b−a)` is positive (both are 0 if equal). After ReLU:
- If `a>b`: `ReLU(a−b)=a−b` survives, `ReLU(b−a)=0`.
- If `b>a`: roles flip.
- If `a=b`: both are exactly 0.

So `ReLU(a−b) + ReLU(b−a) = |a−b|` — a sum where at most one term is nonzero. ==**Layer 1 only computes the raw ingredients; layer 2 does the actual summing.**==

Why this maps cleanly onto WX/YZ

Rows 1–4 only touch a,b (→ WX comparison); rows 5–8 only touch c,d (→ YZ comparison) — zero mixing between halves. This clean separation is exactly why the first 4 neurons later get aligned with `WX` and the last 4 with `YZ` — a **known, ground-truth localist alignment**, built deliberately rather than discovered.

---
Layer 2 — Pairwise Distance Comparison

Formula:
$ReLU(W_2 ReLU(W_1[a,b,c,d]))=[ ∣a−b∣−∣c−d∣, ∣c−d∣−∣a−b∣, ∣a−b∣, ∣c−d∣, 0,0,0,0 ]$

`W2` gets `.transpose(0,1)` before loading — ==so **columns** of the printed matrix become each output neuron's weights.==

Naming layer 1's output slots: `x0,x1 = ReLU(a−b)` per coord; `x2,x3 = ReLU(b−a)`; `x4,x5 = ReLU(c−d)`; `x6,x7 = ReLU(d−c)`.

- **Column 0** → `y0 = (x0+x1+x2+x3) − (x4+x5+x6+x7) = |a−b| − |c−d|`
- **Column 1** → `y1 = |c−d| − |a−b|` (exact sign-flip of column 0 — same mirror trick, one level up)
- **Column 2** → `y2 = x4+x5+x6+x7 = |c−d|`
- **Column 3** → `y3 = x0+x1+x2+x3 = |a−b|`
- **Columns 4–7** → all zero (<mark style="background: #FF5582A6;">unused padding to match layer width</mark>)

Small note:
The notebook's formula states slot 2 = `|a−b|` and slot 3 = `|c−d|`. **The literal matrix computes the opposite**: slot 2 = `|c−d|`, slot 3 = `|a−b|`. Verified by direct derivation above.
- **Why it doesn't matter:** Layer 3's weights (`W3`) apply the **identical coefficient** (`−0.999999`) to both slot 2 and slot 3 — so only their _sum_ (`|a−b|+|c−d|`) matters for the final output, not which slot holds which value. The notebook's formula is pedagogically loose but the network's correctness is unaffected.

After the outer ReLU
- `y2` and `y3` are already non-negative (absolute values), untouched by ReLU. `y0`/`y1` are the mirror pair — ==ReLU zeroes out whichever is negative.==
**Final layer-2 output:** `[ReLU(|a−b|−|c−d|), ReLU(|c−d|−|a−b|), |c−d|, |a−b|, 0,0,0,0]`

---
Layer 3 — Thresholding into a Binary Decision

Formula:
$W_3 ReLU(W_2ReLU(W_1[a,b,c,d]))=[ ∣∣a−b∣−∣c−d∣∣−0.999999∣a−b∣−0.999999∣c−d∣, 0 ]$

The matrix (also transposed):
```python
W3 = [[1,0],[1,0],[-0.999999,0],[-0.999999,0],[0,0],[0,0],[0,0],[0,0]]
bias = [0, 0.00000000000001]
```

After transpose: **Output neuron 0** weights = `[1, 1, −0.999999, −0.999999, 0,0,0,0]`; **Output neuron 1** weights = all zero.

Deriving the logits:
- Let `d1=|a−b|`, `d2=|c−d|`, and `h0,h1` = layer 2's mirror pair (exactly one nonzero, equal to `|d1−d2|`).
```
logit0 = (h0+h1) − 0.999999·(h2+h3) = |d1−d2| − 0.999999·(d1+d2)
logit1 = 0.00000000000001   (fixed near-zero constant)
```

Verifying with cases:

|Case|d1, d2|logit0|Winner|Correct?|
|---|---|---|---|---|
|Both pairs equal|0, 0|`0`|logit1 (True)|✅ WX=YZ=True → O=True|
|One pair equal, other not|0, 5|`≈0.000005`|logit0 (False)|✅ WX≠YZ → O=False|
|Both pairs unequal, similar magnitude|3, 7|`≈−6`|logit1 (True)|✅ WX=YZ=False → O=True|

Why the `0.999999` coefficient?
- ==Since `|d1−d2| ≤ d1+d2` always (triangle-inequality-like fact), multiplying the sum by slightly-less-than-1 makes `logit0`negative/zero whenever the two distances are comparable (including both zero) — but leaves a small positive residue the instant one distance is exactly 0 and the other isn't, just enough to clear the tiny `logit1` bias.== This single scalar comparison implements the boolean `O = (WX==YZ)` purely via magnitude arithmetic, no explicit branching.
---
Validating the Handcrafted Network — Dataset Generation + Evaluation
```python
n_examples = 100000
examples = equality_model.generate_factual_dataset(
    n_examples, equality_model.sample_input_tree_balanced
)
X = torch.stack([example['input_ids'] for example in examples])
y = torch.stack([example['labels'] for example in examples])

preds = handcrafted.forward(inputs_embeds=X)
print(classification_report(y, preds[0].argmax(1)))
```
Cell 1 — generating the dataset from the causal model
- **`generate_factual_dataset`**: uses the `CausalModel` to generate 100,000 plain forward-pass examples (no interventions) — sample random W,X,Y,Z, run through the model's functions, record input + correct output.
- **`sample_input_tree_balanced`**: the sampling strategy — "balanced" implies roughly even True/False coverage, avoiding a lopsided dataset that would make accuracy look artificially good via majority-class guessing.
- **`X`**: stacked `input_ids` — the 100,000 concatenated `[W,X,Y,Z]` vectors, shape ~`(100000, 8)`.
- **`y`**: stacked `labels` — the 100,000 correct `O` values.

Cell 2 — running the network and scoring it
- **`handcrafted.forward(inputs_embeds=X)`**: feeds all 100,000 inputs through the traced-through 3-layer network. ==`inputs_embeds=` skips any embedding-lookup step since inputs are already raw vectors.==
- **`preds[0]`**: the two raw logits per example, shape `(100000, 2)` — column 0 = False-score, column 1 = True-score.
- **`.argmax(1)`**: ==converts logit pairs into hard predictions==.
- **`classification_report(y, preds[0].argmax(1))`**: precision/recall/F1/accuracy, network's predictions vs. ground truth.

`.argmax(1)` (example)

For 3 examples:
```
             False-score   True-score
example 1:      -5.2           8.1
example 2:       3.0          -1.4
example 3:       0.01          0.02
```

- "Argmax" = ==the **index** of the maximum value==, not the value itself (`max([-5.2,8.1])=8.1`; `argmax([-5.2,8.1])=1`, the position of 8.1).
- Dimension 0 = "which example" axis (length 100,000); dimension 1 = "which class" axis (length 2).
- `.argmax(1)` scans across dimension 1 (the two class columns) **independently for every row**, returning one predicted class index (0 or 1) per example → result `[1, 0, 1]` for the 3-example toy case above.
- `.argmax(0)` would instead scan down each column across all examples — a different, here-meaningless question (which example had the single highest score for a given class).

Why does this cell matter?
- Empirical proof the handcrafted weights (traced through Sections 3–5 above) genuinely implement hierarchical equality correctly — not just on a few hand-picked examples, but across 100,000 randomly sampled inputs. A perfect classification report validates the by-hand derivation; this is the checkpoint before moving on to causal abstraction analysis (aligning known/hypothesized subspaces and later running DAS to see if it can rediscover this same structure without being told it in advance).
---
Generating the Counterfactual Training Dataset
```python
data_size = 2048
batch_size = 16
dataset = equality_model.generate_counterfactual_dataset(
    data_size,
    intervention_id,
    batch_size,
    device="cpu",
    sampler=equality_model.sample_input_tree_balanced,
)
```

This is the code-level implementation of the training-data recipe worked through earlier in the DAS-paper (Appendix A.1, "Building the Training Dataset for R"): _base input + source input(s) + a ground-truth counterfactual label, computed via the causal model's own interchange-intervention mechanism._
- **`data_size=2048`**: total number of counterfactual training examples generated.
- **`batch_size=16`**: how many examples get grouped per training step later — doesn't affect generation itself, only downstream consumption.
- **`generate_counterfactual_dataset(...)`**: for each of the 2048 examples — samples a base input, samples source input(s), applies an intervention (per `intervention_id`), runs `equality_model`'s mechanism under that intervention (i.e. `run_interchange`, as demonstrated earlier in cell 19) to compute the correct counterfactual label, and packages base + source(s) + label together.
- **`intervention_id`**: ==controls _which_ variable(s) get intervened on for each example== — ==mirrors the single-vs-double intervention mixing rule from the appendix notes (sometimes just `WX`, sometimes just `YZ`, sometimes both with separate dedicated sources)==.
- **`device="cpu"`**: "fixes CUDA error on macbook*" — confirms this call needs to stay `"cpu"` (or `"mps"`, if supported) rather than defaulting to `"cuda".
- **`sampler=equality_model.sample_input_tree_balanced`**: reuses the same balanced sampler, now applied to both base _and_ source inputs, keeping even True/False coverage.

**Output:** `dataset` — 2048 structured counterfactual examples ready to feed into the actual DAS training loop (the rotation-matrix `R` training via gradient descent + cross-entropy, per Definition 4).

---
Reading One Dataset Entry (`dataset[0]`)
Output:
```python
{'labels': tensor([1.]), 'base_labels': tensor([0.]),
 'input_ids': tensor([0.47, 0.35, 0.47, 0.35, 0.78, -0.83, -0.56, 0.18]),
 'source_input_ids': tensor([[-0.16, -0.94, 0.66, 0.24, 0.07, 0.95, 0.07, 0.95],
                              [0., 0., 0., 0., 0., 0., 0., 0.]]),
 'intervention_id': tensor([0])}
```

- **`input_ids`** → base `[W,X,Y,Z]`. Here `W=X=[0.47,0.35]` (so `WX=True`) and `Y≠Z` (so `YZ=False`).
- **`base_labels`** → the _un-intervened_ factual output: `O=(WX==YZ)=(True==False)=False=0`. 
- **`source_input_ids`** → shape `(2,8)`, <mark style="background: #FF5582A6;">always 2 slots (fixed shape for batching), even though most examples only need 1 or 2 depending on how many variables are intervened on.</mark> Row 0 here is the real source (`W≠X` → source-`WX=False`); **row 1 is all-zero padding**, unused.
- **`intervention_id=0`** → says _which_ variable is targeted — index 0 → `WX`. Tells the training code "overwrite the base's `WX` with the source's `WX`; leave `YZ` alone."
- **`labels`** → the counterfactual target: force `WX=False` (from source) onto the base, keep `YZ=False` (base's own) → `O=(False==False)=True=1`. 

This is one (base + source + counterfactual label) triple — exactly the training-example structure from Appendix A.1 notes, now visible as real tensors. The base's factual answer flips (0→1) purely because `WX` was swapped in from the source — this mismatch (base's `R`-predicted counterfactual vs. this `labels` target) is what cross-entropy trains `R` against.

---
`"sources->base"` Syntax — Position Mapping in pyvene Interventions

The full structure, fully expanded (batch_size=2 example):
```python
{
    "sources->base": (
        # Element 0: SOURCE read-positions
        [
            [[0], [0]],   # WX's source-read-position, per example
            [[0], [0]],   # YZ's source-read-position, per example
        ],
        # Element 1: BASE write-positions
        [
            [[0], [0]],   # WX's base-write-position, per example
            [[0], [0]],   # YZ's base-write-position, per example
        ],
    )
}
```

The 5 nesting levels, top to bottom:

| Level | Type | Distinguishes |
|---|---|---|
| 1 | Tuple (2 elements) | source-side positions vs. base-side positions |
| 2 | List (2 elements) | which intervention target (WX vs. YZ) |
| 3 | List (`batch_size` elements) | which example in the batch |
| 4 | List (1+ elements) | which position(s) *within* that example |
| 5 | Number | the literal position index |

- `[[0]] * batch_size` uses Python's list-repetition operator to duplicate `[0]` once per example — e.g. `batch_size=2` → `[[0], [0]]`.
- The innermost `[0]` is a list (not a bare number) because pyvene's format supports intervening at *multiple* positions per example simultaneously (e.g. `[3,5,7]`) — this task just never needs more than one.

Why everything is hardcoded to position `0`?
**Key distinction: this task has no real token sequence.**
- **Real sequence case (Alpaca/Boundless DAS paper):** input is an actual sentence, tokenized into many positions (`"Please"`, `"say"`, ..., `"1"`, `"."`, `"5"`, `"0"`, ...) — dozens of token positions, each with its own hidden representation at every layer. ==*Which* position to intervene at was a genuine, meaningful research question== (the paper's heatmaps scan across real positions like `'X' (70)`, `'.' (71)`).
- **This task (hierarchical equality):** each input is just `W,X,Y,Z` concatenated into **one flat vector** (e.g. 16 numbers) — no words, no tokens, no sentence at all. Fed straight in as `inputs_embeds`.

Since `pyvene`'s API (`IntervenableModel`, `RepresentationConfig`) is built generally to support real language models with real sequences, it always expects a "position" to be specified — even when the task doesn't have multiple positions to choose from. To satisfy this, an earlier reshaping step (`.unsqueeze(1)`) artificially inserts a fake "sequence" dimension of length exactly 1, so the single input vector is treated as "a sequence with only one position," always labeled index `0`.

**Bottom line:** every position value is `[0]` not because the code is lazy or omitting something, but because ==`0` is the *only* valid position that exists in this task's data==. The elaborate nested-bracket structure is pyvene's general-purpose position-specification machinery, faithfully filled in — it just happens this particular non-sequence task only ever has one possible answer to give it.

---
Gradient Accumulation Block — Training Loop Syntax
The code:
```python
if gradient_accumulation_steps > 1:
    loss = loss / gradient_accumulation_steps
loss.backward()
if total_step % gradient_accumulation_steps == 0:
    optimizer.step()
    intervenable.set_zero_grad()
total_step += 1
```

What problem this solves?
**Gradient accumulation**: simulates a larger batch size than fits in memory, by summing gradients across several small batches *before* applying a weight update — mathematically similar to training on one big batch, without the memory cost.

In this notebook, `gradient_accumulation_steps = 1`, so this entire mechanism is functionally inert — it behaves exactly like ordinary step-by-step training (back propagation+ update every single batch). The code is written generally so it *would* support real accumulation if that value were increased later.

Line by line
- **`if gradient_accumulation_steps > 1: loss = loss / N`** — scales the loss down before backprop, so gradients accumulated across N batches end up roughly the same magnitude as one N-times-larger batch's gradient would. With N=1, this never runs.
- **`loss.backward()`** — computes gradients and **adds** them into each trainable parameter's `.grad` (PyTorch accumulates by default, doesn't overwrite) — this accumulation-by-default is exactly what makes calling `.backward()` repeatedly without resetting in between work as intended.
- **`total_step % gradient_accumulation_steps == 0`** — the `%` (modulo) operator gives the remainder of `total_step ÷ N`. This is `True` only every Nth step (e.g. N=4 → True at steps 0,4,8,...). With N=1, `x % 1` is always 0, so this is **always True** — update happens every batch.
- **`optimizer.step()`** — applies the accumulated gradient(s) to actually update the weights (here, just the rotation matrix R).
- **`intervenable.set_zero_grad()`** — resets `.grad` back to zero so the next group of batches starts accumulating fresh.
- **`total_step += 1`** — increments the global step counter every single batch, regardless of the above; this is what the modulo check compares against on the next iteration.

With `gradient_accumulation_steps = 1`, every line still runs, but the conditionals always evaluate to "do it now" — so in practice this reduces to plain, ordinary training: backprop → step → zero-grad, every batch. The extra scaffolding is there so the loop *would* correctly support larger accumulation values without needing any other changes.

---
