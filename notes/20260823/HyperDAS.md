ICLR 2025; Jiuding Sun , Jing Huang

---
HyperDAS's 4 Core steps
1. Encode a language description of a concept using a transformer that attends to the target LM processing a base prompt and a counterfactual prompt.
2. Pair tokens in the base and counterfactual prompts using <mark style="background: #FF5582A6;">attention</mark>, with the encoding from (1) as query and token-pairs as keys/values.
3. Select features of the residual stream via a fixed orthogonal matrix that undergoes a Householder transformation, using the encoding from (1).
4. Patch the selected residual stream features of aligned base tokens to the values they take on in aligned counterfactual tokens. (Actual interchange intervention step)

Example from paper: concept = "the country a city is in," base = "Vienna is in," counterfactual = "I love Paris" → target model should now predict "France" instead of "Austria."

Step 1 — Concept encoding
- The hyper-network = a transformer processing the concept description's tokens.
- ==At every layer: standard **self-attention** among the concept-description's own tokens, PLUS two extra **cross-attention** blocks — one attending to the target LM's residual stream while processing the base prompt, one while processing the counterfactual prompt.==
- Query in these cross-attention steps comes from the hyper-network's tokens; keys/values come from the target model's hidden states.
- Result after all `N` layers: the **last token's** final vector, `v` (= `e_E^(N)` in the paper's notation) — <mark style="background: #ABF7F7A6;">a rich summary encoding "this concept, specifically in the context of this base/counterfactual pair."</mark>

Step 2 — Token pairing
- **Also cross-attention**, but with a specific role assignment: `v` (from step 1) = the **query**; every possible base-counterfactual token pair = the **keys/values**.
- Dot product between `v` and each candidate pairing → raw scores → **softmax** normalizes into an attention pattern (paper's matrix `G`).
- Result: a soft, learned answer to =="which specific base token and which specific counterfactual token actually carry this concept's information"== — e.g., "nna" (last token of Vienna) paired with "Paris."

Step 3 — Subspace selection (Householder transformation)
- Same underlying goal as DAS's rotation matrix `R` (learned in the tutorial): find a subspace of directions in the residual stream that captures a specific concept.
- **HyperDAS's twist:** instead of training a separate `R` per concept, **generate** an appropriate `R` in one forward pass from `v`.
- Mechanism: start with a fixed generic orthogonal matrix `R'`. Build a **Householder reflection** `H = I − 2vv^T/(v^Tv)` from `v` — "reflect across the plane perpendicular to `v`." Compute `R = R'H`.
- `R` is guaranteed to remain a valid orthogonal matrix (property of combining an orthogonal matrix with a Householder reflection), ==now steered toward the specific concept `v` represents — no separate training run needed per concept.==

Step 4 — Patching (the intervention)
- An Identical mechanism to `RotatedSpaceIntervention.forward()`from the DAS coding tutorial I worked on earlier: rotate the base's hidden state (at the step-2-selected token) into the `R`-coordinate system, rotate the counterfactual's hidden state (at its own selected token) into the same system, swap the relevant subspace, rotate back.
- Patched result replaces the base's hidden state at that position; target model continues its forward pass from there.
- **Worked example outcome:** "nna"'s country-subspace gets replaced with "Paris"'s value → model now predicts "France" instead of "Austria."

What's new here vs. already understood from Vanilla DAS :

| Step                                  | Status                                                                                                    |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 4 (Patching)                          | Zero new material — same as what i saw in the DAS tutorial.                                               |
| 3 (Subspace selection)                | One new trick (Householder reflection) layered on the orthogonal rotation matrix concept.                 |
| 1–2 (Concept encoding, token pairing) | Completely new: automates two choices (which subspace, which token positions) that DAS hardcodes by hand. |

Connections to my prior knowledge:
- Step 1–2's cross-attention = the exact mechanism from the "cat sat mat" example — query from one source, keys/values from another, dot-product → softmax → weighted combination.
- `v` = last-token summary vector, same convention as the **un-embedding** step (GPT predicting next-token from the *last* vector in the sequence).
- Step 3's Householder trick connects to the **superposition** concept (near-perpendicular directions encoding many concepts in high-dim space) — HyperDAS's subspace-generation approach is essentially saying it can disentangle a target concept's direction even from a superposed representation, via a learned reflection.
- Step 4 = same code in DAS intro tutorial I worked on (`RotatedSpaceIntervention`), applied at automatically-selected rather than hardcoded positions/subspaces.

#### The differenced between Boundless DAS and HyperDAS

The two brute-force problems DAS has:
1. **Subspace size** — how many dimensions encode a given variable (originally hand-searched by trying different k values).
2. **Token position** — which token position(s) to intervene at (this is trivial in a single-vector task like the DAS tutorial's hierarchical equality; but a real brute-force search in a real LLM sentence with many positions).

Boundless DAS solves problem #1 only
- Learnable sigmoid **boundary mask** (`b_j`) — subspace size learned automatically via gradient descent, instead of manually trying different sizes.
- ==**Still requires manually picking the token position(s)** to intervene at.==
- ==**Still task/variable-specific** — `R` is trained from scratch per fixed causal model/variable set (exactly as in the DAS tutorial: one `R`, trained once, for `equality_model`'s WX/YZ).==
- Applied (in the papers read) to: toy tasks (hierarchical equality) and Alpaca-7B on one fixed task.

HyperDAS solves problem #2, and removes the need to retrain per concept entirely
- **Token position selection is learned automatically** via attention-based pairing (step 2, matrix `G`) — something Boundless DAS never addresses.
- **No retraining needed per concept.** One hyper-network, trained once, generates an appropriate `R` **on the fly**, in a single forward pass, for *any* concept description in English — via the Householder transformation (`R = R'H`, built from the concept-summary vector `v`).
- Applied to: Llama3-8B across the RAVEL benchmark (5 entity domains, dozens of distinct attributes) — designed for answering many different ad hoc concept questions without retraining for each.

Side-by-side comparison of both:

|                           | Boundless DAS                         | HyperDAS                                                                    |
| ------------------------- | ------------------------------------- | --------------------------------------------------------------------------- |
| Subspace size             | Learned automatically (boundary mask) | Fixed hyperparameter (dimension of subspace), not the paper's focus         |
| Token position            | Manually chosen                       | Learned automatically (attention pairing, matrix G)                         |
| Reusable across concepts? | No — retrain `R` per task/variable    | Yes — one hypernetwork, `R` generated fresh per concept                     |
| What gets trained         | Just `R` (target model frozen)        | A full hypernetwork (cross-attention transformer + MLP + Householder logic) |
| Compute cost              | Relatively cheap                      | Notably more expensive (~2.4x MDAS's compute)                               |

In summary:
- **Boundless DAS** = a smarter version of DAS **for one fixed task** — same setup (fixed variables, hand-picked position), just automates subspace sizing.
- **HyperDAS** = a fundamentally more **general tool** — built to handle arbitrary, ad hoc concepts without retraining, which requires solving the position problem too, at the cost of a much heavier architecture.
- They solve **different scopes** of the same underlying "automate DAS's manual choices" problem.
- <mark style="background: #ABF7F7A6;">I think that technically, the two ideas could be combined (HyperDAS's position-automation + cross-concept generalization, plus Boundless DAS's learned-boundary-mask for automatic subspace sizing instead of a fixed dimension).</mark>

RAVEL Benchmark — Cause vs. Iso Labeling Logic

The example
- **Base prompt:** "Albert Einstein studied the field" (asking about field of study).
- **Counterfactual prompt:** "Poland declared 2011 the Year of Marie Curie" (same entity type — Nobel laureate — different person).
- **Targeted attribute:** varies — either "birth year" or "field of study."

Case 1 — target = birth year → correct label = physics (unchanged)
- ==Curie's *birth year* info is patched into Einstein's representation; the *field-of-study* subspace is deliberately left untouched.==
- ==Since nothing relevant to field-of-study was intervened on, the model's field-of-study output should stay exactly as it was: **physics**.==
- Tests the **Iso score**: "proportion of interchange interventions that successfully do NOT change an attribute that was not targeted."

Case 2 — target = field of study → correct label = chemistry (changed)
- ==Curie's *field of study* info (chemistry) is patched into Einstein's representation.==
- ==A correctly-working intervention should flip the model's output from physics to **chemistry**.==
- Tests the **Cause score**: "proportion of interchange interventions that successfully change the attribute that WAS targeted."

RAVEL tests **disentanglement**: <mark style="background: #FF5582A6;">can a method (HyperDAS, MDAS, etc.) surgically touch only the targeted attribute's subspace, without bleeding into other attributes of the same entity (birth year, field of study, country, gender, award year — all coexisting in the same underlying representation)?</mark> Same base/counterfactual pair, but the correct label flips entirely depending on which attribute was targeted — this is exactly the test case that catches entanglement if it exists.

> [!note]
> This is the same underlying principle as the DAS tutorial's `input_sampler` deliberately randomizing the un-targeted variable (WX vs. YZ) — ==preventing a spuriously entangled subspace where touching one variable accidentally affects the other.== RAVEL's Cause/Iso distinction is that same idea, applied at LLM scale with real-world named attributes instead of abstract booleans.

HyperDAS Equation 1 — Distributed Interchange Intervention Formula

**Formula:** H ← h̄ + Rᵀ(R(ĥ) − R(h̄))
- **h̄** — hidden representation on the **base** prompt.
- **ĥ** — hidden representation on the **counterfactual** prompt.
- **R** — rotation matrix (low-rank) into the target subspace's coordinate system — same role as `R` in the DAS tutorial.
- **R(·)** — rotate a vector into that coordinate system.
- **Rᵀ** — transpose of R; for an orthogonal matrix, Rᵀ = R⁻¹, so <mark style="background: #FF5582A6;">this is exactly the "un-rotate" step.</mark>
- **←** — assignment/intervention ("H gets updated to..."), not an equation to solve.

Reading it step by step
1. **R(ĥ)** — rotate counterfactual's hidden state into R's coordinate system.
2. **R(h̄)** — rotate base's hidden state into the same system.
3. **R(ĥ) − R(h̄)** — the difference between them, measured only along the directions R cares about.
4. **Rᵀ(...)** — rotate that difference back into the original (unrotated) space.
5. **h̄ + (...)** — add the resulting "difference vector" onto the base's original hidden state.

How does this equal a subspace swap??
- This is mathematically equivalent to "rotate → replace the target subspace with the counterfactual's value → rotate back," just written as one combined subtract-and-add formula instead of three explicit steps. ==Because the subtraction/addition only ever touches components R projects onto, every direction **outside** R's subspace is left completely untouched.==

Direct mapping to the DAS tutorial's own code
`RotatedSpaceIntervention.forward()`:
```python
def forward(self, base, source, subspaces=None, **kwargs):
    rotated_base = self.rotate_layer(base)      # = R(h̄)
    rotated_source = self.rotate_layer(source)  # = R(ĥ)
    # interchange
    ...
```
- <mark style="background: #BBFABBA6;">Same underlying operation as Equation 1 — the tutorial's code implements it explicitly via masking/swapping in the rotated space then unrotating; the paper's formula achieves the identical result via subtract-then-add</mark>. 
- Different notation, same math.

Note specific to HyperDAS/RAVEL:
- R here is explicitly **low-rank** (e.g., dimension 128 out of 4096) — ==this is what makes the intervention a **partial**, targeted swap of one specific concept's subspace (e.g., "country" or "birth year"), rather than replacing the entire hidden state.==

---
### Formulas 3–5 — Cross-Attention Decoder Layer Mechanics

Shape setup:
- **h̄ ∈ R^(B×L×d)** — base prompt's hidden states: B tokens × L layers × d hidden dim, all stacked.
- **ĥ ∈ R^(C×L×d)** — same structure for the counterfactual prompt (C tokens instead of B).
- ==Not just "one hidden state per token" — **every token's hidden state, at every layer** of the target model M, stacked into one 3D array.==

<mark style="background: #FF5582A6;">Why do we store every layer?</mark>
> "Each layer of the hyper-network H can attend to every layer of the target model."
- <mark style="background: #ABF7F7A6;">Different concepts may live at different depths</mark> (recall BoundlessDAS : info appeared at specific layers, got copied forward). <mark style="background: #BBFABBA6;">Storing all L layers lets the hyper-network's own layers freely choose — via attention — which target-model layer(s) matter, rather than a human hardcoding one fixed layer.</mark>

Two cross-attention modules
- **$M\bar{H}A$** — attends to $\bar{h}$ (base prompt's hidden states).
- **$M\hat{H}A$** — attends to $\hat{h}$ (counterfactual prompt's hidden states).
- Same Q/K/V, dot-product, softmax mechanism — just wired to two different data sources.

#### The three formulas, <mark style="background: #FFB86CA6;">per hyper-network layer p
</mark>
**Formula 3 — self-attention (nothing new):**
$$e'_p = \text{MHA}(Q(e_p), K(e_p), V(e_p))$$
Ordinary self-attention — concept-description tokens (e_p) attend to each other. Same mechanism as the "cat sat mat" worked example.

**Formula 4 — cross-attention to base prompt:**
$$e''_p = \bar{M}HA(\bar{Q}(e'_p), \bar{K}(\bar{h}), \bar{V}(\bar{h}))$$
Query from e'_p (formula 3's output); key/value from h̄ (base prompt's full hidden-state stack, all tokens/layers). Asks: <mark style="background: #FF5582A6;">"given the concept so far, what in the base prompt's representations is relevant?"</mark>

**Formula 5 — cross-attention to counterfactual prompt:**
$$e_{p+1} = \hat{M}HA(\hat{Q}(e''_p), \hat{K}(\hat{h}), \hat{V}(\hat{h}))$$
Same pattern, attending to ĥ instead. Output e_{p+1} becomes the **input to the next hyper-network layer**.

#### The per-layer chain
- Within one layer p: self-attend among concept tokens (3) → cross-attend to base prompt (4) → cross-attend to counterfactual prompt (5) → hand off e_{p+1} to layer p+1.
- ==After N layers, the last token's vector = v = e_E^(N), the concept-summary vector used in steps 2 (token pairing) and 3 (Householder subspace selection).==

#### Section 3.2 — Dynamically Selecting Token-Positions

The goal:
- Automates what the DAS tutorial hardcoded by hand (`"sources->base": [[0]]*batch_size` — always position 0 to position 0). <mark style="background: #BBFABBA6;">Here, with real sentences having many positions, an attention mechanism learns which base token to intervene on and which counterfactual token to pull the value from.</mark>

The matrix G
- **G ∈ R^(B×(C+1))** — B = base tokens, C = counterfactual tokens.
- G_(b,c) = how strongly base token b should be replaced using counterfactual token c's value (0 to 1).
- **Extra "+1" column** = G_(b, C+1) = ==how strongly base token b should be **left alone** (no intervention).==
- Each row = a soft decision spread across all candidates: leave alone, or swap in from one specific counterfactual token.
- Example (Fig 1): "nna" (Vienna) gets score 1 for "Paris," 0 elsewhere → full replacement.

Building raw candidates — Eq 6 & 7:

**Eq 6 (real pairings):** `g_(b,c) = F([h̄_b^(l); ĥ_c^(l)])`
- <mark style="background: #ABF7F7A6;">Concatenate base token b's hidden state and counterfactual token c's hidden state (doubles dim, d→2d).</mark>
- <mark style="background: #ABF7F7A6;">F(·): linear projection squeezing back down to d — blends the two tokens' info into one combined representation.</mark>

**Eq 7 (the "leave alone" option):** `g_(b,C+1) = h̄_b^(l)`
- Just the base token's own hidden state, untouched — represents "do nothing."

Result: for every base token, a full row of candidate vectors (one per counterfactual token + one "leave alone" option).

Scoring the candidates — Eq 8 (= ordinary attention)
$$G'_{(b,c)} = \sum_{i=1}^{M} \frac{(e_E^{(N)})Q^{(i)} \cdot (g)K^{(i)}}{M\sqrt{d}}$$

This is exactly the attention mechanism already understood, with paper-specific names:
- `e_E^(N)` = v (concept-summary vector from 3.1) = the **query**.
- `g` (candidate vectors from Eq 6/7) = the **key**.
- `Q^(i)`, `K^(i)` = learned projections, same role as W_Q/W_K.
- Dot product = the same alignment-scoring step as the "cat sat mat" example.
- `√d` division = same numerical-stability scaling as standard attention.
- Sum over `i=1...M` = ==**multi-headed** attention (M heads run in parallel, summed)==.

**Plain meaning:** "how relevant is this specific base/counterfactual pairing (or 'leave alone') to the concept v?"

Normalizing — G = ColumnSoftmax(G')
- Softmax applied **per column** (per counterfactual token), not per row.
- Why: ensures each individual counterfactual token's total influence, spread across all base tokens, sums to 1 — preventing one counterfactual token from being duplicated/injected into many base tokens uncontrolled.

Constructing the intervention — Eq 9
$$\tilde{h}_b^{(l)} = G_{(b,C+1)}\bar{h}_b^{(l)} + \sum_{c=1}^{C} G_{(b,c)}\hat{h}_c^{(l)}$$

Same "weighted sum of values" step as ordinary attention (recall: creature's new value = 30.7%×cat + 18.6%×sat + 50.7%×mat).
- `G_(b,C+1)·h̄_b^(l)` = keep this fraction of the base's own value.
- `Σ_c G_(b,c)·ĥ_c^(l)` = blend in this fraction of each counterfactual token, weighted by relevance.

**Extreme cases:**
- G_(b,C+1)=1 (rest 0) → h̃_b = h̄_b → **no intervention**.
- G_(b,c)=1 for one c (rest 0) → h̃_b = ĥ_c → **full replacement** by that token.
- Between these (during training) → genuine soft blend; snapped to hard 0/1 at test time via double-argmax (Section 3.6).

This is directly the learned, general-purpose version of the DAS tutorial's hardcoded `"sources->base"` position mapping — same underlying decision ("which position maps to which"), just computed automatically via attention instead of specified by hand, and usable on real sentences with many candidate positions instead of a single trivial position.

---
Section 3.3 — Dynamically Identifying a Linear Subspace (Householder Transformation)

The goal:
- Section 3.2 automated **where** to intervene (which tokens). Section 3.3 automates **what** to intervene on — which subspace of the residual stream. Instead of training a separate R via gradient descent per concept (like the DAS tutorial's `R`), HyperDAS **generates** an appropriate R in a single forward pass.

Step 1 — v = MLP(e_E^(N))
- `e_E^(N)` = the concept-summary vector from Section 3.1.
- Passed through one more small MLP (same up-projection/ReLU/down-projection structure from the [[Transformers, Attention, and Multi Layer Perceptrons - Recap]] notes) → produces `v`, which drives the Householder trick.

Step 2 — R': the generic starting point
- A **fixed, low-rank matrix with orthogonal columns** — d rows, k columns (k = subspace dim, e.g. 128, far less than d = 4096).
- Same kind of object as the DAS tutorial's `R`, but here it's a fixed baseline, not the thing being directly trained.

Step 3 — the Householder reflection: $$H = I − 2vv^⊤/v^⊤v$$
**Term-by-term:**
- `I` = identity (d×d) — "do nothing" baseline.
- `vv^⊤` = outer product of v with itself → a full d×d matrix (different from v^⊤v, a dot product = single number).
- `v^⊤v` = v's squared length.

**Geometric verification — what H does:**
- Applied to v itself: `Hv = v − 2v = −v` → **flips v exactly**.
- Applied to any u perpendicular to v (v^⊤u = 0): `Hu = u − 0 = u` → **leaves u unchanged**.
- **Combined:** H flips the one direction v points in, leaves every perpendicular direction untouched — exactly a **mirror reflection** across the plane perpendicular to v.

**Why H is orthogonal:** a reflection preserves lengths/angles by nature — the defining property of orthogonality. (Algebraically: H is symmetric, H²=I ⟹ H^⊤H=I.)

Why R = R'H stays orthogonal?
> "H is orthogonal and R' has orthogonal columns, which means R'H has orthogonal columns."

- Multiplying a matrix with orthogonal columns by *any* orthogonal matrix preserves that orthogonality — <mark style="background: #ABF7F7A6;">applying the same reflection to two originally-perpendicular columns keeps them perpendicular to each other</mark>. (Formally: $c_i^⊤H^⊤Hc_j = c_i^⊤c_j$ since $H^⊤H=I$ — inner products, and thus orthogonality, exactly preserved.)

Putting it together: R = R'H
- Take the fixed generic R', multiply by <mark style="background: #FF5582A6;">the Householder reflection H built specifically from v (tied to whatever concept — "country," "birth year," etc.)</mark>. Result R is **guaranteed** to remain a valid orthogonal matrix, <mark style="background: #FF5582A6;">now "steered" toward the concept v represents</mark>.

Why did they use this specific trick?
- Mathematically **impossible** to produce an invalid (non-orthogonal) R, regardless of v.
- Just **one matrix multiplication** — no gradient descent, no training loop needed per new concept. <mark style="background: #BBFABBA6;">Feed a new v through this formula → instant, valid, concept-appropriate R.</mark>

#### Small note comparing to the DAS coding tutorial
- The DAS tutorial's `R` was produced via `torch.nn.utils.parametrizations.orthogonal(...)`, using `matrix_exp` internally to guarantee orthogonality *throughout gradient-descent training* (the exact operation that wasn't supported on MPS).
- HyperDAS's Householder trick achieves the same goal — a guaranteed-orthogonal matrix — via a completely different, far cheaper mechanism: ==**constructing** orthogonality directly via one reflection formula, rather than **training** toward it step by step.==

---
Equation 11 — The Actual Intervention
$$H_b^{(l)} \leftarrow \bar{h}_b^{(l)} + R^\top\left(R(\tilde{h}_b^{(l)}) - R(\bar{h}_b^{(l)})\right)$$
- structurally identical to Equation 1
- Equation 1 (general RAVEL/DAS formula): `H ← h̄ + R^⊤(R(ĥ) − R(h̄))`
- Same "rotate, subtract, un-rotate, add" pattern. **Exactly one thing changed.**

The one difference: ĥ → h̃_b^(l)
- **Equation 1:** counterfactual value = `ĥ`, a single fixed hidden state, no token selection involved.
- **Equation 11:** counterfactual value = **h̃_b^(l)**, which is exactly the quantity computed by **Equation 9** (Section 3.2):
$$\tilde{h}_b^{(l)} = G_{(b,C+1)}\bar{h}_b^{(l)} + \sum_{c=1}^{C} G_{(b,c)}\hat{h}_c^{(l)}$$
<mark style="background: #FF5582A6;">A learned, soft weighted blend — not a fixed value — combining "keep base's own value" with "pull in each counterfactual token's value," weighted by the learned attention matrix G.</mark>

What each symbol now means
- **$\bar{h}_b^{(l)}$** — base token b's hidden state at layer l (unchanged).
- **$\tilde{h}_b^{(l)}$** — the **automatically-selected** counterfactual value for token b, via Section 3.2's learned attention (replaces "just use ĥ directly").
- **R** — the **automatically-generated, concept-specific** rotation matrix from Section 3.3's Householder trick (replaces a matrix trained per-task).
- $R(\tilde{h}_b^{(l)}) - R(\bar{h}_b^{(l)})$ — ==difference between (auto-selected) counterfactual and base values, measured along R's directions.==
- **$R^⊤(...)$** — un-rotate back to original coordinate space.
- $\bar{h}_b^{(l)} + (...)$ — add onto base's original hidden state → updated H_b^(l), which continues through the target model's forward pass.

The full pipeline over the sections:
1. **3.1** → v (concept-summary vector).
2. **3.2** → uses v to compute G, then h̃_b^(l) via Eq 9 (automatic "which value to patch in").
3. **3.3** → uses v (via MLP + Householder) to compute R (automatic "which subspace to touch").
4. **Eq 11** → combines h̃_b^(l) and R into the actual patch — mechanically identical to `RotatedSpaceIntervention.forward()` from the DAS tutorial, now operating on automatically-computed inputs instead of hand-specified ones.

- Equation 11 is the precise math behind: *"HyperDAS performs an intervention by patching the subspace of the hidden vector for the token 'nna' to the value it takes on in the hidden vector for the token 'Paris.'"*
- $\tilde{h}_{b^{(l)}}$ supplies the "Paris" value (learned via G); R supplies the "country" subspace (learned via Householder) — both automated, where vanilla DAS would require a human to specify both by hand.

---
G's Many-to-One Flexibility, and the Roles of G vs. R

"A single counterfactual token paired with a weighted combination of multiple base tokens"
- **G's normalization:** softmax is column-wise — for a fixed counterfactual token `c`, weights down that whole column (across all base tokens) sum to exactly 1.
- **What this does NOT enforce:** nothing stops that column's total weight of 1.0 from being spread thinly across *many* base tokens instead of concentrated on just one.
- **Example:** "Paris" column could legally be `G_(nna,Paris)=0.4, G_(is,Paris)=0.3, G_(in,Paris)=0.2, G_(the,Paris)=0.1` — summing to 1, but "Paris" now influences 4 different base tokens simultaneously.
- **Why it matters:** this many-to-one flexibility works fine under soft/weighted training, but breaks at test time — the discrete argmax (Eq 14-15) forces strict one-to-one pairing, and a model relying on spread-out influence loses most of its signal when snapped to hard 0/1. This is exactly why the sparsity loss (`L_sparse`, Eq 13) exists — to penalize this spreading during training.

Row-view vs. column-view of G
- **Row view (what G is actually used for, in Eq 9):** for each base token, G gives the weighted combination of counterfactual tokens (+ "leave alone") that gets blended into that base token's updated value.
- **Column view (the "one counterfactual token → many base tokens" issue above):** for each counterfactual token, how many different base tokens is it feeding into simultaneously.
- Same matrix, two complementary readings — row view = what's computationally used; column view = what reveals the many-to-one problem sparsity loss addresses.

G vs. R — content vs. scope (Equation 11)
Formula: `H_b^(l) ← h̄_b^(l) + R^⊤(R(h̃_b^(l)) − R(h̄_b^(l)))`
- **G determines the CONTENT of the change** — via h̃_b^(l) (Eq 9): what value the base token would take on if fully replaced by the relevant counterfactual blend.
- **R determines the SCOPE of the change** — **which specific directions of the difference are allowed through**. ==Rotating into R's coordinate system before subtracting means any part of the difference lying outside R's subspace never enters the computation at all — it's a **filter**, not a destination.==
- **Concrete picture:** if h̄ and h̃ differ across all 4096 dimensions, but R is a 128-dim subspace, only those 128 directions' differences are ever computed/applied — the other ~3968 dimensions of the base token's original value stay completely frozen, regardless of how different h̄ and h̃ might be along those directions.

<mark style="background: #ABF7F7A6;">G tells you *what value* the base token would become if fully absorbing the relevant counterfactual information; R tells you *which dimensions* of that change are actually permitted through, leaving everything else untouched.</mark>

Equation 13 — Sparse Attention Loss (L_sparse)

What G_(*,c) and Sum(G_(*,c)) mean
- `G_(*,c)` = fix one counterfactual token c, look at its entire column (every base token's weight for that token).
- `Sum(G_(*,c))` = total weight-mass that counterfactual token c has spread across all base tokens.

Doesn't column-softmax already force sum = 1?
- Yes, `G = ColumnSoftmax(G')` forces each column to sum to exactly 1 by construction. The practical intuition that matters: **this loss penalizes a counterfactual token's influence being spread thinly across multiple base tokens** — the exact many-to-one pattern already identified as problematic.

The structure:
- **Sum ≤ 1** (well-behaved/concentrated) → **zero penalty**.
- **Sum > 1** (spread beyond an acceptable, concentrated amount) → penalty **= ==the excess itself**, growing linearly with how spread-out it is.==
- No penalty until crossing a line, then linear growth.

Averaging across all counterfactual tokens
- `(1/C) Σ_c (...)` — compute the excess penalty separately for every counterfactual token, then average → <mark style="background: #FFB8EBA6;">one scalar loss representing "on average, how much is influence spreading beyond clean one-to-one correspondence" for this example.
</mark>

Why zero-below-threshold, not always-penalize
- Doesn't force *perfect* one-to-one correspondence during training (would break differentiability) — just discourages the worst cases where influence is too spread out to plausibly collapse into a hard one-to-one pairing later. A soft, gradient-friendly nudge toward sparsity, not a hard constraint.

- Final loss: `L = L_RAVEL + λL_sparse`
- Early training: L_sparse has near-zero influence — model freely finds any soft solution to RAVEL, however spread-out.
- Second half of training: sparsity penalty ramps up, pressuring the model to consolidate toward genuine one-to-one token pairing — needed to survive the hard arg-max snapping at test time.

#### HyperDAS Figure 2

The example
- **Base:** "Springfield is a city in the country of" → model originally outputs "China."
- **Counterfactual:** "The city of Macheng's official language is."
- **Targeted attribute:** country. Correct intervention should redirect the output toward Macheng's actual country (The United States) instead of China.

Panel (a) — Raw Intervention Score (soft, before snapping)
- Grid: rows = counterfactual tokens ; columns = base tokens
- **Key values (highlighted red in paper):** `(Mach, field)=0.24`, `(eng, field)=0.67` — the two token-pieces of "Macheng" both score meaningfully on "field" (second half of "Springfield," the entity-name token in the base).
- <mark style="background: #FF5582A6;">Confirms the model correctly identified the entity-name tokens as the relevant pairing (Macheng ↔ Springfield's name-token), even before locking onto one clean pairing.</mark>
- Bottom "[SELF]" row (leave-alone option) has high values (0.78–0.99) across most base tokens → ==most tokens correctly left untouched; only the entity-name token gets meaningfully intervened on.==

Panel (b) — Intervention Score at Inference Time (hard-snapped)
- Same computation, now with the double-argmax operation (Eq 14–15, Section 3.6) forcing strict 0/1 decisions.
- **Dramatic simplification:** entire grid collapses to 0 except one cell: `(eng, field) = 1`.
- Every other base token's "[SELF]" (leave-alone) entry becomes 1; only "field" gets fully replaced, entirely by "eng"'s value.

Why this is a good outcome?
- Panel (a)'s soft spread (both Mach and eng scoring on field, plus minor residual noise) is exactly the pattern `L_sparse` (Eq 13) is designed to discourage during training.
- Because sparsity loss did its job, the soft distribution was already reasonably concentrated — unlike Figure 7's "no sparsity loss" condition, where influence smeared across many base tokens and snapping would have destroyed the model's actual learned behavior.

Connecting to Equation 11:
- The single surviving cell `(eng, field)=1` becomes exactly `h̃_field^(l)` in Equations 9/11 — =="field"'s hidden state gets fully replaced (h̃ = ĥ_eng) by "eng"'s hidden state, restricted to the country-subspace defined by R==. Concrete end-to-end trace: swap "field" (Springfield) for "eng" (Macheng) in the country subspace only, leave everything else untouched → model now correctly outputs Macheng's country instead of defaulting to China.

Masking of the Base Prompt (Trivial Solution & Fix)

The example
$$\begin{cases} \text{Base: } \bar{x} = \text{Vienna, known for its Imperial palaces, is a city in the country of} \\ \text{Counterfactual: } \hat{x} = \text{I love Paris} \\ \text{Instruction: } x = \text{Localize the latitude of the city} \end{cases}$$

- Target attribute (from instruction) = **latitude**.
- Base sentence's own query = **country** (it literally ends "...in the country of").
- These **mismatch**.

What should happen (correct behavior):
- HyperDAS should genuinely locate the **entity token "Vienna"** (not the query phrase), find/patch specifically the **latitude subspace** within it, and leave every other subspace (including country) untouched. Since country was never touched, the model still correctly answers "Austria" when it reaches the query.

Superposition applies here!
- A single entity token ("Vienna") encodes **many attributes simultaneously** in different subspaces of the same vector — country, latitude, continent, language, etc. — same idea as the earlier superposition/Michael-Jordan discussion.
- RAVEL tests whether a method can touch just one attribute's subspace without disturbing the others, all coexisting in that one token.

The shortcut the hyper-network can learn instead
- Because the hyper-network cross-attends to the base prompt's hidden states (Section 3.1), it can "see" what attribute the base sentence's own query is asking about (here: country) — in addition to the instruction's stated target attribute (latitude).
- This lets it learn a cheap rule: ==**if target attribute matches the base sentence's own query, actually intervene; if they mismatch, default to [SELF] (leave everything alone).**==

Why the shortcut passes evaluation without doing real work
- When target (latitude) ≠ base query (country): doing **nothing** still produces the correct answer (Austria), since the base's own country knowledge was never at risk.
- This satisfies the **Iso score** — but the hyper-network never even attempted to locate the latitude subspace.
- ==It's not disentanglement; it's lazily skipping the hard cases (mismatches) where genuine disentanglement would need to be demonstrated==.
- Confirmed empirically in Figure 9 (Goroka/Zurich-Country example): raw intervention grid is entirely zero except [SELF]=1 everywhere.

Why this matters?
- Perfect Iso score for the wrong reason: never intervening, rather than intervening precisely and correctly avoiding unrelated attributes.
- Directly relevant cautionary example for my thesis's commutativity-criterion framing: a supervised interpretability method might look successful on standard metrics while having learned a shortcut rather than the intended causal structure — motivates why independent validation matters over self-validating metrics.

The fix — masking the query phrase (NOT the entity token)
$$\bar{x} = \text{Vienna, known for its Imperial palaces, is a city in the } \textcolor{red}{\text{country of}}$$

- Only the **query phrase** ("country of") is masked — hidden from the hyper-network's cross-attention. "Vienna" itself stays fully visible.
- With the query phrase hidden, the hyper-network **cannot see what the base sentence's own query is about** — it can no longer compare target vs. base attribute to decide whether to skip intervening.
- <mark style="background: #FF5582A6;">Forces the hyper-network to make its localization decision purely from the instruction + the entity token, rather than gaming the shortcut.</mark>

Clarifying misconception: "no tokens to replace" when mismatched?
- **Not correct** — there are always tokens available (the entity token "Vienna," holding all attributes in superposition).
- The problem was never a lack of intervention targets; it was that the hyper-network could see *extra* information (the query phrase) letting it realize it didn't need to bother finding the real target at all.
- Masking removes exactly that extra information, not the entity token itself.