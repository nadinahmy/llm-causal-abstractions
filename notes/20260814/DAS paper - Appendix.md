
---
Appendix A.1 — Building the Training Dataset for R

What each training example consists of?
- Each training example = one **base input** (a full setting of all input objects) + one or more **source inputs** (used purely to harvest values from) + a **ground-truth counterfactual label** (computed from the symbolic high-level model).

When multiple source inputs are needed
- **Single intermediate variable being tested** (e.g. just $V_1$) → only **one** source input needed, since only one variable is being intervened on
- **Multiple intermediate variables involved simultaneously** (e.g. both $V_1$ and $V_2$, the "Both Equality Relations" hypothesis) → **two** source inputs needed — one dedicated to harvesting each variable's new value

==Each source input is only ever queried for the **one** variable it was designated for== — its own value for any other variable, and its own output, are never used.

The random sampling rule:
For each training example, randomly choose between two modes:
 - **Single intervention** — sample one source input, intervene on only one variable (chosen at random), leaving the other variable computed normally from the base
- **Double intervention** — sample two source inputs, intervene on both variables simultaneously (one source feeds each variable)

**Why mix both modes ?:** ==training only on double-interventions risks $R_\theta$ learning $V_1$ and $V_2$ as only meaningful *together* — never separately==. Mixing in single-variable interventions ==forces each subspace ($Y_1$, $Y_2$) to be independently, robustly meaningful on its own== — exactly the property a genuinely "distributed" alignment needs.

<mark style="background: #ABF7F7A6;">Exactly when source inputs get used?</mark>
1. **Computing the ground-truth label** — take the base, but with the target variable(s) forced to the value(s) computed from the source input(s) via the *symbolic* model $B$. This produces the <mark style="background: #FFB8EBA6;">counterfactual label</mark> for this example.
 2. **Performing the actual distributed intervention on the network** — the source input's hidden representation gets rotated into $Y$-space, and <mark style="background: #FFB8EBA6;">only the subspace it was designated for gets copied into the base's rotated representation, overwriting the base's own value there.</mark>
 Outside of these two moments, a source input is never touched again for that example.

What happens to the ground-truth counterfactual label?
- The label becomes a **delta distribution** (100% on the correct answer, 0% elsewhere) — the target for <mark style="background: #BBFABBA6;">cross-entropy loss</mark> (Definition 4). It's compared against the network's own probability distribution, produced by running the base + sources through the full distributed-intervention pipeline (rotate → swap targeted subspace(s) → rotate back → forward pass → softmax) using the *current*, not-yet-converged $R_\theta$.
- The mismatch (cross-entropy) is ==back-propagated **only into $R_\theta$==** — the network's own weights stay frozen throughout. This is repeated over a large dataset of such examples (640K for the hierarchical equality experiment, mentioned in Appendix A.2), across many epochs, until $R_\theta$ converges.

Raw-identity hypothesis
 - "maybe the network just carries forward object identity, not any equality relation."

Task-Specific Sampling Recipes

==Hierarchical Equality==
1. "Both Equality Relations" and "Left Equality Relation"
	- Swap the **equality relation itself** (the boolean value of $V_1=(w{=}x)$ and/or $V_2=(y{=}z)$) — not the raw shapes. If source pair is $(D,D)$ → its equality relation is True → that "True" gets interchanged into the base; base's own shapes stay unchanged.
	- "Both Equality Relations" → 1 or 2 source inputs (per the single/double mixing rule). "Left Equality Relation" → always 1 source (only $V_1$ tested).
2. "Identity of First Shape"
	- Structurally different — ==swaps a **raw identity**, not a relation==. Source $(D,E,F,G)$, base $(A,A,B,C)$ → base's $w{=}A$ becomes $w{=}D$; $x,y,z$ unchanged. ==Tests the deflationary hypothesis from Section 4.2: does the network just carry forward object identity rather than compute equality?==

Monotonicity NLI
1. "Negation" or "Lexical Entailment"
	- Same pattern — swap a **boolean value**, not raw words. If source has negation present, swap "negation = True" onto the base regardless of which words the source actually used. Same for entailment: swap the source's True/False entailment value, not its word pair.

2. "Identity of the Replacing Lexeme"
	- Raw-identity analogue for MoNLI (parallel to "Identity of First Shape"). Swaps in a completely different **word** from a different example, ==tests whether that word's identity is recoverable in the network's representation.==
	<mark style="background: #FFB8EBA6;">	Why this last one needs a validity constraint?</mark>
	- Randomly grabbing any replacement word risks nonsensical/ambiguous examples — e.g. swapping in "tree" where "car" was expected has no clear entailment relationship (not hypernym/hyponym of each other), so there's no principled ground-truth label.
	The fix — three steps :
	 1. Pick a valid English word that is specifically a **hypernym or hyponym** of the premise's original word — guarantees a well-defined entailment relationship
	 2. Draw a fresh **sentence template** from the training set (a sentence with a lexeme "slot," e.g. "a man is talking to someone in a [lexeme]")
	 3. Plug the new word into the template to construct a brand-new, valid premise/hypothesis pair
	==Ensures every generated example has a well-defined, unambiguous ground-truth label== — same underlying concern as always (need a clean counterfactual label to train against), plus an extra validity check specific to language.
	- This hypothesis tests whether the network's "entailment" representation secretly still contains the _specific word_ $w_h$​ as a separately swappable piece — verified by substituting in a totally different (but relationally valid) word from a freshly constructed sentence, and checking whether the network's output correctly updates to reflect the _new_ entailment relationship between the unchanged premise word and the newly swapped-in word, rather than the original one.

All six sub-recipes follow the same basic pattern: sample a base, sample source material, swap something specific from the source into the base, compute the ground-truth counterfactual label. What changes is ==**what gets swapped**==:
 - **"Genuine variable" hypotheses** (equality relations, entailment, negation) → swap an **abstract relation**
 - **"Deflationary/identity" hypotheses** (first shape, replacing lexeme) → swap a **raw identity**

### MoNLI's core setup

Every example is a premise/hypothesis pair that differ by exactly one word: $w_p$​ in the premise, $w_h$​ in the hypothesis.
**Example:**
- Premise: "A man is talking to someone in a taxi." → $w_p$ = taxi
- Hypothesis: "A man is talking to someone in a car." → $w_h$ = car
- Label: **entails** (since "taxi" is a kind of "car" — taxi is a hyponym of car, so the premise being true about a taxi makes the hypothesis, about a car, also true)
- **"Lexical Entailment" hypothesis:** does the network represent, as an intermediate variable, just the _boolean fact_ "does $w_p$​ entail $w_h$​?" (True/False) — nothing about which specific words they are
- **"Identity of the Replacing Lexeme" hypothesis:** does the network _also_ separately represent the specific _identity_ of $w_h$ itself — i.e., not just "yes these entail," but "the word is specifically car" — as something recoverable and swappable on its own
---
Appendix A.3 — Brute-Force Search Baseline Mechanics
- Describes how Step 4 (Section 3.7) — the "old way" alignment search — actually works: no rotation, no learning, just picking raw neurons.

The setup:
- Hidden layer of, say, 16 neurons in order: 1, 2, 3, ..., 16. Intervention size chosen in advance (e.g. 4) — how many neurons should represent one high-level variable.
What "mapping to low-level variables" means here ---> Localist alignment $\Pi$ assigns a high-level variable (e.g. $V_1$) to a group of **individual, literal neurons** — no rotation, no blending. Just: which raw neurons "count" as $V_1$?

The sliding window restriction:

> [!important]
> Instead of checking every possible combination of neurons (intractable —  Appendix D's note on $C_n^m$ exploding), only **contiguous blocks** of neurons are considered, slid one position at a time:
> - Window 1: {1,2,3,4}
> - Window 2: {2,3,4,5}
> - Window 3: {3,4,5,6}
> - ... up to Window 13: {13,14,15,16}
>
> Window width = the intervention size for that experiment (matches DAS's own hyperparameter, e.g. |N|=16, intervention size=8 from Table 1). Fewer/wider windows as intervention size grows: 16 − size + 1 total positions.

The search procedure:
	For each window position:
	1. Run a plain interchange intervention (Definition 2, no rotation) using exactly that neuron group as the candidate $V_1$
	2. Compute IIA on validation data
	3. Move to next window
Report whichever window scored highest — ==this is the number reported in Table 1's "Brute-Force Search" row.==

Why sliding windows instead of exhaustive subsets?
- Checking every possible subset of neurons explodes combinatorially as layer size grows (Appendix B's "BFSmax" column, Appendix D's FAQ both flag this as intractable). Sliding contiguous windows is much cheaper — only ~13 (or fewer) positions checked instead of potentially millions of subsets.
- **Cost:** ==can only find alignments where the "right" neurons happen to sit *next to each other* in the layer's arbitrary ordering.== If the true structure spans neurons scattered at positions 1, 5, 9, 13 — a sliding window can never find it. Only DAS's rotation, which isn't restricted to contiguous neurons at all, could uncover that.

Why this explains brute-force's consistent underperformance?
Brute-force is restricted twice over:
1. Individual raw neurons only — no blending/rotation (same limit as the "Localist Alignment" baseline)
2. **Contiguous** groups only — an even narrower search than the full localist space would allow
<mark style="background: #FF5582A6;">DAS's rotation-based search faces neither restriction — which is exactly why it consistently beats brute-force search in Table 1.</mark>
---
### Super important - Algorithm 1
#### Finding the Localist Alignment Matrix

The pseudocode behind Step 5 ("closest localist alignment" baseline) — the actual procedure for snapping DAS's learned rotation matrix $R$ onto the nearest neuron-only alignment.

What's the goal?
- $R$ mixes all neurons together (fractional weights everywhere). Find $L$ — the closest **signed permutation matrix**: <mark style="background: #FF5582A6;">each new axis assigned to exactly one original neuron, possibly sign-flipped, no two axes sharing a neuron</mark>.

Pseudocode:
```
FindLocalistAlignment(R)
1  Ra = R.absolute_value()
2  L = torch.zeros_like(R)
3  P = []
4  for i = 0; i < R.shape[0]; i++
5      P += [(Ra == torch.max(Ra)).nonzero()]
6      Ra[P[-1].row, :] = 0.
7      Ra[:, P[-1].col] = 0.
8  for p in P
9      L[p.row, p.col] = 1.
10 P = P * get_sign(R)
11 return L
```

> [!note] `Ra = R.absolute_value()`
> Take absolute values — at this stage only the *strength* of each rotated-axis/neuron association matters, not sign (handled separately later).

> [!note] Greedy selection loop (repeated once per row)
> Each iteration:
> - Find the single largest remaining value in `Ra` → strongest remaining rotated-axis/neuron association
> - Zero out its entire **row** — that rotated axis is now claimed, can't be picked again
> - Zero out its entire **column** — that neuron is now claimed too, no other axis can take it
>
> Result: greedily pick the strongest remaining match, one at a time, removing both sides from future consideration.

Why greedy, not exact?
- This is an **assignment problem** (matching $n$ axes to $n$ neurons one-to-one, maximizing total association strength). Maybe an exact optimum exists, but this is a simpler greedy algorithm — not guaranteed optimal, but fast and captures the same intuition: always take the strongest signal first.

> [!note] Building L and restoring sign
> - Set $L[\text{row},\text{col}]=1$ for each matched pair → $L$ becomes a **permutation matrix** (exactly one 1 per row/column)
> - `P * get_sign(R)` ==restores sign info discarded earlier== — checks whether the original $R$ entry was positive or negative, so $L$ preserves any legitimate sign flip (reflection)

Connecting to the earlier plain-English description in DAS pipeline:
- "for each new axis, find the neuron it points most strongly toward, snap to it (possibly flipped)." The one added subtlety: processing happens in order of overall strength with ==claimed neurons **removed from consideration** — guarantees a valid one-to-one matching, so two rotated axes never snap to the same neuron.==

Once $L$ is built, it's used instead of $R$ to re-run the same interchange-intervention evaluation → t==he resulting IIA is Table 1's "Localist Alignment" row==. Since $L$ discards all blending info (keeps only the single strongest neuron per axis), the drop from DAS's own IIA to this localist-snapped IIA (e.g. 1.00 → 0.73) is a ==direct, quantified measurement of how much the network's true representation depends on **combining multiple neurons**, rather than living in any one alone.==

Appendix C - Does the rotation matrix found really have an impact at all?
> [!important]  
> This is direct empirical support for the core distributed-representation story of the whole paper. ==If DAS's learned rotations were close to trivial (angles bunched near 0°), it would suggest that the "distributed" part of "distributed alignment search" wasn't doing much real work — that whatever DAS found could have been found almost as well by a plain neuron-aligned (localist) search==. Since the rotations are _substantially_ non-trivial across most basis directions, this is consistent with — and supports — the paper's repeated finding elsewhere (the DAS vs. localist-alignment gap in Table 1 and 2) that the relevant concepts genuinely live in _combinations_ of neurons, not in any rotation-free, near-identity transformation of the raw neuron basis.

---
Some notes from the FAQ section:
 When people say a **matrix** is <mark style="background: #FF5582A6;">orthonormal</mark> (or, more precisely, "has orthonormal columns" — this is basically synonymous with what the paper calls an "orthogonal matrix"), they mean: <mark style="background: #FF5582A6;">every column of the matrix, treated as a vector, is length 1, and every pair of columns is perpendicular to every other pair.</mark>
	- if you let gradient descent freely update a matrix's entries without this constraint, ==there's nothing stopping it from drifting into a matrix that stretches or shears the space instead of purely rotating it — which would break the entire "R is a valid rotation preserving all information" assumption the whole DAS method depends on.==

==DAS is a tool for _testing and refining_ a causal hypothesis you already have some version of — confirming whether it's genuinely implemented inside a network, and at what granularity — rather than a tool that hands you a causal graph from scratch with no prior hypothesis at all.==

Quoted directly from FAQ section : What are some practical usage of DAS?
- Practically, DAS transforms representations into an operatable state where interchange intervention results in interpretable model behaviors. ==DAS, itself, is a powerful tool for conducting causal abstraction analysis of a neural network.==
