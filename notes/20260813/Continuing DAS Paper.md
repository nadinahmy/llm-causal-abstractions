
---
- Quote from paper : "a neural network specifies an output distribution for a given input"
	- they mean: instead of thinking of the network's output as one label, think of it as a full probability distribution over the possible labels.
- Definition 4 pre-quote : In the following definition we assume that a neural network specifies an output distribution for a given input, which can then be pushed forward to a distribution on output values of the high-level model via an alignment function τ
	- ==the network gives you probabilities, not just hard answers== — and we can translate those probabilities into the high-level model's terms using τ — so that we have something differentiable to optimize.
---
Deterministic Models as Delta Distributions — Why & How

The mismatch being solved
- The **network** N outputs a probability distribution (e.g. 73% True, 27% False)
- The **symbolic model** B is deterministic — it just gives one definite answer, no uncertainty at all

To compare them with a loss function, both sides need to be the *same kind* of object. ---> fix: ==delta distributions==

> [!note] Delta distribution
> Even though B is deterministic, treat its output as a distribution too — just one that's completely certain. It puts 100% of the probability on the one value B actually outputs, and 0% on everything else.

Example: if B outputs "True" with certainty:

| Outcome | Probability |
|---|---|
| True | 100% |
| False | 0% |

This adds no new information — B was always going to say True — it just **relabels** that certainty in the same format as the network's probabilistic output, so the two become directly comparable.
> [!important]
> Now both sides are probability distributions over the same output space:
> - Network's (pushed-forward via τ) distribution — e.g. 73% True / 27% False
> - Symbolic model's distribution — always either 100%/0% or 0%/100%, since deterministic
>
> Comparing two distributions is something you can compute a **smooth, differentiable loss** on — specifically **cross-entropy**. Comparing "a probability" against "a hard label" directly would either need an ad hoc rule or lose the smoothness needed for gradient descent.

"After interchange intervention" — when this comparison happens
Not on ordinary forward passes — specifically after running an intervention on both models. Training loop:
1. Run high-level model B with an intervention (e.g. forcing V1 = T from a source) → get deterministic output → represent as delta distribution
2. Run network N with the corresponding distributed intervention (using current rotation matrix R) → get probabilistic output → push forward through τ into B's output space
3. Compute cross-entropy loss between the two distributions
4. Back-propagate to update R so the network's counterfactual behavior matches B's more closely
This step doesn't add conceptual depth — it just puts the deterministic symbolic model's answer and the network's probabilistic answer into the ==**same mathematical forma==t** (both distributions), so that after an intervention they can be plugged into cross-entropy loss and used to train R via gradient descent. [Direct setup for definition 4]
---
Definition 4 notes:
- **The summation** — why sum over many b, s1,...,sk combinations? Because you're training $R^θ$, and training needs many examples, not just one. So in practice, you'd sample lots of different base inputs and lots of different combinations of source inputs, run this comparison for each one, and add up (or average) the losses. This is exactly like training any neural network parameter with a dataset — except here what's being trained is the rotation matrix, and the "dataset" consists of intervention scenarios rather than ordinary labeled examples.
- **N** answers "==which chunk of the network are we even searching in?==" (a layer, a set of neurons) — a structural decision we make in advance. **The subspace dimension** (|Y1|, |Y2|, etc.) answers "==once we've rotated that chunk, how many of the new coordinate axes do we hand over to this particular high-level variable, versus leaving alone?==" — also a structural decision we make in advance, before gradient descent ever starts. Both are choices we'd typically sweep over (try several values, as Table 1 does) to see which combination gives the best-aligned, most faithful representation — the R matrix itself is the only thing gradient descent actually learns _within_ a given choice of N and subspace size.

The DAS Training Objective
- **Goal:** find the rotation $R^θ$ that makes the network's counterfactual behavior line up as closely as possible with the symbolic model's counterfactual behavior.

Ingredients
- **L** — the low-level neural network (e.g. N, the boolean conjunction network)
- **Inputs_L** — all possible inputs to L
- **H** — the high-level symbolic model (e.g. B)
- **Out_H** — H's possible outputs (True/False)
- **τ** — the alignment/translation function between low- and high-level (already fixed — recall from 3.2, τ for inputs/outputs is "given for free")
- **X_j** — intermediate high-level variables we want to find a match for (e.g. V1, V2)
- **N** — a set of neurons in L (a target population/layer) — *not the same N as the network above, just reused notation*
- **R_θ** — the thing being learned: a rotation matrix mapping N into rotated space Y. θ = trainable parameters.

General form of the objective
> Sum, over many (b, s1,...,sk) combinations, of:
> $Loss( DII(L, R_θ, {sj}, {Yj})(b) , II(H, {τ(sj)}, {Xj})(τ(b)) )$

> [!note] Unpacking the two arguments
> - **Network's side** — $DII(L, R_θ, {sj}, {Yj})(b)$ : the distributed interchange intervention (Def. 3) applied to L — rotate with R_θ, swap in source values along subspaces Yj, rotate back, get counterfactual output.
> - **Symbolic model's side** — $II(H, {τ(sj)}, {Xj})(τ(b))$ : a plain interchange intervention (Def. 2) applied to H, but with base/sources translated through τ first — feed H symbolic values (T/F), not raw network numbers.

**Loss** here is left generic — any differentiable function measuring how far apart two outputs are.

**Why sum over many b, s1,...,sk?** ==Because we're training $R^θ$, and training needs many examples, not just one.== Sample lots of base inputs and source combinations, run the comparison for each, sum/average the losses — same idea as training any NN parameter with a dataset, except the "dataset" here is intervention scenarios, and what's being trained is the rotation matrix.

The specific loss actually used: cross-entropy
$$
\text{CE}\Big(P(\text{out}_H \mid II(H,\{\tau(s_j)\},\{X_j\})(\tau(b))),\ P^\tau(\text{out}_H \mid DII(L,R_\theta,\{s_j\},\{Y_j\})(b))\Big)
$$

> [!note] Two terms being compared
> - **First term (symbolic side):** the distribution over high-level outputs after intervening on H. Since H is deterministic → this is the **delta distribution** (100% on the actual output, 0% elsewhere).
> - **Second term (network side):** the network's own output distribution after the distributed intervention, **pushed forward through τ** into H's output space.

Cross-entropy measures how well the network's (pushed-forward) probability distribution matches the symbolic model's near-certain answer. Network confidently correct → low loss. Network confidently wrong or uncertain → high loss.

==Example (boolean conjunction)==
1. Base = [F,T], source = [T,T]
2. **Symbolic side:** intervene on V1 in B → B outputs True → delta distribution: 100% True
3. **Network side:** using current R_θ (maybe not yet the ideal −20° rotation) — rotate base + source hidden values, swap Y1 subspace, rotate back, compute O, convert O → probability of True/False (sigmoid/softmax)
4. Compute CE between "100% True" and the network's current probability
5. One term in the big sum — repeat over many base/source pairs, sum, minimize

> [!important] Not everything is learned by gradient descent
> **Fixed in advance (discrete hyperparameters, not optimized by SGD):**
> - **N** — which population/layer of neurons we're targeting
> - **|Y0|, |Y1|, ..., |Yk|** — how many rotated dimensions each subspace gets
>
> **Learned by SGD:** only **R_θ** — given N and subspace sizes already fixed.

> [!question] N = which chunk of the network are we even searching in?
> A structural decision made in advance — e.g. "the 16 neurons of Layer 1" vs. "Layer 2" vs. "Layer 3." **Not learned** — you pick it (or sweep over several choices), then search for an alignment *within* that choice.
>
> Table 1 shows exactly this sweep: same network, same method, but IIA for "Both Equality Relations" drops from 1.00 (Layer 1) → 0.57 (Layer 2) → 0.50/chance (Layer 3). ==The concept may simply not survive intact into later layers — no rotation can recover information that isn't there anymore.==

> [!question] Subspace dimension (intervention size) = how many rotated axes go to this variable?
> Once N is picked (say 16 neurons), $R^θ$ re-expresses those 16 numbers as 16 rotated coordinates Y. We then choose how many of those rotated coordinates to hand to Y1 (aligned with the high-level variable) vs. leave in Y0 (untouched "leftover," per Def. 3's soft-intervention idea).
>
> Also a structural choice, also swept over in Table 1 (intervention sizes 1, 2, 8 for |N|=16). More dimensions given to the variable → generally better IIA, since the concept has more "room" to be expressed — up to a point.

**Concrete example:** N = 16 neurons of Layer 1, intervention size = 8 → learn a 16×16 R_θ, then treat the first 8 rotated coordinates as Y1 (aligned to V1) and the remaining 8 as Y0 (untouched).

> [!Summary]
> Definition 4: to train a rotation matrix aligning a network's internal representations with a symbolic model's intermediate variables — repeatedly sample a base input plus source inputs, compute what the network *would* output under a distributed intervention (current $R^θ$), compute what the symbolic model *actually* outputs under the equivalent intervention (as a delta distribution via $τ$), measure the mismatch with cross-entropy, and use gradient descent to shrink it. Repeat over many sampled interventions → end up with the $R^θ$ that best aligns the network's rotated subspace with the symbolic variable. ==This is what makes DAS a **learned** alignment search rather than brute-force or hand-picked.==

---
- Dimensionality of the sub-spaces used for each high-level variable:
![[IMG-20260813222819990.png|535]]
Once you've picked N (say, the 16 neurons of Layer 1), and trained a 16×16 rotation matrix R, you have 16 rotated coordinates to work with. "Dimensionality of the subspace for each high-level variable" just means: ==**how many of those 16 rotated coordinates do you hand over to each variable, versus leave alone.**==

Table 1's caption says the intervention size k is split between the two variables (roughly k/2 each, rounded). So for |N| = 16 and intervention size = 8:

- **4 rotated dimensions** get assigned to Y1, aligned with V1 ("is the first pair equal?")
- **4 rotated dimensions** get assigned to Y2, aligned with V2 ("is the second pair equal?")
- **8 rotated dimensions** are left in Y0, the untouched leftover — this is the "soft intervention" background that stays as the base input had it

> [!Summary]
When you run a distributed interchange intervention, you only ever swap values within the 4 blue cells (to test V1) or the 4 teal cells (to test V2) — never touching the 8 gray cells. <mark style="background: #FF5582A6;">If instead you'd chosen intervention size = 2 (so only 1 dimension per variable), each variable would have far less "room" to express itself in the rotated space, which is exactly why Table 1 shows lower IIA at smaller intervention sizes for the same layer — 1 dimension per variable is often too cramped to capture the full concept, while 4 gives it enough room to be found and expressed cleanly.</mark>

---
- R is an ==orthogonal matrix==
	- Orthogonal matrices are always square, and being orthogonal is precisely the mathematical property that guarantees "this transformation only rotates/reflects, it never stretches, shrinks, or changes dimensionality." So: N has 16 neurons → the space you're rotating is 16-dimensional → R must be square and 16×16 to rotate within that same 16-dimensional space without losing or inventing any information.
---
Section 3.7 — General Experimental Setup

- Short, administrative section — lays out the six-step recipe both experiments (hierarchical equality, MoNLI) follow.

The two tasks
- **Hierarchical equality** — feed-forward network trained on (w=x)=(y=z)
- **Monotonicity NLI (MoNLI)** — pre-trained BERT fine-tuned on an NLI task

> [!note]
> Both networks reach 100% train/test accuracy first. This matters — <mark style="background: #FF5582A6;">it means any imperfect IIA found later is about *how* the network solves the task internally, not *whether* it solves it.</mark>

The six-step evaluation paradigm

> [!important] Step 1 — Train the network
> Ordinary supervised training, nothing DAS-specific. Get a working, perfectly-accurate network first.

> [!important] Step 2 — Build interchange intervention training data from a high-level causal model
> This is where R's "dataset" comes from (Def. 4 needs many sampled base/source scenarios, not ordinary labels). For each example: pick a base input + source input(s), decide which high-level variables are intervened on, compute the counterfactual label from the high-level model (B, or the MoNLI program). This label is what the network's pushed-forward output gets compared against via cross-entropy.

> [!important] Step 3 — Learn R by maximizing IIA
> Definition 4 in action — train $R^θ$ via gradient descent + cross-entropy on the Step 2 dataset. They sweep different **hidden dimension sizes** ($|N|$) and different **intervention site sizes/locations** — discrete hyper-parameters, not learned by SGD (see earlier notes on N and subspace dimensionality). This sweep is exactly what produces the rows/columns in Tables 1 and 2.

> [!important] Step 4 — Brute-force search baseline
> The "old way" — exhaustively (or near-exhaustively) search alignments of high-level variables to ==*groups of individual neurons*==, no rotation. Comparison point: does DAS's learned rotation beat brute-force?

> [!important] Step 5 — "Closest" localist alignment baseline
> Different from Step 4. Take the rotation DAS *already learned*, snap it back onto the nearest standard neuron axes, and see how well that reduced version performs. Isolates: is DAS better because it searches better, or specifically because it can use non-neuron-aligned directions?

> [!important] Step 6 — Test whether the learned representation decomposes further
> Freeze the first learned rotation matrix, train a **second** rotation matrix on top of it targeting a sub-concept (e.g. does "lexical entailment" decompose into two word-identity representations?). This is what revealed the paper's key twist: MoNLI's entailment representation *decomposes* into word identities, while hierarchical equality's representation genuinely does *not* — a real difference between tasks only this test could surface.

#### Summary of the DAS pipeline
- Train a network to perfection → build a dataset of intervention-based counterfactual examples from a hypothesized high-level model → train a rotation matrix via DAS on that dataset → compare against two baselines (brute-force over raw neurons; nearest neuron-aligned snap of DAS's own solution) → test whether the found alignment decomposes into simpler sub-representations.
---
Step 5 — "Closest Localist Alignment" Baseline

What DAS produces?
- DAS trains a rotation matrix R that, in general, ==mixes many raw neurons together into each new axis — it's a *blended* rotation, not aligned to any single neuron==. (Recall the toy example: the learned direction Y1 wasn't neuron H1, it was cos(20°)×H1 − sin(20°)×H2 — a genuine blend.)

- The question Step 5 answers ---> How much of DAS's success comes specifically from being able to use blended, non-neuron-aligned directions? Maybe the blending barely matters — or maybe it's doing all the work. You need a second number to find out.

How the comparison number is built?
- <mark style="background: #ADCCFFA6;">Snapping R onto neurons</mark>
- Take the rotation matrix R that DAS already learned. ==Force it back onto the **nearest** alignment using only individual neurons — no blending==. For each new axis, find the single original neuron it points most strongly toward, and snap that axis to point at *exactly* that neuron (possibly flipped in sign), ignoring everything else it was blending in.
- This produces a cruder rotation matrix L — the closest possible localist approximation of what DAS found. (Algorithm 1 in appendix)

Step 5 vs Step 4 — the key distinction

|                   | Step 4 (brute-force)                                                        | Step 5 (closest localist)                                                 |
| ----------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Approach          | Searches many *different* localist alignments from scratch                  | Takes the *one* localist alignment nearest to DAS's own learned R         |
| Question answered | "How good is the best possible neuron-only alignment, found independently?" | "If I strip DAS's blending ability away, how much does performance drop?" |
- ==If Step 5's score is much worse than DAS's own score, that specifically shows the *blending* — the "distributed" part of distributed alignment search — is essential. The concept genuinely doesn't live in any single neuron, only in a combination of several.==

Grounding using Table 1 numbers
**"Both Equality Relations," Layer 1, |N| = 16, intervention size 8:**

| Method | IIA |
|---|---|
| DAS | 1.00 |
| Localist Alignment (Step 5 — closest snap of DAS's R) | 0.73 |
| Brute-Force Search (Step 4 — independent search) | 0.60 |

> [!tip] Reading the gap
> - **1.00 vs 0.73:** even the best possible neuron-only approximation of what DAS found loses real accuracy — confirms the concept is genuinely distributed, not sitting in one neuron.
> - **0.73 vs 0.60:** snapping DAS's already-good rotation to nearby neurons gives a head start that a blind independent search doesn't have — so the closest-localist baseline naturally beats brute-force too.
---

---
Section 4.1 — Describing the Low-Level Neural Model they used in their experiments
- The architecture actually used for the hierarchical equality experiment — the network DAS gets applied to.

Architecture
- Three-layer feed-forward network, ReLU activations. A stack of layers, each doing a linear transformation followed by ReLU (zeroes out negatives, keeps positives unchanged).

Input representation
- Random vector embeddings
- Task: four objects w, x, y, z, label = $(w{=}x) = (y{=}z)$. Instead of feeding symbols like "A" or "B" directly, each distinct object gets its own **randomly initialized vector** in $\mathbb{R}^n$, fixed once and reused — "A" becomes some vector like [0.3, -1.2, 0.8, ...], "B" a different random vector, etc.
- Standard way of turning discrete symbols into something a network can process (similar to word embeddings).
- Full input = the four object-vectors **concatenated**: $[x_1; x_2; x_3; x_4]$ — one long vector.

The formulas
$$
h_1 = \text{ReLU}([x_1;x_2;x_3;x_4]W_1 + b_1)
$$
First hidden layer: concatenated input × $W_1$ + $b_1$, then ReLU. Standard feed-forward layer.

$$
h_j = \text{ReLU}(h_{j-1} W_j + b_j)
$$
Each subsequent layer: takes previous layer's output, applies another linear transform + ReLU. Repeats for $k=3$ layers total.

$$
y = \text{softmax}(h_k W_k + b_k)
$$
Final layer: one more linear transform, then **softmax** → converts raw numbers into a probability distribution over True/False.
- The network doesn't just output True or False — it outputs a probability for each. ==This is exactly the "output distribution" needed for the cross-entropy training objective used to train R.==

Dimensions
- **Input vectors ∈ $\mathbb{R}^n$** — each object's vector has $n$ dimensions ($n$ is a hyperparameter)
- **Biases ∈ $\mathbb{R}^{4n}$** — the concatenated four-object input is $4n$-dimensional, so hidden layers operate in that space
- **Weights ∈ $\mathbb{R}^{4n \times 4n}$** — each weight matrix maps $4n \to 4n$, keeping hidden layers a fixed width throughout

> [!tip] Where Table 1's |N| comes from
> $4n$ = hidden-layer width = exactly the "N" (target neuron population) from Definition 4 discussions. If $n=4$ → $4n=16$, matching |N|=16 in Table 1. |N|=32 → $n=8$.

Evaluation on held-out data
- "Held-out random vectors unseen during training" — fresh random vectors generated for each object *at test time*, not reused from training. ==Tests whether the network learned the abstract relational rule $(w{=}x)=(y{=}z)$, or just memorized specific training vector patterns.==
- 100% accuracy on unseen vectors confirms genuine learning of the relational structure — important, since DAS analysis should be applied to a network that actually solved the *task*, not one that overfit to training-specific patterns.
---
> [!important]  
Every number in Table 1 answers the same underlying question — **"out of many, many test interventions, what fraction did the network's internal behavior match the symbolic model's prediction?"** — but computed under a different combination of: which layer, how much rotated space you allow, and which hypothesis about what the network represents. High numbers = strong evidence for that hypothesis, in that layer, at that subspace size. Low numbers (near 0.50) = no evidence.

---
Section 4.2 — High-Level Models
- Answers: why does Table 1 test four different hypotheses instead of just the "obvious" one?
The main hypothesis

$$
V_1 = (w{=}x), \quad V_2 = (y{=}z), \quad V_3 = (V_1 = V_2)
$$

> [!note]
> First compute whether the left pair is equal, separately compute whether the right pair is equal, then compare those two results for the final answer. Mirrors how a human would solve this by hand — matches what's hypothesized in the cognitive-science literature the paper cites (Premack, Thompson et al.) for how humans/primates solve hierarchical relational tasks.

This is Table 1's "Both Equality Relations" row-group — the one that hits 1.00 IIA at Layer 1.

Why one hypothesis alone isn't enough?
- Evaluating this high-level model alone is insufficient, as there are obviously many other high-level models of this task."
- Many different internal algorithms could produce the exact same input-output behavior (100% task accuracy). A high IIA for one hypothesis alone doesn't prove the network implements *that specific* algorithm — you need to also test rival hypotheses and show they score worse. Otherwise a high score could be misleadingly easy to get, or not meaningfully diagnostic. This is a ==scientific control==.

The two alternative hypotheses:

> [!question] Alternative 1 — "Left Equality Relation" only
> Network computes just $V_1 = (w{=}x)$ as a genuine intermediate variable; doesn't cleanly represent $V_2$ separately.

> [!question] Alternative 2 — "Identity of First Argument"
> Much more deflationary: no equality relation computed as an intermediate step at all. Network just carries forward the raw identity of object w, defers the actual comparison logic to later (or does it non-modularly, without a clean intermediate "equality" concept).

> [!important] Key point
> All three hypotheses solve the task with 100% behavioral accuracy. **Behaviorally, from the outside, the three internal algorithms are indistinguishable.** The only way to tell them apart is looking inside via interventions — exactly what DAS is for.

Connecting back to Table 1's numbers

| Hypothesis                 | IIA at Layer 1 (N=16, k=8) | What it means                                                                                                                            |
| -------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Both Equality Relations    | 1.00                       | Strong evidence network computes V1, V2 as separate intermediates                                                                        |
| Left Equality Relation     | 0.90                       | V1 alone is also separately findable — expected, since it's part of the full hypothesis; slightly lower than the full two-variable story |
| Identity of First Argument | ~0.50–0.52                 | Essentially no evidence — network doesn't store raw object identity as a clean isolatable variable                                       |

Why the comparison matters?
- "Both Equality Relations" clearly beating "Identity of First Argument" — and matching or exceeding "Left Equality" — is the evidence that the network implements the natural, structured Figure 2 algorithm, not some alternative. If "Identity of First Argument" had scored just as high, that would undercut the whole claim: it would suggest DAS just aligns to *anything*, not something diagnostic of real task structure.

> [!tip]
> Section 4.2 sets up three competing stories for how the network solves hierarchical equality, and explains why testing only the "obvious" one wouldn't be convincing — since the alternatives are behaviorally indistinguishable from the outside. Running DAS against all three and comparing IIA scores turns the high score for "Both Equality Relations" into real evidence for that specific algorithm, not an artifact of DAS being able to align *something* to *anything*.

> [!important]
> "100% behavioral accuracy" only checks whether the final answers are correct; it says nothing about the internal steps that produced them. Interventions are the tool that lets you actually probe those internal steps directly — which is why DAS (built entirely on interventions) can distinguish between hypotheses that ordinary accuracy testing cannot.

---
The "Identity Subspace of Left Equality" Result (also from table 1)

What's being tested?
- DAS already found a subspace aligning with $V_1 = (w{=}x)$ (the "Left Equality Relation" row, IIA 0.85–0.99). Follow-up question: **can that V1-subspace be broken down further** — into a representation of "what is w" and a representation of "what is x," glued together — or is "w=x" genuinely its own thing, not reducible to the identities that go into it?
(This is Step 6 from Section 3.7 — the decomposition test.)

How it's tested?
> [!note] Method
> Freeze the rotation matrix already aligned to V1. Train a **second** rotation matrix on top of it, trying to align a sub-portion of that same subspace with a new hypothesis: "identity of the first argument" (just "what object is w," nothing about equality).
>
> If a piece of the V1-subspace cleanly represents "identity of w" with high IIA → the V1-representation is really just a data structure secretly holding "identity of w" and "identity of x" — equality is superficial, decomposable into simpler parts.

What was found
- Result: very low IIA (~0.49–0.51, chance level)
- No rotated sub-piece of the V1-subspace represents "identity of the first argument" well. The decomposition attempt **fails**.

Why this is a meaningful positive finding?
- Two possible ways a network could represent "w=x":

|                                    | Option A — decomposable ("data structure")                  | Option B — truly abstracted                                     |
| ---------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------- |
| What's stored                      | Identity of w and identity of x, separately, compared later | Just the abstract yes/no relational fact — identities discarded |
| Expected decomposition test result | Succeeds (identity pieces still recoverable)                | Fails (nothing to recover — identities are gone)                |
What the low IIA rules in/out?
- The very low decomposition IIA rules out Option A and supports Option B: whatever distinguished "w is A" from "w is B" appears to be thrown away once equality is computed — only the abstract relational fact survives in the V1-subspace.

> [!tip] Key takeaway
> DAS doesn't just reveal *whether* a concept is represented — it can reveal *how*: as a truly abstracted symbolic quantity, or as a disguised container for simpler underlying pieces.

---
Section 4.4
- This section is a pure control experiment: instead of the trained hierarchical-equality network, DAS is applied to a <mark style="background: #D2B3FFA6;">randomly initialized, untrained version of the same architecture </mark>(which solves the task at chance level) across a range of hidden-layer sizes. At sizes comparable to the real experiments, IIA stays at chance — confirming DAS isn't just fabricating alignments out of nothing. <mark style="background: #FF5582A6;">But at much larger hidden sizes, IIA creeps upward — revealing that with enough raw search capacity, DAS's gradient-based optimization can start to manufacture the _appearance_ of structure even where none genuinely exists, a caveat the authors flag explicitly as a real limitation of the method.</mark>
> [!important]  
> This result serves as an important calibration check: DAS's high IIA scores on the _actual trained_ networks (Table 1, Table 2) can't be dismissed as "DAS would find high IIA in literally any network of that size" — because Section 4.4 shows that at comparable, reasonably-sized hidden dimensions, a random network gives you chance-level IIA, not inflated results. The paper's positive findings elsewhere are therefore more trustworthy — they're not an artifact of DAS's search flexibility alone.

At the same time, this section is an honest caveat about the method's limits: ==if you're going to apply DAS to something with an enormous hidden dimension (which is increasingly the case for large real-world models), you need to be careful, because DAS's optimization _can_ start to fabricate spurious structure purely from having enough parameters to search through — a genuine limitation to keep in mind, not just a clean positive result.==

---
