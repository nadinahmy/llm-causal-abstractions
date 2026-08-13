
---
- DAS ---> allows individual neurons to play multiple distinct roles by analyzing representations in non-standard bases—==distributed representations==
- Causal abstraction requires a ==computationally intensive brute-force search== process to find optimal alignments between the variables in the high-level model and the states of the low-level one.
- “==Localist==: they artificially limit the space of possible alignments by presupposing that high-level causal variables will be aligned with disjoint groups of neurons.”
- ==Distributed neural representations== ---> “patterns” consisting of linear combinations of unit vectors.
- <mark style="background: #FF5582A6;">The key insight is that viewing a neural representation through an alternative basis that is not aligned with individual neurons can reveal interpretable dimensions.</mark>

### DAS: Symbolic Abstraction vs. Data Structure
- **Core idea:** Distributed Alignment Search (DAS) doesn't just locate information in a model — it reveals _how_ the model represents it. Perfect causal test scores (IIA) can mask two very different internal logics.
#### 1. Hierarchical Equality → True Symbolic Abstraction
- Task: judge if two object pairs share the same relation (e.g., is left-pair equality = right-pair equality?)
- DAS found a distributed alignment with **100% IIA**
- Key finding: intermediate representations **cannot be decomposed** back into the original objects (w, x, y, z) — ==the model "forgets" the objects and keeps only the pure equality relation==
- → Model implements a genuine **symbolic, tree-structured algorithm**
#### 2. Monotonicity NLI → "Data Structure" Packet
- Task: BERT determines lexical entailment (e.g., "dog" ⟹ "mammal")
- DAS also found **100% IIA** for a "lexical entailment" variable
- Key finding: this representation **can be decomposed** back into the original word identities
- → The subspace isn't computing an abstract relation — ==it's a **container packaging raw data** for a later layer to process==

> Same IIA score, opposite internal logic. Decomposability is the tell.

|Feature|Hierarchical Equality|Monotonicity NLI|
|---|---|---|
|**IIA Score**|100% (Perfect)|100% (Perfect)|
|**Internal Logic**|True abstraction — pure relation|Data structure — raw data passed through|
|**Decomposable?**|No — inputs unrecoverable|Yes — original words still visible|
|**Conclusion**|Symbolic algorithm|Causal "packet" of info|

---

- For each variable in the high-level model, τ tells you ==how to _read off_ its value from the corresponding variable(s) in the low-level model.== It's the translation dictionary between "network activations" and "symbolic variable values."
- V1 and V2 are the _intermediate_ high-level variables — in the causal story, V1 and V2 are supposed to represent "value of the first argument" and "value of the second argument" internally, before they get combined into V3 = V1 ∧ V2. 
	- Note : Nothing forces the network to represent these intermediate quantities in any particular hidden unit or direction. Unlike the input/output case, there's no definitional shortcut: you don't know a priori _which_ neurons or which linear combination of neurons correspond to "the value of the first argument" inside the network.
	- τ_V1 and τ_V2 — the mappings from hidden-layer activations to the symbolic values of V1 and V2 — have to be **found empirically**: this is precisely the alignment-search problem the rest of the paper is about.
---
### Definition 1
$$τ(L_{I←i}(x))=H_{τ(I←i)}(τ(x))$$

> [!note] In plain words  
>  Take any low-level input $x$, and any low-level intervention $I←i$ (force some low-level variables to specific values). There are two ways to get from that intervention to a high-level answer:
> - **Path A:** intervene at the low level → run the network → translate the _result_ up to high-level symbols with $τ$
> - **Path B:** translate the intervention itself up to the high-level model → run the high-level model with that intervention
> 
> $H$ is a faithful abstraction of $L$ (under alignment $Π$) only if **Path A and Path B always agree.**

This is a strong requirement: it's not just "does the network get the right answer normally" — it's "<mark style="background: #FF5582A6;">does the network's internal structure behave the way the symbolic model says it should, even under artificial interventions.</mark>"

---
## Definition 2 — Interchange Interventions

> [!note] Core idea  
> Take a value computed from one input, and force the model to use that value while processing a different input.

### The Setup
- Start with a model M (could be the high-level symbolic model or the low-level network)
- Pick k **source inputs** ${s_1,…,s_k}$ — extra inputs you run the model on, purely to extract values from
- Pick **non-overlapping sets of variables** ${X_1,…,X_k}$ — the "target" variables you're going to intervene on, one set per source
### The Formula

$$
II(M, \{s_j\}_{j=1}^{k}, \{X_j\}_{j=1}^{k}) = M_{\Lambda}
$$
Where $\bigwedge$ is :
$$
\Lambda = \bigwedge_{j=1}^{k} \left( X_j \leftarrow \operatorname{GetVals}_{X_j}(M(s_j)) \right)
$$

**In words:** Λ is the ==combined intervention== — for every j from 1 to k, set $X_j$​ to whatever value it took when you ran M on source input $s_j$​. Then $M_Λ$​ means "run M but with intervention Λ applied."

> [!tip] Reading the right-hand side  
>  "Run M on each source input $s_j$​, read off whatever values the variables in $X_j$​ take on during that run, then go back and force those exact values onto $X_j$​ while processing some _other_ (base) input."

### Step by Step

1. **Run** $M(s_j)$ — process source input $s_j​$ normally
2. **Extract** $GetVals_{X_j}(M(s_j))$ — look at what values the variables $X_j$​ took during that run
3. **Intervene**: $X_j ← …$ — hard-code those values into variables $X_j$​
4. **Combine** (the ⋀): do this simultaneously for several different groups of variables, each sourced from a different input
5. **Result**: a new model — the "interchange-intervened" model — which you can feed a base input into. Everywhere except the intervened variables, it behaves normally on the base input; at the intervened variables, it's forced to use values borrowed from the sources

> [!important]  
> This implements a counterfactual question: *"What would the model output on this base input, if the internal variable X had instead taken on the value it takes when processing a totally different input?"*

<mark style="background: #FFB86CA6;">If X really represents some meaningful causal quantity, swapping it this way should change the output in a predictable, interpretable way</mark> — matching what the high-level symbolic model predicts under the analogous swap. That match rate becomes **IIA (interchange intervention accuracy)**, the paper's main evaluation metric.

---
#### Section 3.3 — Constructive Causal Abstraction
- The setup for the worked example

| | Boolean model $B$ | Neural network $N$ |
|---|---|---|
| Inputs | $P, Q \in \{T, F\}$ | $x_1, x_2 \in \{0,1\}$ (0=F, 1=T) |
| Intermediate | $V_1 = P,\ V_2 = Q$ | $H_1 = [x_1,x_2]W_1,\ H_2 = [x_1,x_2]W_2$ |
| Output | $V_3 = V_1 \wedge V_2$ | $O = [H_1,H_2]w + b$, predict T iff $O>0$ |

- The proposed alignment $\Pi$ (the "dashed lines" in the paper's diagram) pairs $V_1 \leftrightarrow H_1$ and $V_2 \leftrightarrow H_2$ — i.e., it guesses that the first hidden unit represents the first argument, and the second hidden unit represents the second argument.
The interchange intervention being tested
- **Base input:** $[F,T]$ → i.e. $x=[0,1]$ for $N$
- **Source input:** $[T,T]$ → i.e. $x=[1,1]$ for $N$
We're testing: "If I take the $V_1$-value from the source and force it onto the base, do $B$ and $N$ agree on the consequence?"

$$
II\big(B, \{[T,T]\}, \{\{V_1\}\}\big) \quad \text{vs.} \quad II\big(N, \{[1,1]\}, \{\{H_1\}\}\big)
$$
Running it through B (the easy, symbolic side)

> [!success] High-level computation
> - Source $[T,T]$ → $B$ gives $V_1 = T$
> - Base $[F,T]$, but with $V_1$ forced to $T$ (borrowed from source), $V_2$ computed normally from base ($Q=T \Rightarrow V_2=T$)
> - Result: $V_3 = V_1 \wedge V_2 = T \wedge T = T$

Symbolic logic ---> deterministic and easy to compute by hand

#### Running it through N (the numeric side)

The weight matrices given are just a 20° rotation:

$$
W_1 = \begin{bmatrix}\cos 20° \\ -\sin 20°\end{bmatrix}, \quad
W_2 = \begin{bmatrix}\sin 20° \\ \cos 20°\end{bmatrix}, \quad
w = \begin{bmatrix}1 \\ 1\end{bmatrix}, \quad
b = -1.8
$$

**Step 1 — get the intervention value from the source [1,1]:**

$$
H_1(\text{source}) = 1\cdot\cos 20° - 1\cdot\sin 20° \approx 0.9397 - 0.3420 = 0.60
$$

**Step 2 — apply it to the base [0,1]**, forcing $H_1 = 0.60$, but letting $H_2$ compute normally from the base (since only $H_1$ was targeted):

$$
H_2(\text{base}) = 0\cdot\sin 20° + 1\cdot\cos 20° \approx 0.94
$$

**Step 3 — compute the output:**

$$
O = H_1\cdot 1 + H_2\cdot 1 + b = 0.60 + 0.94 - 1.8 = -0.26
$$

Since $O < 0$, the network predicts F.

#### The mismatch —> this is the whole point:

| Model | Prediction after interchange intervention |
|---|---|
| $B$ (symbolic) | T |
| $N$ (neural, under alignment $V_1 \leftrightarrow H_1$) | F |

> [!warning] Counterexample to Definition 1
> The two paths disagree. Even though $N$ gets 100% behavioral accuracy on ordinary inputs (it never gets a normal prediction wrong), it fails the interventional test: forcing $H_1$ to imitate "$V_1$ from the source" does not produce the same downstream effect as forcing $V_1$ itself in the symbolic model.
>
> Conclusion: under this alignment (standard neuron-aligned basis, $V_1 \leftrightarrow H_1$, $V_2 \leftrightarrow H_2$), B is not a constructive causal abstraction of N — despite N solving the task perfectly.

- This negative result is the paper's entire motivation.
- Two questions naturally follow:
	1. Is this alignment just wrong? Maybe $V_1$ and $V_2$ aren't individually represented by single neurons $H_1, H_2$ — maybe they ==live in some rotated combination of both==.
	2. Was picking the "standard basis" (individual neurons) an unjustified assumption?
> [!important] This is the core motivation for DAS
> The information that "should" correspond to $V_1$ and $V_2$ was there in the network all along — it just wasn't aligned with individual neurons, only with a rotated direction in activation space. A brute-force / localist search would have concluded "no abstraction exists" and been wrong. This is exactly the failure mode DAS is designed to avoid.

---
#### Figure 1 — A Generic Multi-Source Distributed Interchange Intervention

- Section 3.3's example only ever intervened on **one** target variable at a time, using **one** source input, and used the **standard basis** (individual neurons). Figure 1 is the visual companion to Definition 3 (Distributed Interchange Interventions) — it generalizes in three ways simultaneously:
1. **Multiple sources** — not just one source input, but $k$ of them (the figure shows $k=2$)
2. **Multiple target variables** — several high-level variables can be aligned to different subspaces at once
3. **Non-standard basis** — instead of picking individual neurons, we rotate into a new coordinate system first

> [!note] The three colored settings
> - **Middle (red)**: the *base* input run through the model — this is the input whose behavior you ultimately want a counterfactual answer for
> - **Top-left (green)**: a *source* input, run through the model independently, purely to harvest a value
> - **Top-right (blue)**: a second *source* input, run through the model independently, to harvest a different value
>
> All three are complete, ordinary forward passes of the model — nothing unusual happens yet. Each one produces a full "total setting" (values for every variable in the model), labeled $X_1, X_2, X_3$ in the figure (three example hidden units/dimensions).

1. Step 1 — Rotate into a new basis

Each of the three total settings has its 3 hidden units $(X_1, X_2, X_3)$ passed through the **same** orthogonal matrix $R$:

$$
R : X \to Y
$$

This produces new coordinates $(Y_1, Y_2, Y_3)$ for all three settings (green, red, blue alike). Instead of asking "does neuron $H_1$ mean something," we first rotate the whole representation and ask the question in the ==*rotated* coordinates==. Geometrically, $R$ doesn't stretch or distort anything — it's a pure rotation (orthogonal ⇒ preserves lengths and angles) — it just ==looks at the same point in space from a different angle==.

2. Step 2 — Carve the rotated space into subspaces and intervene

> [!important] The orthogonal decomposition
> The rotated space $Y$ is split into subspaces $Y_0, Y_1, \dots, Y_k$ that don't overlap and together cover the whole space:
> $$
> Y = Y_0 \oplus Y_1 \oplus \cdots \oplus Y_k
> $$

In the figure, each subspace is one axis (dimension) of $(Y_1, Y_2, Y_3)$ — but in general a subspace can be several dimensions bundled together (that's what lets DAS assign a *group* of rotated dimensions to a single high-level variable).

Now the intervention happens **in this rotated space**, on the **base (red)** setting:
- The $Y_3$ portion of the base's own rotated representation is **kept** (this is why it's called a *soft* intervention — part of the base input's own identity survives)
- The $Y_1$ portion is **overwritten** with the rotated value taken from source 1 (green)
- The $Y_2$ portion is **overwritten** with the rotated value taken from source 2 (blue)

This is exactly Definition 3's formula:

$$
F_N^{*}(v) = R^{-1}\left(\text{Proj}_{Y_0}\big(R(F_N(v))\big) + \sum_{j=1}^{k}\text{Proj}_{Y_j}\big(R(F_N(M(s_j)))\big)\right)
$$

Reading it left to right:
- $F_N(v)$: the base setting's original hidden values
- $R(\cdot)$: rotate them into $Y$-space
- $\text{Proj}_{Y_0}(\cdot)$: keep only the base's own $Y_0$ slice
- $+\sum_j \text{Proj}_{Y_j}(R(F_N(M(s_j))))$: add in the $Y_j$ slice taken from each source $s_j$'s rotated representation
- $R^{-1}(\cdot)$: rotate the recombined vector back to the original (neuron) basis


3. Step 3 — Un-rotate and read off the counterfactual output
Once the mixed representation (base's $Y_0$ + source 1's $Y_1$ + source 2's $Y_2$) is assembled in rotated space, it's transformed back with $R^{-1}$ into the network's normal coordinate system. The model then continues its forward pass from there as usual, producing a new — counterfactual — output for the base input.

What makes this "distributed"

> [!tip] The core conceptual shift
> A *localist* intervention picks specific neurons and swaps their raw values. A *distributed* intervention picks specific **directions** (linear combinations of neurons) in a rotated space and swaps values along those directions instead. Because $R$ can mix all the neurons together, a single "direction" $Y_1$ might correspond to some weighted combination of every original neuron — this is precisely the kind of structure the boolean-conjunction example needed (the −20° rotation) and which no amount of neuron-by-neuron search could ever find.

## Connecting to DAS
> "In DAS, the orthogonal matrix is found with gradient descent using a high-level causal model to guide the search process."

That is: $R$ is not hand-picked (as it was in the −20° toy example, where the authors *knew* the answer in advance). Instead, DAS **learns** $R$ by training it with gradient descent so that, after the distributed intervention, the network's counterfactual output matches what the high-level causal model $H$ predicts as often as possible.

## Summary — the pipeline in one line

$$
\text{(base + sources)} \;\xrightarrow{\;R\;}\; \text{rotate} \;\xrightarrow{\;\text{swap subspaces}\;}\; \text{mix} \;\xrightarrow{\;R^{-1}\;}\; \text{unrotate} \;\rightarrow\; \text{new counterfactual output}
$$

Figure 1 is showing you this entire pipeline for the general case ($k$ sources, arbitrary subspace sizes) — while the −20° worked example in Section 3.4's text is the same pipeline specialized to $k=1$ source and 1-dimensional subspaces, worked out with actual numbers.

---
### Definition 3 — Distributed Interchange Interventions

#### The problem it's solving

Definition 2 (regular interchange interventions) only let you swap in values for specific *neurons* — the "standard basis." But as the −20° rotation example showed, ==sometimes the meaningful structure isn't sitting in individual neurons at all — it's sitting in some rotated combination of them.== Definition 3 generalizes interchange interventions so you can intervene on *any* direction in activation space, not just the neuron axes.

Setting the stage, start with:
- A model $M$, with some source inputs we'll pull values from
- A set of "target" variables $N$ inside the model — these are the raw neurons we're interested in (e.g., a whole hidden layer)
- A way to rotate those neurons into a new coordinate system — call it $R$.
	- We can think of $R$ as a pair of glasses that lets us look at the same hidden layer from a different angle. It doesn't change what information is there, just how it's organized into axes.
- Once we're in this new rotated space, we carve it up into several non-overlapping chunks (subspaces) — some chunks for each source you're pulling from, ==plus one "leftover" chunk that isn't touched at all.==
## What the intervention actually does, step by step

> [!note] The procedure
> 1. **Take the base input's hidden representation** — this is the input whose behavior you eventually want a counterfactual answer for.
> 2. **Rotate it** into the new coordinate system using $R$.
> 3. **Keep one chunk of it untouched** (the "leftover" subspace, $Y_0$). This is what makes it a *soft* intervention rather than a hard one — you're not wiping out the base input's identity entirely, just part of it. This matters because it preserves some causal connection between the original input and the output, rather than fully replacing the base with a Frankenstein mix of sources.
> 4. **For every other chunk, throw away the base's value and substitute in a value pulled from a source input.** Each source input gets its own dedicated chunk. To get that value: run the model on the source input, take its hidden representation, rotate it the same way with $R$, and grab just the portion in that chunk.
> 5. **Reassemble everything** — the base's leftover chunk, plus each source's substituted chunk — back into one vector, still in the rotated coordinate system.
> 6. **Rotate back** to the original neuron coordinates using the inverse of $R$.
> 7. **Let the rest of the network compute forward as normal** from this doctored hidden representation. Whatever comes out is the counterfactual output.

Why this is more powerful than Definition 2?

> [!tip] Definition 2 is a special case
> Ordinary interchange interventions (Definition 2) are really just a special case of this: if $R$ is chosen to be the "do-nothing" rotation (the standard neuron basis) and the chunks are just individual neurons, you get exactly the old localist intervention. What Definition 3 adds is the freedom to choose $R$ to be *any* rotation — so you can search for a coordinate system where the concept you're looking for (like "the value of the first argument") lives cleanly in its own chunk, even if it's smeared across every neuron in the standard basis.

> [!important] Geometric picture
> Imagine the hidden layer's activity as a cloud of points in space. In the standard basis, "meaningful" directions might run diagonally through that cloud — invisible if you only ever look along the neuron axes. A distributed intervention first **tilts your entire view** (rotates the cloud) so that the meaningful direction lines up with one of your new axes, **swaps values only along that axis**, and then **tilts back**. The model never "feels" anything strange happen — from its own numeric perspective, it's just getting a modified but still valid hidden vector to keep computing with.

- This is exactly the machinery that later lets DAS learn $R$ via gradient descent, instead of the having to hand-pick the −20° rotation by inspection, as in the toy example before.
---
The Worked Numerical Example — Fixing the Section 3.3 Mismatch

- This example takes the exact same base/source pair that failed in Section 3.3 (predicted F instead of T) and reruns it — but this time using a distributed intervention instead of a plain neuron-swap.

The two runs we're combining:

| | Input | H1 | H2 |
|---|---|---|---|
| **Base** (yellow, bottom-left) | [0,1] (i.e. [F,T]) | -0.34 | 0.94 |
| **Source** (blue, bottom-right) | [1,1] (i.e. [T,T]) | 0.6 | 1.28 |

- These are exactly the two runs from Section 3.3

Step 1 — Rotate both into Y-space

> [!note] Why rotate?
> In Section 3.3, we intervened directly on H1 (a single neuron) and it failed. Here, instead of touching H1 or H2 individually, we first look at the hidden layer through rotated glasses — apply a rotation of -20° to both the base and the source representations.

**Rotating the base (-0.34, 0.94):**

Using cos(20°) ≈ 0.94 and sin(20°) ≈ 0.34:

- Y1(base) ≈ cos(20°)×(-0.34) + sin(20°)×(0.94) ≈ -0.32 + 0.32 = **0.0**
- Y2(base) ≈ -sin(20°)×(-0.34) + cos(20°)×(0.94) ≈ 0.12 + 0.88 = **1.0**

→ This is the **yellow pair of circles** in the middle of the figure: (0.0, 1.0).

**Rotating the source (0.6, 1.28):**

- Y1(source) ≈ cos(20°)×(0.6) + sin(20°)×(1.28) ≈ 0.56 + 0.44 = **1.0**
- Y2(source) ≈ -sin(20°)×(0.6) + cos(20°)×(1.28) ≈ -0.21 + 1.20 = **1.0**

→ This is the **pink/red pair of circles**: (1.0, 1.0).

> [!tip] Sanity check
> Notice these rotated numbers are suspiciously clean — (0.0, 1.0) and (1.0, 1.0). That's not a coincidence: this is exactly the point of the -20° rotation. Once you're in the right coordinate system, Y1 behaves exactly like "value of the first argument" (0 = F, 1 = T) and Y2 behaves exactly like "value of the second argument." The messy decimals (-0.34, 0.94, etc.) were just what those clean boolean values look like *before* undoing the network's internal 20° rotation.

Step 2 — Do the intervention, but in Y-space

Instead of swapping raw neurons, we now swap along the rotated axis Y1 only (this is the low-level counterpart of intervening on V1 in the symbolic model):

- **Keep** the base's own Y2 value: 1.0 (unchanged — this is the "soft intervention" part of Definition 3, meaning "everything not targeted stays as the base had it")
- **Overwrite** Y1 with the value taken from the source: 1.0 (replacing the base's own 0.0)

Combined intervened vector in Y-space: **(1.0, 1.0)**

This is exactly what Definition 3 computes — the middle circles in the figure, now merged into one vector, get fed forward into the next step.

Step 3 — Rotate back to the original neuron basis

Now undo the rotation — apply a rotation of +20° (the inverse):

- H1' ≈ cos(20°)×(1.0) - sin(20°)×(1.0) ≈ 0.94 - 0.34 = **0.6**
- H2' ≈ sin(20°)×(1.0) + cos(20°)×(1.0) ≈ 0.34 + 0.94 = **1.28**

→ This matches the **top box** in the figure: H1 = 0.6, H2 = 1.28.

Step 4 — Compute the final output

Output = H1' + H2' + b = 0.6 + 1.28 - 1.8 = **0.08**

Since the output is greater than 0, the network now predicts **T**.

> [!success] Now it matches!
> | Model | Prediction |
> |---|---|
> | B (symbolic, intervening on V1) | T |
> | N (distributed intervention on Y1) | T ✅ |
>
> Compare to Section 3.3, where the same base and source, under a plain intervention on raw neuron H1, gave an output of -0.26 → predicted F — a mismatch. Here, after rotating into the right basis first, the exact same information produces the correct counterfactual output. Nothing about the network changed; only the coordinate system we intervened in changed.

## Reading the full figure left to right

1. **Bottom-left (yellow box):** base input [0,1] → raw hidden values H1 = -0.34, H2 = 0.94
2. **Bracket next to it:** apply the -20° rotation → rotates the base into Y-space
3. **Middle yellow circles (0.0, 1.0):** the base's rotated representation
4. **Middle pink circles (1.0, 1.0):** the source's rotated representation (from input [1,1], with H1 = 0.6, H2 = 1.28 rotated the same way)
5. **Bracket after the circles:** apply the +20° rotation (the inverse) to the intervened vector — combining the base's kept Y2 = 1.0 with the source's substituted Y1 = 1.0
6. **Top box:** the result after rotating back — H1 = 0.6, H2 = 1.28 — feeding into output O = 0.08, which is greater than 0, so the network predicts T

## The one-sentence takeaway

> [!important]
> The network was never "wrong" about the boolean conjunction task — it just stored the two argument values along a rotated pair of directions instead of along the two raw neuron axes. A distributed intervention finds and swaps values along the *correct* directions, and once you do that, the network's internal counterfactual behavior lines up exactly with the symbolic model's — something a plain neuron-by-neuron intervention could never reveal.