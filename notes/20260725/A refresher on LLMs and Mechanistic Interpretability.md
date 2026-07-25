1. Mechanistic Interpretability for AI Alignment | Callum McDougall, Joseph Bloom
---
- Mechanistic Interpretability ---> Reverse engineering a program's source code from its compiled binary
	- Making systems less black-boxy
	- Mechanistic here means a higher bar for explanations
	- Take this messy code and roll it back to the human interpretable algorithm that this code is implementing
	- most danger of current deep learning is that we don't really understand how the systems we are training are getting to the answers that they are
	- identifying concrete mechanism that leads to this answer
	- we can do things like intervene on half way through the mechanism and predict the effect that the intervention will have on the output
- A lot of the inference that transforms do can be broken down into the discovery of specific circuits inside the model which are like implementing certain algorithms and solving certain tasks
![[IMG-20260725210929159.png|481]]
==Case Study: Indirect Object Identification & ACDC==
Context: Indirect Object Identification (IOI) in GPT-2
What ACDC (Automated Circuit DisCovery) does : 
1. Choose computational graph, task, threshold τ. Start with the full graph of model components (attention heads / connections) that could feed into the output, given an input.
2. At each component, prune unimportant connections. Sever a connection, check whether task performance drops by more than τ. If not → cut it.
3. Recurse until the full circuit is recovered. Repeat pruning across the graph. What's left is the minimal sub-circuit that's actually responsible for the behavior.

How this relates to causal abstraction (in JMLR paper)
- This is a concrete instance of **sub-circuit analysis** (JMLR section 3.5.2) — a special case of abstraction via **ablation**.
- Formally: define a tiny high-level model with input X, output Y, and one binary variable Z = "was the circuit intact or severed." ==ACDC is searching for the minimal circuit C such that this high-level model is a faithful abstraction (in the Def. 25 / exact-transformation sense) of the full low-level network.==

How does this relate to me / my work?
- ACDC / circuit discovery = **structural** abstraction → _which components/connections are responsible for a behavior?_
- My question (and DAS) = **feature-level** abstraction → _does a subspace of activations faithfully represent a specific semantic concept?_
- Different "granularity", not competing approaches — could be complementary: <mark style="background: #ABF7F7A6;">circuit discovery narrows down where to look; a DAS like method then tests what is represented there.</mark>

Follow-up / open questions
- <mark style="background: #FF5582A6;">Would a circuit-discovery step (ACDC) be a useful precursor to my method, to narrow the search space before testing concept abstraction?</mark>
- Read Wang et al. (2023) IOI paper if time allows — original source of this example

==If you understand the way that a transformer is representing certain information, then you can essentially change it in a specific way and get the outcome you are trying to get==

Second Case Study: Othello-GPT — Emergent World Models
The Setup used:
- Li et al. (2023) trained a GPT-style model purely on <mark style="background: #BBFABBA6;">sequences of Othello moves</mark> — no board, no rules, just "predict the next legal move."
- Despite zero explicit supervision on board structure, the model internally learned to track the full board state — an <mark style="background: #ABF7F7A6;">**emergent world model**</mark>
What was the twist? : linear vs. non-linear representation
- **Li et al. (original):** tried linear probes to read out board state (black/white/empty per cell) → poor performance, >20% error. Non-linear probes (2-layer MLPs) did much better (down to ~1.7–4.6% error). Looked like evidence _against_ the linear representation hypothesis.
- **Nanda (2023), re-examination:** the representation _was_ linear all along — the concept was just mis-specified. Instead of "this cell is black," the model represents **"this cell is my color"** (relative to whichever player is currently moving, since the model plays both black and white).
- Reframed this way, linear probes hit **>99% accuracy**.
- <mark style="background: #FF5582A6;">Crucially, this wasn't just correlational: intervening on the probe direction causally changed the model's predicted next move → real causal role, not just a readable correlate.</mark>

Why this matters for my thesis direction
This is essentially my proposed pipeline, already run as a real case study:

| My pipeline step          | Othello-GPT equivalent                         |
| ------------------------- | ---------------------------------------------- |
| Represent a concept       | Board state (per-cell occupancy)               |
| Learn abstraction         | Linear probe                                   |
| Intervene                 | Steer the probe direction                      |
| Test commutativity        | Check next-move prediction changes as expected |
| Measure abstraction error | Probe accuracy (poor → good after reframing)   |

**Key lesson:** the first attempt "failed" not because linearity was wrong, but because the **high-level variable itself was mis-specified** (absolute color vs. player-relative color). <mark style="background: #FF5582A6;">A bad abstraction-error result can mean two very different things:
1. No faithful abstraction exists for this concept, or
2. The concept was defined wrong, and a faithful abstraction does exist under a better specification.
</mark>
### Follow-up / open questions

- Read Li et al. (2023) and Nanda (2023) blog post directly for full method details
- Consider: does my concept of interest risk the same "wrong specification" trap as absolute-color did here?

#### ==AI Alignment ---> making AI systems behave as intended, safely==
![[IMG-20260725215347243.png|408]]
- the harder alignment turns out to be, the more essential it becomes to have ==interpretability tools that look _inside_ the model rather than just judging it by its outputs== — because in the "moderate" and "hard" scenarios, a model's behavior alone can't be trusted to reveal what it's actually doing or "believes."
---
JMLR section 3.1 — Polysemantic Neurons, the Linear Representation Hypothesis, and Modular Features via Intervention Algebras
The core problem
- How should we decompose a black-box deep learning system into meaningful parts? Candidates: individual activations, directions in activation space, or whole model components.
- Would be simplest if single neurons were the right unit of analysis — but neurons are well known to be **==polysemantic==**: a single neuron often participates in representing _multiple_ unrelated high-level concepts, not just one.
- → individual neural activations are **not sufficient** as the unit of analysis for interpretability.
The linear representation hypothesis:
- ==Simplest fix for polysemanticity: apply some **rotation** to the activations so that, in the _new_ coordinate system, each dimension is monosemantic (represents one concept cleanly).==
- The LRH claims linear representations (directions in activation space) are sufficient to capture the complex, non-linear internal structure of deep learning models.
- Geiger et al. are explicitly **skeptical of baking this in as an assumption** — they don't want the theoretical framework to presuppose linearity. Instead:
    - Whether a given decomposition into modular features is _useful_ should be treated as an **empirical hypothesis**, testable/falsifiable via experiment — not an assumed prior.
Formalizing "modular features" via intervention algebras
- Their proposed general definition: a set of **modular features** = any set of variables that form an **intervention algebra**, accessed via a **bijective translation** (from section 2.3.1 — a lossless change of coordinates, e.g. a rotation matrix).
- Intervention algebras (sec 2.2) formalize "separable components with distinct mechanisms" — satisfying:
    - **Commutativity** (order doesn't matter when intervening on different variables)
    - **Left-annihilativity** (a second intervention on the same variable overwrites the first)
- Individual activations, orthogonal directions in vector space, and model components (e.g. attention heads) can all count as modular features under this definition, **regardless of whether they're linear or not**.
- If LRH is correct → rotation matrices (linear bijective translations) are sufficient to find these features.
- If LRH is _not_ correct in some case → non-linear bijective translations may be needed to uncover features that aren't linearly accessible (cited example: "onion" representations found in simple RNNs by Csordás et al. 2024).
- Net effect: the framework stays **agnostic** about linearity — it doesn't need LRH to be true in order to define modular features, but LRH becomes a special, testable case within it.

Relevance of this section for Thesis:
- This section is the direct **theoretical counterpart** to the Othello-GPT case study: Li et al.'s original non-linear probes vs. Nanda's later linear reframing is _exactly_ an empirical test of whether LRH held for that concept — first it looked false, then it turned out true once the concept was correctly specified.
- Relevance for my research question: I'm implicitly assuming/testing something like the LRH for whatever concept I pick. I know this is an **assumption to test, not a given** — and that a "negative" result (no faithful linear abstraction found) could mean either (a) no faithful abstraction exists, or (b) a non-linear bijective translation is needed instead
- Ties back to modular features = intervention algebra + bijective translation → this is the formal vocabulary for what my "learn abstraction" step is actually trying to construct.
==Open question==
- ==If my method assumes linearity (DAS/Abs-LiNGAM) and abstraction error comes out high, how would I distinguish "no abstraction exists" from "need a non-linear bijective translation"?==
---
