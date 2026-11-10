
---
Why I read this
> It reports the **non-linear DAS featurizer** (Hirlimann et al., 2025) that **beats DAS on RAVEL**, the same benchmark HyperDAS uses. This is direct evidence for the "non-linearity" part of my hybrid, and it comes with the warning I have to address (Sutter et al.).

---
Main idea:

- **What it is:** an **overview paper written by the organisers**. It summarises a community competition built on the **MIB benchmark** (Mueller et al., 2025, ICML).
- **Two tracks:**
  - **Circuit localization** (§3): *which components and connections* (the wiring) the model needs for a task → not relevant to me
  - **Causal variable localization** (§4): *where a concept is stored* and whether swapping it works → my topic
- **Submissions:** The causal variable track had only **1 team with 2 methods**, both extending DAS.
- **Headline:** non-linear projections give "significant gains" over DAS, **but** more controls are needed to check the featurizer ==🟡isn't inventing the variable itself==.
---
Section 2: Data and Models (shared setup)

Tasks

| Task | What the model must do | Causal-variable track? |
|---|---|---|
| IOI (Indirect Object Identification) | "When Mary and John went to the store, John gave an apple to ___" → Mary | ✅ |
| Arithmetic (+ / −) | two operands, up to two digits; e.g. "What is the sum of 13 and 25?" | ✅ |
| MCQA | 4-choice questions about object colours; answer is in the prompt | ✅ |
| ARC (Easy / Challenge) | grade-school science multiple choice | ✅ |
| **RAVEL** | city attributes: **country, continent, language**; e.g. "Paris is on the continent of" → Europe | ✅ **only** in this track |

- Splits: train / validation / public test (on HuggingFace) + a **private test** set (only via the leaderboard).
- RAVEL here: 100,000 train / 16,000 validation / 1,000 test.
- **Models:** Llama-3.1 8B, Gemma-2 2B, Qwen-2.5 0.5B, GPT-2 Small. Only model/task pairs with **≥ 75% accuracy** are used. (Same idea as my `filter_for_model.py`: only test what the model actually knows.)

Section 2.2: Counterfactual inputs: the key sentence

> "In the causal variable localization track, activation patching is used to precisely manipulate specific concepts. ==🟢Success is achieved in this setting when a variable in a causal model is a faithful summary of the role a model component plays in input-output behavior==—i.e., interventions on the variable have the same effect as interventions on the model component."

In plain words: you've found a concept when **changing that spot in the network has the same effect as changing the variable in the simple "story" of the task.**

Worked example (RAVEL)
- **Causal model (the story):** "to answer *Which country is Paris in?*, look up the variable *city → country*."
- **Model component:** e.g. a subspace at layer 15.
- **Test:**
    1. In the story: set the variable from *France* to *Japan* → the answer should be "Japan".
    2. In the network: **patch** that spot with the activation from a Tokyo prompt → does it also say "Japan"?
- Same result both ways → the spot really holds "country" ✅

| Term | Simply |
|---|---|
| Causal variable localization | finding *where* a concept is stored |
| Activation patching | copying an activation from one input into another (the DAS swap) |
| Faithful summary | the simple story predicts what the network actually does |

Link to my work ---> This is exactly what **IIA** measures, and what DAS / HyperDAS do on RAVEL.

---
Section 4 : Causal Variable Localization Track

The framework (featurizer view)
- Any hidden vector $h \in \mathbb{R}^d$ can be mapped into a feature space $F_k$ by an **invertible** function $F: \mathbb{R}^d \to F_k$ (e.g. ==🔴multiplying by an orthogonal matrix = DAS's rotation==).
- **Features** $\Pi$ = a set of indices, i.e. which dimensions of $F_k$ hold the variable.
- This covers neurons, orthogonal directions (DAS), SAE features, and **non-linear features**.

 Metric: faithfulness (= IIA)
- Given base $b$, counterfactual $c$, high-level causal model $H$, variable $X$:
    - **High level:** run $H$ on $b$, but fix $X$ to its value under $c$
    - **Low level:** run the network $N$ on $b$, but fix the features $\Pi_X$ (after $F$) to their values under $c$
- ==🔵**Faithfulness** = the fraction of examples where both give the same expected output.==

Section 4.2 : Submission procedure
Teams had to upload:
1. a trained featurizer $F$
2. a trained **inverse** featurizer $F^{-1}$
3. the indices $\Pi$ of the dimensions holding the variable [Subspace Dimension, exactly what I want to automate in HyperDAS!!!]
4. code to save/load the featurizer (if not a built-in type), and optionally a script for dynamic token positions

Important note from the paper : The evaluation **requires an invertible featurizer** ($F$ and $F^{-1}$). An MLP with tanh (below) is **not** built to be invertible. That is one reason the "only applied during training" sentence matters.

---
Section 4.3: Submissions

Method 1: ==🟡Non-linear featurizer== (DAS + MLP)
$$h = \mathrm{GeLU}(W_u x) \qquad (3)$$
$$\hat{x} = \tanh(W_d h) \qquad (4)$$

| Symbol | Meaning | Simply |
|---|---|---|
| $x$ | the model's activation at that spot | input, e.g. 4096 numbers |
| $W_u$ | learned **up-projection** | expand to more numbers |
| GeLU | smooth ReLU (negatives → ~0) | bend #1 |
| $h$ | expanded, bent vector | a hidden "workspace" |
| $W_d$ | learned **down-projection** | compress back to the original size |
| tanh | squashes every number into $[-1, 1]$ | bend #2 |
| $\hat{x}$ | result | the non-linear features DAS works with |

- ==Without the two bends, up + down would collapse into one matrix, i.e. linear again.==
- **"Augments the feature mixing stage":** in DAS, the rotation mixes neurons into features. Here an MLP is added to that stage, so ==🔴features can be **curved** combinations of neurons==.

Why non-linearity could help?

> "This allows the featurizer to 'blend' potentially independent representations and go beyond convex combinations of features, which could allow it to learn dependencies where the signal is not strictly separable by individual directions in the original activation space x."

- DAS can only find a concept that is a **weighted straight-line mix** of activations (e.g. $0.7 \cdot A + 0.3 \cdot B$).
- The MLP can find concepts that only appear when activations **interact** ("A **and** B").

Small example:
- direction A = "is it in Europe?", direction B = "is it a capital?"
- target concept = **"European capital"** (yes only if **both**)
- DAS: $0.5A + 0.5B$ gives "medium" both for a European non-capital and for a non-European capital, so it can't express "both"
- MLP: GeLU/tanh can switch on only when both are high ✅
- With a ruler you can only draw straight lines; the MLP is a pen that can draw curves.

Method 2: Orthogonal non-linear projection
- Simplified variant: features pass only through a **tanh**, no feed-forward layer.
- Still allows feature interactions, with **less** expressive power.

The paper's own caveat:
- Non-linear featurizers are **highly expressive** and can ==🟢locate *any* feature, **including ones not in the model**== (Sutter et al., 2025). This echoes the memorisation problem of probing classifiers (Belinkov, 2022).
- ==🔴"Additional validation is needed to confirm that the learned features capture genuine variables the model employs."==

---
Section 4.4 : Results (Table 4, faithfulness / IIA)

Numbers are the **mean across counterfactual datasets and layers**; (best layer) in brackets.
RAVEL, Llama-3.1 8B:

| Method | Continent | Country | Language |
|---|---|---|---|
| DAS | 75 (83) | 58 (64) | 63 (70) |
| DBM | 68 (80) | 53 (59) | 58 (64) |
| Full Vector | 53 (62) | 47 (53) | 47 (57) |
| **Orthogonal** (tanh only) | **84 (89)** | **70 (79)** | **72 (79)** |
| **Nonlinear** (MLP) | 83 (89) | **70 (78)** | **72 (79)** |

Other tasks (highlights)

| Task / model | DAS | Orthogonal | Nonlinear |
|---|---|---|---|
| MCQA, Llama-3.1: answer / order | 94 / 77 | – | 95 / 81 |
| MCQA, Qwen-2.5: answer / order | 86 / 78 | 90 / 78 | 89 / 81 |
| Arithmetic (+), Llama-3.1: carry | 54 (65) | 53 (65) | – |

- **Takeaway 1:** both beat DAS on RAVEL by about **+9 to +12 points** (country 58 → 70).
- **Takeaway 2:** ==the simple tanh-only version is as good as the full MLP== → ==🔴*a little* non-linearity seems enough==.
- **My observation:** the paper says both "achieve significant gains over DAS across tasks and models", but on **arithmetic (Llama)** Orthogonal is **53 vs DAS 54**, i.e. no gain. ==🔴The claim is strongest for RAVEL.==
---
Section 5 : Conclusions (causal variable part)
- Non-linear mediators give significant gains, "==🔵even when the non-linearity is built on top of a simple architecture==".
- Expressive featurizer training may yield more gains, **but ==🟡more controls are needed** to ensure concepts truly in the model are being isolated (as opposed to the featurizer learning the variable itself)==.
---
Open question: "This is only applied during training"

Is the MLP used at test time?
- **A: no.** It only helps training → the reported scores come from a mostly **linear** featurizer.
- **B: yes.** The sentence just means its weights are learned in training.

Clues
- The intro uses the same wording: improvements came from non-linearities/MLPs **"during training"**.
- The leaderboard requires an invertible $F$ / $F^{-1}$, and an MLP + tanh isn't built to be invertible → points to **A**.

Status: ==unresolved==
- The core research paper titled ==🔴BlackboxNLP-2025 MIB shared task: Exploring the impact of non-linear modules on distributed alignment search== isn't in the BlackboxNLP 2025 proceedings → this overview may be the only public description.
- **2026-11-09:** emailed Philipp Mondorf (co-author) asking about this and for a write-up/code. Waiting on a reply from him.
---
Relevance to my thesis

| Point                                 | What it means for me                                                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Non-linear DAS beats DAS on **RAVEL** | supports the non-linear part of my hybrid, on my benchmark                                                                    |
| tanh-only ≈ full MLP                  | supports the idea of a **small, constrained** non-linearity (e.g. Kim's invertible map with a Lipschitz knob), not a free MLP |
| Leaderboard needs $F$ and $F^{-1}$    | my method should be **invertible by design**, so it fits this framework cleanly                                               |
| Paper's own caveat (Sutter et al.)    | I need the **random-model control**: high IIA on a random network = the featurizer is cheating                                |
| Only 1 team, 2 methods                | the space is **wide open**; nobody has combined non-linearity with HyperDAS or a learned size                                 |
| MIB / RAVEL setup + leaderboard       | a standard place to report results (maybe this is Matej meant by "crush them")                                                |

---
One-line takeaway
- ==A small non-linearity on top of DAS gives about +10 points IIA on RAVEL, but whether it finds real model variables (and even whether it is used at test time) is unverified; a constrained, invertible map plus a random-model control is exactly the gap my thesis can fill.==
