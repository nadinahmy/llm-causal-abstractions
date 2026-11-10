
---
Why I read this
> It is **the main objection to my hybrid** (HyperDAS + learned size + non-linear map). If the alignment map $\varphi$ can be any reversible function, **any algorithm fits any network**, so a high IIA proves nothing.
> I need to explain this and show how my method avoids it.
---

Big Picture
- **Claim:** once you drop the "linear" restriction on $\varphi$, causal abstraction becomes **vacuous** (empty): under mild assumptions, *any* DNN can be perfectly mapped to *any* algorithm (**Theorem 1**).
- **The dilemma:** we *want* to allow non-linear encodings (evidence says some concepts are non-linear), but allowing them ==🔵makes the analysis meaningless unless $\varphi$ is constrained==.
- **Experiments:** re-run the DAS experiments with more powerful maps → layer patterns from the DAS paper **vanish**; **>80% IIA on randomly initialised MLPs**; **near-perfect IIA on random Pythia models** for IOI.
- **Author's advice:** decide case by case. Believe the linear hypothesis → constrain $\varphi$ to linear. Don't → study the **accuracy vs complexity** trade-off of $\varphi$.
---
Figure 1: the paper in one picture
- **Left:** a small True/False network for hierarchical equality, and the algorithm it should match ($\eta_1$: a == b, $\eta_2$: c == d).
- **Hidden layer** drawn as a 2-D plot (neurons $\psi_1, \psi_2$): ○ = label True, × = label False; the network has accuracy 1.0.
- **Middle:** the same points mapped three ways (identity / linear / non-linear) → swap the $\eta_1$ coordinate between example inputs → map back with $\varphi^{-1}$ → rest of the network.
- **Right:** IIA rises with flexibility.

| $\varphi$ | IIA |
|---|---|
| Identity | 0.8 |
| Linear | 0.9 |
| Non-linear | **1.0** |

---
Section 2.2 : Deep Neural Networks

The network as a chain of functions (Eq. 3):

$$h_{\psi^1} = f^0_N(x), \qquad h_{\psi^{\ell+1}} = f^\ell_N(h_{\psi^\ell}), \qquad p_N(y \mid x) = f^L_N(h_{\psi^L})$$

| Symbol | Meaning |
|---|---|
| $x$ | input |
| $\psi^\ell$ | set of neurons in layer $\ell$ |
| $h_{\psi^\ell}$ | their activation values (a vector) |
| $f^\ell_N$ | what layer $\ell$ computes (layer $\ell$ → layer $\ell+1$) |
| $p_N(y \mid x)$ | output: a probability per answer (a point in $\Delta^{\lvert Y\rvert-1}$, probabilities summing to 1) |
| $f^{\psi'}_N(x')$ | "run on $x'$ and report neurons $\psi'$" ($\psi'$ can be any subset) |

- Neurons are computed **in order** ($\psi^0 \prec \psi^\ell \prec \psi^L$), a *partial* order: neurons in the same layer aren't ordered. This mirrors the algorithm's variable order and is what lets section 3 match the two.
- **Intervention:** $f^{\psi^{L+1}}_N\big(x, (h_\psi \leftarrow c_\psi)\big)$ = run on $x$ but **force** neurons $\psi$ to values $c_\psi$; everything else runs normally.

> [!note] Link to DAS
> An **interchange intervention** is this definition with $c_\psi = f^\psi_N(\text{source})$: run the base, but give $\psi$ the values it had for the source.
---

Section 3 : Constraint on τ: same input, same output

> "we restrict: $\psi_{\eta_x}$ to be the neurons in layer zero and $\tau_{\eta_x}$ to be the identity; and $\psi_{\eta_y}$ to be the neurons in layer L+1 and $\tau_{\eta_y}$ to be the argmax operation."

- **Input:** the algorithm's input variable = layer 0 (the raw input), read with the **identity** → both get **exactly the same input**.
- **Output:** the algorithm's output = the last layer's probabilities, read with **argmax** (top prediction) → the network's answer must equal the algorithm's.
- **Why:** ==🔴otherwise $\tau$ could quietly reinterpret inputs or relabel answers. Now **all the freedom is in the inner variables**.==
- Footnote 4: this implies the algorithm and the network must give the **same outputs on every input**.
---
Section 3.1 : Three hypotheses about how information is stored

| Def.  | Hypothesis                                       | A concept lives in…                                                         | Methods that assume it                                       |
| ----- | ------------------------------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **3** | Privileged bases (Elhage et al. 2023)            | **whole neurons**                                                           | early causal abstraction, localist alignment                 |
| **4** | Linear representation (Alain & Bengio 2016)      | a **straight subspace** (any rotated direction; neurons are a special case) | **DAS**, Boundless DAS, HyperDAS                             |
| **5** | Non-linear representation (Pimentel et al. 2020) | a **curved** subspace                                                       | non-linear featurizers (Kim, Hirlimann), **==🟡my hybrid==** |

- **Def. 3 evidence** is indirect: outlier neurons, Karpathy's (2015) single meaningful neuron. It fits **constructive abstraction** (each variable in its own neurons).
- Each step is more flexible. ==By Def. 5 you are so flexible you can find "north" anywhere, even where there isn't one.==

> [!note] Link to the DAS paper
> Neuron-aligned interventions **failed** on the toy example and the **−20° rotation** fixed it: evidence against Def. 3, for Def. 4, in that network.

---
Section 3.2 : Distributed causal abstractions (Def. 6)

**Problem:** constructive abstraction (Def. 2) needs each concept in its own separate neurons, but real networks spread concepts across neurons → too strict.

**Fix: a distributed abstraction map $\tau$, with two conditions**
1. **Alignment map $\varphi$:** a **bijective** (reversible), same-size map, applied **layer by layer** ($\varphi_\ell: \mathbb{R}^{\lvert\psi^\ell\rvert} \to \mathbb{R}^{\lvert\psi^\ell\rvert}$), turning neurons into **latent variables** $h_{\psi^\varphi} = \varphi(h_\psi)$.
2. **Partition:** split the latents into non-overlapping groups, one per inner concept ($\psi^\varphi_\eta$) plus an unused group ($\psi^\varphi_\perp$); each concept is read only from its own group (by $\tau_\eta$).

**Distributed intervention (3 steps)**
1. map in: $h \to \varphi(h)$
2. swap only the concept's group: $h_{\psi^\varphi_\eta} \leftarrow c_{\psi^\varphi_\eta}$
3. map back: $h' = \varphi^{-1}(h'_{\psi^\varphi})$ → the network continues

**Definition 6:** A is a **distributed abstraction** of N iff some $\tau$ makes A a **strong** $\tau$-abstraction (every intervention matches) **and** $\tau$ is a distributed abstraction map.

| $\varphi$ is… | Then it's… |
|---|---|
| identity | privileged bases (Def. 3) |
| rotation | DAS (Def. 4) |
| **any** reversible function | non-linear (Def. 5) → the source of the dilemma |

---
Input-restricted interventions (Def. 7)

**Problem:** "strong" abstraction requires matching under **every** possible intervention, including arbitrary values (set a neuron to 7.3, −1000, …) → infinitely many, mostly meaningless tests.

**Fix:** only allow values **produced by running a real input**.

| Case | Formula | Simply |
|---|---|---|
| simple | $c_{\psi'} = f^{\psi'}_N(x')$ | the value those neurons had for another real input $x'$ |
| nested | $c_{\eta'} = f^{:\eta'}_A\big(x', (v_{\eta''} \leftarrow f^{:\eta''}_A(x''))\big)$ | the value on $x'$ while another variable was already swapped from $x''$ |

- ✅ allowed: set WX in base (red, red, blue, green) to its value from a real source input
- ❌ not allowed: random numbers no input ever produces
- ==This is exactly an interchange intervention.==

**Definition 7:** A is an **input-restricted distributed abstraction** of N iff A is an input-restricted $\tau$-abstraction (weaker: checked only on realistic swaps) and $\tau$ is a distributed abstraction map.

> [!important] Why it matters
> **Def. 7 is what DAS / HyperDAS / IIA actually test.** 100% IIA strictly shows Def. 7, not Def. 6, and the paper shows even Def. 7 becomes trivial with an unrestricted $\varphi$.

---
Section 3.3 : Finding distributed abstractions (= DAS)

> "one must assume a partitioning $\{\psi^\varphi_\eta\} \cup \{\psi^\varphi_\perp\}$ which remains fixed during the method's application; we term $\lvert\psi^\varphi_\eta\rvert$ the intervention size"

- Before DAS you **choose** which latent coordinates belong to which concept; this split **doesn't change** during training. DAS only learns $\varphi$.
- **Intervention size** $\lvert\psi^\varphi_\eta\rvert$ = how many coordinates per concept = **`--das_dimension`** (my k).
- Example: 16 latents → WX = 1–4, YZ = 5–8, unused = 9–16.

| Their notation | DAS / my code |
|---|---|
| $x_\emptyset$ | base input |
| $x_\eta$ | source (counterfactual) input for variable $\eta$ |
| $I_N$ (Eq. 7) | swap in the network, after $\varphi$ |
| $I_A$ (Eq. 7) | same swap in the algorithm → correct answer $y$ |
| Eq. 8 | cross-entropy loss, trained by gradient descent |

- DAS **never builds $\tau$ explicitly**; it relies only on the output constraint ($\tau_{\eta_y}$ = argmax). A low loss is **taken as** evidence of abstraction.

> [!note] Link to my thesis
> Boundless DAS made the intervention size **learnable**, breaking the "fixed partition" assumption. My idea attacks exactly this sentence, inside HyperDAS.

---
Section 4 : Unbounded abstractions are vacuous (Theorem 1)

**Theorem 1:** Given any algorithm A and any network N satisfying Assumptions 1–5, A is an **input-restricted distributed abstraction** of N. (Proof: App. F, skipped.)

| # | Assumption | Plain question | Realistic? |
|---|---|---|---|
| 1 | Countable input space | Can all inputs be listed one by one? | ✅ texts, images |
| 2 | **Input-injective** in all layers | Does every layer keep all input information (different inputs → different states)? | ✅ almost surely for transformers (App. G); approximate in practice (rounding, neural collapse) |
| 3 | **Strict output-surjective** in all layers | From each layer, can some state produce any output? | 🟡 can fail in theory (softmax bottleneck); mostly fine in practice |
| 4 | Matchable partial orderings | Can the network's layers be lined up with the algorithm's steps? | ✅ for big networks |
| 5 | The network solves the task | Correct answers without interventions? | ✅ needed anyway |

**Why it works (rough idea)**
- **(2)** every hidden state is a unique **fingerprint** of its input → $\varphi$ can act like a **lookup table** ("this fingerprint = input #5,213 → WX = True") and write that value into a latent coordinate, **whether or not the network computed WX**.
- **(3)** after the swap, $\varphi^{-1}$ can always pick some hidden state that makes the rest of the network output what the algorithm predicts.
- ==So $\varphi$ doesn't *find* WX; it computes WX itself and steers the output.==

> [!important] The honest limit
> The proof is **extreme overfitting**: it shows such a $\varphi$ **exists**, not that it can be **learned** from limited data or **generalises**.

---
Section 5 : Experimental setup

Three alignment maps, from least to most flexible (all bijective and easy to invert):

| Map | Definition | Hypothesis |
|---|---|---|
| $\varphi_{\text{id}}$ | $\varphi(h) = h$ | privileged bases |
| $\varphi_{\text{lin}}$ | $W_{\text{orth}}\, h$ (orthogonal matrix, as in pyvene) | linear |
| $\varphi_{\text{nonlin}}$ | RevNet with $L_{rn}$ layers, hidden size $d_{rn}$ (more = more flexible) | non-linear |

**Greedy search for $\varphi_{\text{id}}$**
- No map to learn, so they **search the partition**: start with empty groups, each round add **one neuron per concept**, ==🔴the one that most lowers the loss (Eq. 8), up to the maximum intervention size==.
- Least flexible = hardest test → passing it is the strongest evidence.

**Metric:** IIA on a held-out test set, compared with the network's own accuracy without interventions.

**Tasks**
- **Hierarchical equality:** $y = (x_1 == x_2) == (x_3 == x_4)$, 3-layer MLP (hidden 16). Three algorithms: *both equality relations*, *left equality relation*, *identity of first argument*.
- **IOI:** "Friends Juana and Kristi found a mango at the bar. Kristi gave it to" → "Juana". Pythia models (31M–410M), ABAB-ABBA algorithm.

---
Section 6 : Results

| Finding                                        | Details                                                                                                                                      |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Linear layer patterns vanish**               | with $\varphi_{\text{lin}}$, IIA drops in layer 3 (as in the DAS paper); with $\varphi_{\text{nonlin}}$, near-optimal IIA at **all** layers  |
| **Random MLPs: >80% IIA**                      | the most complex map aligns even an untrained network (Fig. 3 right)                                                                         |
| **Simpler maps catch up with training**        | as the MLP trains, simpler maps reach high IIA too                                                                                           |
| **IOI: all Pythia sizes align**                | including 31M / 70M models that **can't do the task**, and **randomly initialised** models (near-perfect)                                    |
| **$\varphi_{\text{lin}}$ tracks real ability** | ==🟡on Pythia-410M, linear IIA follows the model's actual accuracy over training==; the 8-layer non-linear map is near-perfect at every step |
| Identity of first argument                     | ~50% for all maps; App. H suggests ==🔴insufficient RevNet capacity==                                                                        |

---
Section 7 : Discussion

**Causal abstraction is not enough**
- Early methods assumed **privileged bases**, DAS assumes **linear**; the framework itself doesn't solve how representations are encoded.

**Accuracy vs complexity of $\varphi$: the probing analogy**
- A **probe** = a classifier trained on hidden states to predict a concept. Simple probes assume linear encoding; complex probes can **memorise the task**.
- Same problem here: probe architecture ↔ $\varphi$, probe accuracy ↔ IIA, complex probe memorises ↔ complex $\varphi$ "finds" any algorithm.
- The probing debate was **never solved**. Proposed fixes:

| Fix                                                  | Idea                                                      | For my thesis                                                           |
| ---------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------- |
| Control tasks (Hewitt & Liang 2019)                  | test on a nonsense version; doing well there = memorising | ✅ **random-model control**                                              |
| Complexity–accuracy trade-off (Pimentel et al. 2020) | report accuracy across a range of complexities            | ✅ ==🔴sweep the **Lipschitz knob** / map size; plot IIA vs complexity== |
| MDL probes (Voita & Titov 2020)                      | real information compresses well; memorisation doesn't    | too advanced?                                                           |
| Unsupervised probes (Burns et al. 2023)              | no labels to guide the probe                              | doesn't seem too relevant                                               |

**The role of generalisation**
- On IOI with **disjoint names** in train and test, ==🟢the learned map **fails to generalise** → generalisation should be central to causal abstraction analysis==.

**Investigating representation encoding**
- Comparing maps helps: $\varphi_{\text{lin}}$ follows the real model's accuracy, $\varphi_{\text{nonlin}}$ is always near-perfect → ==🔴intuitively supports the linear hypothesis here, but the authors **can't make this formal**==. $\varphi_{\text{lin}}$ also sometimes exceeds the model's accuracy, so it may "learn the task" too.
---
Section 8 : Conclusion and limitations

- ==🔵Case by case: linear $\varphi$ if you believe the linear hypothesis; otherwise face the dilemma and study the accuracy–complexity trade-off.==
- **Limitations:** the proof relies on overfitting, yet their learned maps *do* generalise to unseen test data (worth studying why); Assumptions 2 and 3 may fail in some settings (e.g. softmax bottleneck).
---
Relevance to my thesis

| Lesson                                            | What I'll do                                                                                                                   |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| A flexible map can "find" anything                | **constrain** the non-linear map: invertible by design (i-ResNet or coupling layers) + a **Lipschitz "how curved" knob** (Kim) |
| High IIA alone is not evidence                    | **random-model control**: high IIA on a random SmolLM2 / MLP = cheating                                                        |
| Generalisation separates real from memorised      | test on **held-out entities** (RAVEL cities not seen in training)                                                              |
| Compare map complexities                          | report **IIA vs complexity** (linear → slightly curved → more curved), like Fig. 3                                             |
| Linear DAS is "safe" only because it's restricted | state the linear restriction as an **explicit assumption** of HyperDAS                                                         |
| Track over training                               | if possible, check whether IIA follows the model's real ability across checkpoints                                             |

- In summary, Sutter et al. show that unrestricted non-linear maps make causal abstraction vacuous; my map would be constrained (invertible, bounded curvature) and I'd validate it with a random-model control, a held-out-entity test, and an IIA-vs-complexity sweep."
- ==With an unrestricted alignment map, any network matches any algorithm (100% IIA even on random models), so a non-linear method is only meaningful with constraints on $\varphi$ and controls that a memorising map would fail.==
