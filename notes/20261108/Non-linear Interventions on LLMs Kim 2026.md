
---
Why I read this?

> I'm looking into **what else counts as an intervention beyond a subspace swap** and to look into **non-linearity**. This paper replaces the DAS rotation with an **invertible non-linear map** $f_\theta$. Linear DAS is the special case where $f_\theta$ is linear.

---
Big Picture
- **Task:** refusal bypass steering. Edit the model's insides so a safety-trained chat model stops refusing harmful prompts. This is a standard *test case*, not the goal itself.
- **Claim 1:** their non-linear edit works **about as well** as strong linear baselines…
- **Claim 2:** …while editing **~1 spot** instead of hundreds or tens of thousands.
- **Claim 3:** the advantage comes from refusal being stored **non-linearly** (on a curved surface), mostly in the **middle layers**. If $f_\theta$ is made linear, the method stops working.
Kinda similar to : The linear methods turn down the slider on *every channel*. This method finds *the one knob* that actually controls the volume.
---
Eq. 1: the standard linear intervention

$$\mathbf{h} \leftarrow \mathbf{h} + \sum_{i=1}^{k} \alpha_i \mathbf{v}_i$$

Short definition: push the hidden vector a little along a few chosen directions.

| Symbol | Meaning |
|---|---|
| $\mathbf{h}$ | hidden state at one (layer, token); $d$ numbers (576 for SmolLM2, 4096 for Llama-3-8B) |
| $\leftarrow$ | "replace with": an edit, not an equation to solve |
| $\mathbf{v}_i$ | a **feature direction** in hidden space (e.g. "refusal"); orthonormal = length 1, at 90° to each other |
| $\alpha_i$ | **how far** to push along $\mathbf{v}_i$ (+ or −) |
| $k$ | how many directions are edited (cf. `--das_dimension`) |


> [!note] Link to DAS
> DAS is Eq. 1 with $\alpha_i = \mathbf{v}_i^\top \mathbf{s} - \mathbf{v}_i^\top \mathbf{h}$ ($\mathbf{s}$ = source hidden state): "move the slider from where the base has it to where the source has it." In code: `base + (rotated_source - rotated_base) @ R.T`.

---
What $\mathbf{v}_i$, $\mathbf{e}_i$, $\alpha_i$ represent

| Symbol | Lives in | What it is |
|---|---|---|
| $\mathbf{v}_i$ | **hidden space** (the model's own coordinates) | the tilted direction where feature $i$ sits; in DAS = columns of the rotation $R$ |
| $\mathbf{e}_i$ | **feature space** (after $W\mathbf{h}$ or $f_\theta(\mathbf{h})$) | standard basis vector $(0,\dots,1,\dots,0)$: feature $i$ is just "coordinate $i$" |
| $\alpha_i$ | a single number | how far to move along feature $i$ |

- Link: $W^\top \mathbf{e}_i = \mathbf{v}_i$, i.e. rotate axis $i$ back and you get $\mathbf{v}_i$.
- Tilted-map analogy: in the real city "north" is diagonal ($\mathbf{v}_i$); rotate the map upright and north is just "up" ($\mathbf{e}_i$).

> [!important] Why the paper switches from $\mathbf{v}_i$ to $\mathbf{e}_i$
> In the linear case, $\mathbf{e}_i$ always maps back to the **same straight** $\mathbf{v}_i$. With a non-linear $f_\theta$ there is **no single** $\mathbf{v}_i$ any more: a straight axis in feature space becomes a **curve** in hidden space, and its direction depends on where $\mathbf{h}$ is.

---
Eq. 2 → Eq. 3: from rotation to curved map

**Eq. 2** is Eq. 1 written as a change of basis:

$$\mathbf{h} \leftarrow W^{-1}\Big(W\mathbf{h} + \sum_{i=1}^{k} \alpha_i \mathbf{e}_i\Big)$$

1. rotate into feature coordinates ($W\mathbf{h}$)
2. edit coordinate $i$ by $\alpha_i$
3. rotate back ($W^{-1}$) → this is exactly what DAS does

**Eq. 3** (the paper's idea) replaces $W$ with an invertible non-linear $f_\theta$:

$$\mathbf{h} \leftarrow f_\theta^{-1}\Big(f_\theta(\mathbf{h}) + \sum_{i=1}^{k} \alpha_i \mathbf{e}_i\Big)$$

- "Straighten" the space with $f_\theta$, edit there, bend it back with $f_\theta^{-1}$.
- If $f_\theta$ is linear, Eq. 3 = Eq. 2, so ==linear DAS is a special case==.

---
Eq. 4: the interchange intervention (= DAS in curved coordinates)

$$\mathbf{h}^- \leftarrow f_\theta^{-1}\Big(f_\theta(\mathbf{h}^-) + \sum_{i=1}^{k} \alpha_i \mathbf{e}_i\Big), \qquad \alpha_i = \big(f_\theta(\mathbf{h}^+) - f_\theta(\mathbf{h}^-)\big)_i$$

| Symbol | Meaning | RAVEL / DAS analogue |
|---|---|---|
| $x^+$ | input that **has** feature F (here: harmless prompt, the model answers) | source |
| $x^-$ | input that **doesn't** (here: harmful prompt, the model refuses) | base |
| $\mathbf{h}^\pm$ | their hidden states at the intervention site | `source`, `base` |
| $\alpha_i$ | source value − base value on coordinate $i$ | `rotated_source - rotated_base` |

> [!note] "The edit"
> Whenever the paper says "after the edit" it means: run $x^-$, stop at the intervention site, apply Eq. 4, then **continue the forward pass** with the edited $\mathbf{h}^-$.

---
Eq. 5: what gets trained

$$\mathcal{L}(\theta) = \mathbb{E}_{(x^-, x^+)}\big[\ell\big(\mathcal{M}_{\text{int}}(x^-, x^+; \theta)\big)\big]$$

- Average the loss $\ell$ over random (negative, positive) pairs.
- The **LLM is frozen**; only $\theta$ (the map $f_\theta$) learns.
- Same structure as DAS / Hyper DAS. ==The difference is $\ell$== → Eq. 6.
---
#### Site selection: building the "thermometers" (before training)

Why it's needed: refusal is an **implicit feature**. There is no single correct output word to check, so cross-entropy can't be used. Instead they grade the **inside** of the model.

Recipe, computed **once** and then fixed:
1. Run the **unedited** model on **all** $x^+$ and **all** $x^-$ inputs.
2. At every site $s$ (layer, token) **downstream** of the intervention: $\mathbf{v}_s$ = mean($x^+$ states) − mean($x^-$ states) (difference of means).
3. $\mathbf{v}_s^\top \mathbf{h}_s$ = one number, a "looks like $x^+$" meter: high = $x^+$, low = $x^-$.
4. Keep only the sites where this meter separates the two groups well (**AUC > τ**) → the set $\mathcal{S}$.
5. $\mu_s^+$ = the average meter reading on real $x^+$ inputs (the target value).

> [!tip] Hallway analogy
> The model is a hallway of rooms. The edit happens in **one room** near the start. Some later rooms have **thermometers** ($\mathcal{S}$), and you know the normal temperature of a warm input in each ($\mu_s^+$).
---

Eq. 6: the loss (hinge on the thermometers)
$$\ell(\mathcal{M}_{\text{int}}) = \sum_{s \in \mathcal{S}} \max\big(0,\; \mu_s^+ - \mathbf{v}_s^\top \mathbf{h}_s^{\text{int}}\big)$$
In plain words: after the edit, read every thermometer. If it's **too cold**, the penalty is the missing degrees. If it's **warm enough or warmer**, the penalty is 0. Add it all up.

| Room | Should be at least | Edit gave | Penalty |
|---|---|---|---|
| A | 20° | 15° | **5** |
| B | 20° | 20° | **0** |
| C | 20° | 30° | **0** (no bonus for extra warmth) |

Total = 5 → training warms up room A.

> [!note] Why the hinge $\max(0, \cdot)$
> Without it, an over-shooting site (−3) could cancel an under-shooting one (+3) → total 0 looks perfect but isn't. With the hinge, **every** site must reach its target on its own, and none gets over-pushed.

> [!important] Caveat
> The thermometers are **linear** (difference of means), so a non-linear edit is graded by **linear yardsticks**. RAVEL avoids this, because it checks the actual output token.

---
Baselines: DIM and RDO (all linear)

2 ways to **find** the refusal direction × 2 ways to **use** it = 4 baselines.

| Find | How |
|---|---|
| **DIM** (Difference In Means, Arditi et al. 2024) | mean(refused) − mean(answered); no training |
| **RDO** (Refusal Direction Optimization, Wollschläger et al. 2025) | learned direction with ablation + addition + retain losses |

| Use | What | Where | Spots edited |
|---|---|---|---|
| **Ablation** | remove the direction | every token × every layer | tens of thousands |
| **ActAdd** | add $\alpha \cdot$ direction | one layer × every token | hundreds |
| *Kim (non-linear)* | Eq. 4 at one coordinate | one layer × one token | **1** |

---
Setup choices

| Choice | What | Why |
|---|---|---|
| $f_\theta$ = **i-ResNet** | invertible residual network (Behrmann et al. 2019) | curved **and** reversible |
| $k = 1$ | edit one coordinate | fair one-vs-one comparison with the 1-D baselines |
| One position per model, chosen by hand | Llama: layer 10, token −3 · Qwen: layer 14, token −4 | ==still a manual step==, like layer choice in DAS/HyperDAS |
| $\bar\mu^+$ at test time | average coordinate-1 value of $f_\theta(\mathbf{h})$ over harmless training prompts; each test activation is **set (clamped)** to it | a user types only one prompt, so there's no $x^+$ partner to copy from |

---
Metrics

**StrongREJECT** : "did the steering work?"
- 313 forbidden prompts + an LLM grader; score 0–1
- 0 = refused or useless answer · 1 = real, detailed answer
- also checks **quality**, so gibberish doesn't count as a jailbreak

**Total intervention magnitude ($L_2$)**: "how much did you have to change?"

$$\frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{x \in \mathcal{D}_{\text{test}}} \; \sum_{s \in \mathcal{E}_x} \big\lVert h'_s(x) - h_s(x) \big\rVert_2$$

- $h_s$ / $h'_s$ = hidden state before / after the edit at site $s$
- $\lVert\cdot\rVert_2$ = ordinary vector length (e.g. $\lVert(3,4)\rVert_2 = 5$)
- sum over the **edited** spots $\mathcal{E}_x$, then average over test prompts
- worked example: $x_1$ edits 2 spots (sizes 3, 1) → 4; $x_2$ edits 1 spot (size 2) → 2; average = **3**

> [!important] $\mathcal{E}_x \neq \mathcal{S}$
> | | $\mathcal{E}_x$ | $\mathcal{S}$ |
> |---|---|---|
> | role | spots the method **writes to** | checkpoints, only **read** |
> | Kim's method | 1 spot | many downstream spots |
> | used | at test time (this metric) | only in training (Eq. 6) |
>
> Knock-on changes downstream of the edit are **not** counted, so the metric measures "how much you touch," not "how much the computation changes overall."

---
Results

**Figure 1 (StrongREJECT)**
- models: Llama-3-8B-Instruct, Qwen-2.5-7B-Instruct (frozen)
- training data: 2,000 harmless (Alpaca) = $x^+$, 2,000 harmful (SALAD-Bench) = $x^-$
- conditions: no intervention (low, the model refuses) · 4 linear baselines (high) · Kim (high: **0.813** Llama, **0.760** Qwen)
- RDO-ActAdd on Qwen scored 0.020 in their runs → ==🔴judged unreliable and left out==

**Ablation, Table 2: "is it just the thermometers?"**
- Test: no $f_\theta$; at **every** site in $\mathcal{S}$, push along $\mathbf{v}_s$ until the reading equals $\mu_s^+$ exactly.
- Result: **much worse** than Kim's one edit, on both models.
- Meaning: $f_\theta$ changes the **real cause** at one spot; the thermometers rise **as a side effect**.

> [!tip] Heater vs lighter
> Forcing the thermometers = holding a lighter under each one: they read 20°, but the rooms are still cold. Kim's edit = turning up the heater in the first room: warm air flows down the hallway.

---
Appendix A.1: how $f_\theta$ is built

- **2 residual blocks**, each: output = input + $g$(input)
- $g$ = 2-layer MLP, width **128** ($d \to 128 \to d$), **LeakyReLU** (slope 0.1) → ==🟢this is what makes it curved==
- **Reversible because $g$ is "gentle"** (Lipschitz < 1):
  - spectral normalization on every linear layer (each layer stretches ≤ 1×; estimated with 1 power iteration per step)
  - output scaled by **0.6** → safety margin below 1
- **Inverse = guess and correct** (fixed-point iteration): $x_{\text{new}} = y - g(x_{\text{old}})$, up to **30** rounds, tolerance $10^{-5}$
- **Gradients through the inverse:** ==🟠implicit differentiation== (a shortcut that avoids storing all 30 rounds)
---
Relevance to my thesis:

|                                      | Kim                                | My HyperDAS hybrid                                                              |
| ------------------------------------ | ---------------------------------- | ------------------------------------------------------------------------------- |
| Swap (Eq. 4)                         | ✅                                  | ✅ use as is, $f_\theta$ instead of HyperDAS's rotation                          |
| Frozen model, train $\theta$ (Eq. 5) | ✅                                  | ✅ same                                                                          |
| Loss $\ell$                          | Eq. 6 hinge on linear thermometers | ❌ not needed: **cross-entropy on the source answer** (RAVEL has a target token) |
| $k$                                  | 1, fixed                           | ==🔴larger and **learned** (my idea)==                                          |
| Position                             | chosen by hand                     | HyperDAS's hypernetwork **picks tokens**                                        |
| Test-time value                      | one average $\bar\mu^+$            | **per-example source** (Tokyo ≠ Kyoto, so an average wouldn't work)             |
| Evaluation                           | StrongREJECT + $L_2$               | IIA / causal / isolate / disentangle on RAVEL                                   |

Design ideas this gives me
- **Lipschitz coefficient (0.6) = a "how non-linear" knob** → sweep it (e.g. 0.1 / 0.3 / 0.6) to control flexibility.
- **Copy the structure of their Table 2 ablation:** show my non-linear map does something a linear edit can't.

> [!important] Gaps and caveats
> - Tested only on **refusal steering**: no IIA, no RAVEL, no comparison with DAS/HyperDAS.
> - "Genuinely non-linear" rests on "the linear version fails"; it does **not** rule out the Sutter et al. (2025) concern that a flexible map can fit *anything* → I need the **random-model control**.
> - I couldn't find a venue for the paper.
> - The non-linear edit is graded with linear thermometers (Eq. 6).
---
## One-line takeaway
- ==Kim shows that an invertible non-linear map can do with 1 edit what linear methods need thousands for, but not that what it finds is a real causal variable. Testing it in a causal-abstraction setting (RAVEL + HyperDAS + random-model control) is my contribution.==
