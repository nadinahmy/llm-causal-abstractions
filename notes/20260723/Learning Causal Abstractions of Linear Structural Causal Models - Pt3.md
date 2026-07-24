
---
- $\pi_\mathcal{H}$ is a **relabeling of concrete variables** that arranges them into <mark style="background: #ADCCFFA6;">contiguous, correctly-ordered blocks</mark> (one block per abstract variable), so the concrete model's weight matrix $W$ and the abstraction matrix $T$ can be written in the clean block form defined in (14) and (15).
Theorem 3 (Block Abstraction) — Summary:
> [!abstract] Statement
> $\mathcal{H}$ is a linear $T$-abstraction of $\mathcal{L}$ **iff** there exists a valid topological ordering $\prec_\mathcal{L}$ of $\mathcal{G}_\mathcal{L}$ (matching some valid ordering $\prec_\mathcal{H}$ of $\mathcal{G}_\mathcal{H}$) such that for all $Y_i, Y_j \in Y$:
> $$Y_i \prec_\mathcal{H} Y_j \iff \Pi(Y_i)\prec_\mathcal{L}\Pi(Y_j) \tag{17}$$
> $$W_{ij}\,s_j = m_{ij}\,t_i \tag{18}$$

- Purpose : Converts the **functional** definition of abstraction (interventional consistency, Eq. 4) into a **purely algebraic condition** on matrices $W$, $M$, $T$ — ==checkable without reasoning about every possible intervention.==

---
Coefficient condition :  $W_{ij}\,s_j = m_{ij}\,t_i$

| Term | Meaning |
|---|---|
| $W_{ij}$ | concrete weights from block $i$ → block $j$ |
| $s_j = F_{jj}t_j$ | how block $j$'s internal noise combines into $Y_j$ |
| $m_{ij}$ | abstract edge weight $Y_i\to Y_j$ |
| $t_i$ | how block $i$'s concrete vars combine into $Y_i$ |
Recall :
- $s_j$​ is a vector that tells us how the **exogenous noise** inside block $jj$ turns into the value of the abstract variable $Y_j$​.
Here's the distinction that matters:
- $t_j$ tells us how to combine the **values** of the concrete variables in block $j$ to get $Y_j$ (this is just the abstraction function $τ$ applied to that block — literally a column of T).
- $s_j$​ tells us how to combine the **exogenous noise terms** in block $j$ to get $Y_j$​'s own noise term $U_j$.
Why are these different? 
- Because inside a block, concrete variables can causally affect each other (that's what $W_{jj}$ ​captures — the internal edges within the block). So noise entering at one concrete variable in the block doesn't just show up raw at $Y_j$​ — ==it gets **passed along and transformed** by the internal causal structure of the block before it reaches the relevant variable(s) that $τ$ actually looks at.==
$s_j$​ accounts for that internal propagation. Formally:  $s_j = F_{jj} t_j = (I−W_{jj})^{−1} t_j$​
- $F_{jj} = (I−W_{jj})^{−1}$ is the "reduced form" of the block — it's the matrix that says "if we inject noise at variable $X$ inside this block, here's <mark style="background: #ABF7F7A6;">how it ends up affecting every other variable in the block once all the internal causal effects have propagated through.</mark>" Multiplying by $t_j$​ then ==projects that propagated effect onto the specific combination that defines $Y_j$.==

Bottom line of the theorem :
- LHS = actual aggregate effect of block $j$'s noise on block $i$, computed via the real concrete wiring.
- RHS = the effect **predicted** by trusting the single abstract edge weight $m_{ij}$.
- <mark style="background: #D2B3FFA6;">The theorem demands these match exactly — the abstract edge is a lossless summary of the concrete cross-block machinery.</mark>
---
Worked check — [[Example 5]]
Abstract: $Y_1\to Y_2$, $m_{12}=1$. Three candidate concrete models share identical internal block structure ($s_1,s_2$ same for all).
- 2 of 3 satisfy $W_{12}s_2 = t_1$ → valid $T$-abstractions.
- 1 fails → **not** a valid concretization, despite similar-looking graph.

> [!summary]
> Theorem 3 is the bridge from "abstraction = interventional property" to "abstraction = linear equation on parameters," enabling:
> - **Algorithm 1**: sampling valid concrete models from a given abstract model + $T$
> - **Algorithm 2 (Abs-LiNGAM)**: learning both models + abstraction function from data

Example 5 Solved :
![[IMG-20260723201151185.png|581]]

---
#### Algorithm 1
- Algorithm 1 is the "reverse direction" of Theorem 3: <mark style="background: #BBFABBA6;">instead of _checking_ whether a given concrete model is a valid T-abstraction of an abstract model, it _constructs_ one from scratch, guaranteed correct by construction.</mark>
- **Input:** an abstract adjacency matrix M (weights between abstract variables) and an abstraction function T (which concrete variables map to which abstract variables, and with what coefficients).
- **Output:** a concrete adjacency matrix W such that H (defined by M) is guaranteed to be a T-abstraction of L(defined by W) — i.e., Eq. (17) and Eq. (18) from Theorem 3 hold automatically.
Theorem 3 tells _what conditions a valid concretization must satisfy_, but doesn't by itself give a recipe for generating one ---> Algorithm 1 is the recipe.
#### Pseudocode :

```
W ← 0
for Yj ∈ Y do
    Nj ← |Π(Yj)|
    Wjj ← RandomDAG(Nj)
    sj ← (I − Wjj)^{-1} tj
    for Yi ∈ Y do
        for Xk ∈ Π(Yi) do
            v ~ {v ∈ R^{Nj} | sum_h v_h = 1}
            c ← v / sj
            [Wij]_{k,:} ← mij [ti]_k c^T
        end
    end
end
```

Outer loop: process each abstract variable $Y_j$​ as a "target"
For each abstract variable $Y_j$, we're going to decide:
1. What's the internal structure of its block $Π(Y_j)$ (i.e., $W_{jj}$​)?
2. What are the weights coming **into** this block from every other block (i.e., $W_{ij}$​ for every $Y_i$)?
Step A — sample internal block weights $W_{jj}$ ---> $W_{jj}$ ← RandomDAG($N_j$)
- This randomly samples a DAG structure (an upper-triangular weight matrix) on the $N_j$​ concrete variables inside block j. This is the internal wiring within the block — <mark style="background: #FFB8EBA6;">the paper notes we assume any irrelevant variable in the block has at least one relevant variable as a descendant (this is required by the block's definition — recall Lemma 3, the irrelevant part of a block must feed into the relevant part).</mark>
Step B — compute $s_j$​ from $W_{jj}$​ ---> $s_j ← (I − W_{jj})^{-1} t_j$
- Now that $W_{jj}$ has been chosen, we can compute how the block's internal noise propagates to $Y_j$​'s value.

Inner loops: fill in cross-block weights $W_{ij}$​
- Now, for every _other_ abstract variable $Y_i$​ (potential source/parent of $Y_j$​), and for every concrete variable $X_k$​ in $Y_i$​'s block, we need to assign the row of weights going from $X_k$​ into block j's variables.
Step C — sample a "direction" vector v ---> v ~ {$v ∈ R^{N_j} | \sum_{h=1}^{N_j} v_h = 1$}
- This samples a random vector of length $N_j$ (matching block j's size) whose entries sum to 1. Intuitively, this ==decides _how_ the influence from $X_k$​ is spread across the individual concrete variables of block j== — it's a free choice, because Theorem 3's condition ($W_{ij}s_j = m_{ij}t_i$​) only constrains the _aggregate_ effect (via $s_j$​), not the individual entries of $W_ij$​ themselves.
Step D — turn it into a right-inverse of $s_j$ : $c ← v / s_j$
- This is an *element-wise* division. Why? Because we need c to satisfy $c^⊤s_j=1$ (this is what "right-inverse" means here — c inverts $s_j$​ in the sense of the dot product). Since v sums to 1, and $c = v/s_j$​ element-wise, we get:  $c^\top s_j = \sum_{h} c_h (s_j)_h = \sum_{h} (s_j)_h v_h = \sum_{h} v_h = 1.$
- So c is constructed precisely so that $c^⊤s_j = 1$ always holds, no matter what random v you picked — <mark style="background: #FFB86CA6;">this is the trick that makes the whole procedure work.</mark>
Step E — assign the row of $W_ij$
```
[Wij]_{k,:} ← mij [ti]_k c^T
```

This sets row k of $W_{ij}$​ (i.e., the weights from concrete variable $X_k$​ to every variable in block j) equal to $m_{ij}⋅[t_i]_k⋅c^⊤$ — a scalar times the vector $c^⊤$.
- ==**This is the core trick of the algorithm:** by choosing c to be a right-inverse of $s_j$, every row of $W_{ij}$ automatically satisfies Theorem 3's constraint, regardless of the random direction $v$ chosen.==

Why the random v matters?
- The constraint $c^⊤s_j=1$ doesn't pin down $c$ uniquely — <mark style="background: #ABF7F7A6;">there's a whole family of vectors c satisfying it</mark> (a right-inverse of $s_j$​ is not unique unless $N_j=1$). <mark style="background: #FF5582A6;">The random sampling of v (with sum-to-one constraint) is what explores this whole family, giving genuine randomness/diversity to the sampled concrete models rather than always outputting the same trivial solution. </mark>This is what lets Algorithm 1 sample from the **entire set** of valid T-concretizations, not just one canonical example.

After the loops finish, W is fully populated:
- Diagonal blocks $W_{jj}​$ : random internal DAG structure within each block.
- Off-diagonal blocks $W_{ij}$​ (i<j in the topological order): built row-by-row so that $W_{ij}s_j=m_{ij}t_i$ holds exactly.
- By construction, both conditions of Theorem 3 (the block ordering (17), automatically respected because we loop over $Y_j$​'s in a fixed topological order, and the coefficient condition (18)) are satisfied — so the resulting W is guaranteed to define a concrete linear SCM L for which H (given by M) really is a T-abstraction.

Summary : Algorithm 1 samples a random concrete model layer by layer, and at each step uses a small linear-algebra trick (choosing a right-inverse of $s_j$​) to guarantee that Theorem 3's exact equality is always satisfied — <mark style="background: #FFB8EBA6;">turning the _characterization_ of Theorem 3 into a practical, complete, and correct _sampling procedure_ for generating valid concretizations of any abstract linear SCM.</mark>

---
### Data-Generation Process
- In real-world settings, we typically have _lots_ of low-level sensor/concrete data, but only a few paired observations where we also know the high-level abstract values.
	- For example, we might have millions of readings from individual sensors, but only a handful of cases where we also know some higher-level summary measurement.
Paper defines two datasets:
- $D_L \sim P_L$ — a large dataset of concrete-only observations (just samples of $X$).
- $D_J \sim P_{L,H}$ — a small dataset of paired observations, containing both concrete and abstract values together, with $|D_J| \ll |D_L|$.
**The generative process:**
1. Sample exogenous noise $e(i)$ from an **Exponential distribution** (<mark style="background: #BBFABBA6;">chosen because it's non-Gaussian — needed for LiNGAM-style discovery, which relies on non-Gaussianity for identifiability</mark>).
2. Compute the concrete values $x(i) = L(e(i))$ for every sample (this gives $D_L$​).
3. For a _smaller_ subset, also compute the abstract values $y(i)=H(γ(e(i)))$ using the exogenous abstraction function $γ$ (this gives the paired dataset $D_J$).
Because the models are linear and non-Gaussian, both models are identifiable given enough data.
---
#### Abs-LiNGAM Algorithm

- Four-step pipeline exploiting earlier theory to speed up concrete causal discovery using a small paired dataset $\mathcal{D}_J$ + large concrete-only dataset $\mathcal{D}_\mathcal{L}$.
Steps:
**(i) T-Reconstruction** — Fit $\hat{T}$ via least-squares regression of abstract on concrete values using $\mathcal{D}_J$. <mark style="background: #D2B3FFA6;">Relevant variables read off from nonzero entries</mark>: $\hat{\Pi}_R(Y_i) = \{X_k \mid [\hat{t}_i]_k \neq 0\}$ (small coefficients thresholded to zero for noise).

**(ii) Abstract Causal Discovery** — $\mathcal{D}_J$ too small to learn $\hat{M}$ reliably, so instead abstract the *large* concrete dataset: $\mathcal{D}_{\hat{\mathcal{H}}} = \{\hat{T}^\top x \mid x \in \mathcal{D}_\mathcal{L}\}$. <mark style="background: #ABF7F7A6;">Valid by observational consistency</mark> (Eq. 5). Run **DirectLiNGAM** on this synthetic abstract dataset → $\hat{M}$.

**(iii) Prior Knowledge / Constraints** — Using [[Theorem 1]] (Abstract Connectivity): if no abstract path $Y_i \dashrightarrow Y_j$ in $\hat{M}$, then no concrete var in $\hat{\Pi}_R(Y_i)$ can ancestor any concrete var in $\hat{\Pi}_R(Y_j)$. Collect all such forbidden pairs into constraint set $K$.

**(iv) Concrete Causal Discovery** — Run **DirectLiNGAM** on full $\mathcal{D}_\mathcal{L}$, supplying $K$ as prior knowledge → <mark style="background: #FF5582A6;">shrinks search space → faster discovery, no accuracy cost</mark> (excluded paths were guaranteed forbidden).

> [!summary] Pipeline
> small paired data → $\hat{T}$ → blow up to large synthetic abstract dataset → $\hat{M}$ → forbidden-path constraints $K$ → faster constrained concrete discovery

<mark style="background: #FF5582A6;">Breakthrough of Abs-LiNGAM :
Same quality concrete graph as plain DirectLiNGAM, substantially faster — search space pruned via abstract-model info.</mark>
---
Super important to consider when using Abs-LiNGAM: <mark style="background: #FF5582A6;">Paired Sample Threshold Effect</mark>

> [!question] Quote
> "Whenever the number of paired samples approaches the number of concrete nodes $|X|$, Abs-LiNGAM performs similarly to the baseline and correctly retrieves the concrete causal model."

Figure 2a: ROC-AUC of recovered $\hat{\mathcal{L}}$ vs. $|\mathcal{D}_J|$ (paired samples), comparing:
- **DirectLiNGAM** (baseline) — ignores abstract info
- **Abs-LiNGAM** — uses $\mathcal{D}_J$ to fit $\hat{T}$ (Step i), then derives constraints (Step iii) to speed up final discovery (Step iv)
### Small $|\mathcal{D}_J|$ regime
- $\hat{T}$ estimation is a regression with $\approx d = |X|$ unknowns → too few samples ⇒ ==**unreliable $\hat{T}$==**
- Errors cascade downstream:
$$\hat{T} \text{ wrong} \to \hat{\Pi}_R(Y) \text{ wrong} \to \mathcal{D}_{\hat{\mathcal{H}}} \text{ wrong} \to \hat{M} \text{ wrong} \to K \text{ contains false constraints}$$
- ==DirectLiNGAM (Step iv) forced to respect **wrong** forbidden-path constraints → actively excludes true edges → **underperforms baseline**==
### $|\mathcal{D}_J| \approx |X|$ regime
- Enough data to fit $\hat{T}$ reliably (basic regression rule of thumb: #samples ≳ #parameters)
- Accurate $\hat{T}$ → accurate $\hat{M}$ → constraint set $K$ contains only **genuinely forbidden** paths
- Constrained search in Step (iv) only <mark style="background: #FF5582A6;">**shrinks** search space, never excludes true structure</mark>
- Result: same accuracy as baseline, but **faster** (smaller search space)

> [!summary]
> $|X|$ acts as a **break-even threshold**: below it, Abs-LiNGAM's constraints are unreliable and can hurt accuracy; at/above it, constraints become trustworthy and the method gets its speed advantage "for free," matching baseline accuracy.

---
Point I noticed in end of section 5 : Bootstrapping the Abstract Discovery Step

> [!question] Quote
> "Bootstrapping abstract causal discovery, i.e., aggregating several iterations on randomly extracted sub-datasets, improves the performance on the downstream concrete discovery task without noticeably affecting the execution time, which is still dominated by the final concrete causal discovery run."

### Problem
- Step (ii) learns $\hat{M}$ via a **single** DirectLiNGAM run on $\mathcal{D}_{\hat{\mathcal{H}}}$ → noisy/unstable, especially for small $b$ or limited data.
- Errors in $\hat{M}$ → wrong forbidden-path constraints $K$ (Step iii) → can hurt final concrete discovery (Step iv).

### Fix: Bootstrapping
- Run DirectLiNGAM multiple times on **random sub-samples** of $\mathcal{D}_{\hat{\mathcal{H}}}$, then **aggregate** results (e.g. keep edges consistent across runs) → more stable/robust $\hat{M}$.
- <mark style="background: #FF5582A6;">Standard statistical variance-reduction trick.</mark>
### Two findings

| Effect                  | Why                                                                                                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ✅ **Improves accuracy** | More reliable $\hat{M}$ → better constraints $K$ → better final concrete graph                                                                                                            |
| ✅ **~Free in runtime**  | Abstract graph is small ($b \ll d$, e.g. $b=5$ vs. $d\in[25,50]$) → bootstrap runs are cheap. Bottleneck is Step (iv) (DirectLiNGAM on full $d$-node concrete data), which is unaffected. |

> [!summary]
> Bootstrapping the cheap abstract-discovery step yields a more trustworthy $\hat{M}$ / constraint set, improving downstream concrete recovery — at essentially **no extra time cost**, since total runtime is dominated by the large concrete discovery step, not the small abstract one.

