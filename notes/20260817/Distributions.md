True-vs-empirical-distributions

---
Core distinction between a true/exact distribution and a finite empirical sample — recurring concept across causal inference papers

# True Distribution vs. Empirical Distribution

## The core idea

> [!important]
> A **true distribution** is the *actual, exact* pattern of probabilities governing a random variable — the real thing, not an estimate of it. It's a fixed mathematical fact, completely independent of how much data you've collected.

## The coin-flip intuition

> [!example]
> A fair coin's **true distribution**: $P(\text{heads})=0.5$, $P(\text{tails})=0.5$ — exact, by the coin's physical nature. Doesn't change no matter how many times you flip it, and exists even if you never flip it at all.
>
> Flip it 10 times → maybe 6 heads, 4 tails → **empirical** result $\hat P(\text{heads})=0.6$. Not the true distribution — just what happened in that one batch. Flip another 10 times → might get 4 heads instead. The empirical number bounces around; the true distribution (0.5) never does.

In the NCM paper's notation:

|                            | Symbol                     | What it is                                                                                                                                                                              |
| -------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **True distribution**      | $P^{M^*}(\mathbf{V})$      | ==Fixed, exact property of the true SCM $M^*$ itself== — baked into its structure (functions $F$, noise $P(U)$). Never depends on sample size.                                          |
| **Empirical distribution** | $\hat P^{M^*}(\mathbf{V})$ | ==Computed by counting a *finite sample* of actual observations drawn from $M^*$. Approximates the true one, gets closer as the sample grows, never guaranteed exact for finite data.== |
## Why you never get to see the true distribution directly

> [!warning] The core epistemic problem
> In real life you can never observe $P^{M^*}$ directly — only samples drawn from it (your dataset). The true distribution is what you're always *trying to infer*, never directly handed. This is why methods (Algorithm 3, etc.) are built around finite empirical datasets $\hat P^{M^*}$, never around assuming access to exact $P^{M^*}$.

Example — X,Y with n=10 sample:

Binary $X,Y$. True distribution (exact, from $M^*$):

| Outcome | True $P^{M^*}$ | Empirical $\hat P^{M^*}$ (n=10) |
|---|---|---|
| $X{=}0,Y{=}0$ | 0.40 | 0.40 |
| $X{=}0,Y{=}1$ | 0.10 | 0.10 |
| $X{=}1,Y{=}0$ | 0.10 | 0.20 |
| $X{=}1,Y{=}1$ | 0.40 | 0.30 |

> [!tip] Reading the gap
> The true bars are fixed, perfect numbers — only knowable with infinite data. The empirical bars are what 10 real draws actually gave — close, but noisy (here $X{=}1,Y{=}0$ is off by 10 points purely from sampling luck). Draw a *different* 10 samples and the empirical bars would land differently — the true bars never move.

> [!important]
> - This gap is exactly what divergence terms like $\mathbb{D}_P$ (Algorithm 3) measure — but always between two *empirical-style* quantities: the model's own generated samples vs. the given (already-noisy) dataset. You never get to compare against the clean true bars directly, because you never have them.
> - **The gap shrinks as sample size $n$ grows.** With 10,000 samples instead of 10, empirical bars would sit almost exactly on the true ones — ==exactly why estimation error (MAE) decreases as sample size increases in this paper's experiments, and why more Monte Carlo samples ($m$) give tighter query estimates too.==
## One-line takeaway

> [!tip]
> "True" = exact, fixed, unobservable directly. "Empirical" = what finite real data actually shows — a noisy approximation that converges to the true distribution only as sample size grows toward infinity.

---
# Numerical Walkthrough — Algorithm 2 + Equation 4

Example query: $P(Y_{X=0}=1 \mid X=2)$, extracted from a **trained** GAN-NCM via sampling.

Setup: Trained NCM $\hat M(\theta)$, noise $\hat U$ = a handful of uniform $(0,1)$ variables. Want $m=10$ samples.

Step 1 — draw random noise repeatedly
- Sample fresh $\hat u_j \sim P(\hat U)$ for each attempt — just uniform random numbers, e.g. $\hat u_1=(0.83,0.12,0.47)$.

Step 2 — check the condition (rejection sampling)
- Run each $\hat u_j$ through the NCM **normally** (no intervention), read off $X$. Keep only draws where $X=2$; discard the rest.

| $j$ | $\hat u_j$ | $X(\hat u_j)$ | Kept? |
|---|---|---|---|
| 1 | (0.83, 0.12, 0.47) | 2 | ✓ |
| 2 | (0.05, 0.91, 0.33) | 0 | ✗ |
| 3 | (0.61, 0.44, 0.78) | 2 | ✓ |
| 4 | (0.29, 0.15, 0.90) | 1 | ✗ |
| 5 | (0.94, 0.02, 0.11) | 2 | ✓ |

Keep drawing until $m=10$ samples satisfying $X=2$ are collected.

## Step 3 — evaluate the counterfactual with ==the SAME noise==

> [!important]
> For each **kept** $\hat u_j$, run it through the **mutilated submodel** where $X$ is forced to 0 ($do(X{=}0)$), using that *same* $\hat u_j$. Read off $Y_{X=0}$.

Final set of 10 kept samples' $Y_{X=0}$ outcomes:
$$1, 0, 1, 1, 0, 1, 1, 0, 1, 1$$

## Step 4 — apply Equation 4

$$
P^{\hat M}(Y_{X=0}=1 \mid X=2) \approx \frac{\sum_j \mathbb{1}[Y_{X=0}(\hat u_j)=1,\ X(\hat u_j)=2]}{\sum_j \mathbb{1}[X(\hat u_j)=2]}
$$

> [!tip] Denominator is trivial here
> Because Algorithm 2 already only *kept* samples satisfying $X{=}2$, the denominator is just $m=10$ by construction. Only need to count how many of the 10 kept samples also have $Y_{X=0}=1$: **7 out of 10**.

$P^{\hat M}(Y_{X=0}=1 \mid X=2) \approx \frac{7}{10} = 0.7$

Final answer:
- **0.7** — extracted purely by sampling and counting; no explicit probability formula ever evaluated.
## Why this procedure works

> [!note]
> - **Rejection step** (discarding $X\ne2$ samples) makes the denominator trivially $m$ — more efficient than drawing a fixed batch and computing numerator/denominator separately per Equation 4's general form
> - ==**Using the same $\hat u_j$** for both the condition-check and the counterfactual evaluation makes this a coherent joint sample — asking "what would $Y$ be under $do(X{=}0)$" specifically about the same hypothetical worlds where $X$ happened to actually equal 2, not some unrelated random world==
> - **Unconditional queries** (e.g. $P(Y_{X=1}=1)$, no "$\mid$") skip Step 2's rejection entirely ($X_*=\emptyset$) — just average the outcome across every sampled $\hat u_j$ directly

