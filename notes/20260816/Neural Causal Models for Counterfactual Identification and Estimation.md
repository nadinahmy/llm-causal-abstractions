
---
What Is Meant by "Markovian Settings"
- "Markovian" refers to an assumption about **unobserved confounding**: ==a setting is Markovian if there are no hidden common causes shared between two or more observed variables.== Every source of randomness belongs to exactly one variable — nothing "leaks" between variables behind the scenes.

SCM setup: exogenous variables $U$ (unobserved noise) and endogenous variables $V$ (observed), with each $V_i$ computed by $f_{V_i}$ taking some parents in $V$ plus exogenous input from $U$.
Markovian vs. non-Markovian:
- **Markovian** — each exogenous variable $U$ feeds into **at most one** function $f_{V_i}$. No two observed variables ever share the same hidden noise source.
- **Non-Markovian** — some exogenous variable is shared as input across **multiple** functions — i.e. two or more observed variables have a common, unobserved cause.

How is this represented in the causal diagram?
- dashed bi-directed arrows represent (Section 1's notation): $(V_j \dashleftarrow\dashrightarrow V_i)$ whenever $U_{V_i}$ and $U_{V_j}$ are not independent.
- **Markovian diagram** → only directed edges, no dashed bi-directed edges at all
- **Non-Markovian diagram** → can have dashed bi-directed edges, each one flagging "these two variables share some unobserved cause we haven't measured"

Example (Appendix D.4, Fig. 23)
- Studying exercise ($X$) → cholesterol ($Y$), with age ($Z$) as a confounder — older people might exercise less *and* have different cholesterol.
> [!question] Observed vs. unobserved confounder
> - **Age observed** (data collected on it) → age is just a regular variable in $V$, directed edges $Z\to X$ and $Z\to Y$. Still Markovian — nothing hidden.
> - **Age unobserved** (never collected, but still causally affects both) → $X$ and $Y$ now share an unmeasured common cause. This is **non-Markovian** — drawn as a bidirected dashed edge $X \dashleftarrow\dashrightarrow Y$ directly, with age omitted from the diagram entirely.

Markovianity is a common <mark style="background: #ABF7F7A6;">simplifying assumption</mark> elsewhere in the literature because it dramatically simplifies the math:
- Any interventional query $P(y_x)$ can be computed with a simple, always-valid adjustment formula (Eq. 38) — just condition on $X$'s non-descendants
- Distributions factorize cleanly: $P(v) = \prod_V P(v \mid pa_V)$, since each variable's randomness is genuinely independent of every other's
- This is why architectures like Pawlowski et al.'s DeepSCM and Sanchez-Martin et al.'s VACA can use relatively simple structures — ==they lean on Markovianity to sidestep the harder identification problem entirely==

Why does this paper deliberately avoid the assumption?
- Real-world settings almost always have some unmeasured confounder somewhere (their example: smoking → cancer, with unmeasured gene expression affecting both). By handling non-Markovian settings from the start, their method (NCM + causal diagram with bi-directed edges + Algorithm 1) works in the harder, more general, more realistic case — at the cost of a much more involved theoretical treatment.
---
Recall [[Neural Causal Abstractions#Definition 2 G-Constrained Neural Causal Model]]

---
Definition 3 ($G^{(L_i)}$-Consistency)
- "Let $G$ be the causal diagram induced by the SCM $M^*$. For any SCM $M$, $M$ is said to be $G^{(L_i)}$-consistent (w.r.t. $M^*$) if $L_i(M)$ satisfies all layer $i$ equality constraints implied by $G$."

> [!note] Breaking down the pieces
> - $M^*$ — the true, unobserved SCM generating the real-world data
> - $G$ — the causal diagram $M^*$ induces (arrows/bidirected edges)
> - $L_i(M)$ — the collection of layer-$i$ distributions induced by some model $M$ (e.g. $L_1$ = observational, $L_2$ = interventional, $L_3$ = counterfactual)
> - $M$ is $G^{(L_i)}$-consistent if its layer-$i$ distributions obey every equality constraint $G$ implies should hold at layer $i$

But why does this need to be defined per layer?
- "Does $M$ respect $G$?" is **not** a single yes/no question — it's graded, and you have to specify *which layer*.
- ==$G$ implies **different constraints at different layers**==, and satisfying layer-1 constraints doesn't automatically get you layer-2, which doesn't automatically get you layer-3.

Example 3 (from appendix D.2, Fig. 17)

Two graphs — $X \to Z \to Y$ (true) vs. $X \leftarrow Z \to Y$ (alternate) — both imply the **same** layer-1 constraint: $P(y \mid z, x) = P(y \mid z)$.

> [!question] Consistent at L1, inconsistent at L2
> - A model $M$ trained only to match observational data could satisfy this → $G^{(L_1)}$-consistent
> - But $M$ could secretly have the alternate graph's structure underneath → **fails** $G^{(L_2)}$-consistency: the true graph implies extra layer-2 constraints (e.g. $P(x_z)=P(x)$) that the alternate structure would violate, since the causal direction is reversed
>
> $M$ can be consistent with $G$ at layer 1 while inconsistent at layer 2 — "$G$-consistent" is a family of increasingly strict properties, one per layer, not a single property.

Nested circles in Figure 2:
- Definition 3 is exactly what defines the nested sets $\Omega^*(G^{(L_1)}) \supseteq \Omega^*(G^{(L_2)}) \supseteq \Omega^*(G^{(L_3)})$. 
- $\Omega^*(G^{(L_i)})$ is the set of *all* SCMs that are $G^{(L_i)}$-consistent. ==As $i$ increases, this set shrinks — higher layers add *more* constraints on top of lower ones, not different unrelated ones.==

<mark style="background: #BBFABBA6;">Summary : a model can perfectly match a graph's implications observationally while completely failing at the interventional or counterfactual level, that's why we define a Model $M$ respecting a causal graph $G$ in terms of Layers of the PCH.</mark>

#### Important Note:
==An equality constraint is a "distribution A equals distribution B" fact; $C^{(L_i)}$ collects every such fact that's structurally guaranteed by the graph $G$ alone (true across _every_ SCM sharing that structure); and a model is $G^{(L_i)}$-consistent only if it reproduces **every one** of those structurally-forced equalities in its own layer-i distributions — not approximately, not mostly, but exactly, for every single constraint in that (potentially huge) set.==

---
Casino Example:
- Two calculations from the same model $M_0$, shown side by side — the whole point is that they come out **equal**, setting up a later reveal where the same model *disagrees* at layer 3.

Recap of $M_0$

$$
f_X(u_M) = u_M, \qquad f_Y(x, u_M, u_+, u_=, u_-) =
\begin{cases}
u_= & x = u_M \\
u_- & x = (u_M - 1)\%3 \\
u_+ & x = (u_M + 1)\%3
\end{cases}
$$

$U_M$ uniform on $\{0,1,2\}$; $P(U_+=1)=0.6$, $P(U_==1)=0.4$, $P(U_-=1)=0.2$.

Plain-English model:
- Casino "predicts" the customer's machine choice from mood ($U_M$). Predicted machine → average payout ($U_=$). Machine just *before* the prediction → bad payout ($U_-$). Machine just *after* → good payout ($U_+$).

1. First calculation — conditional (observational):
$$
P(Y=1 \mid X=0) = P(U_==1) = 0.4
$$
- $f_X(u_M)=u_M$ — the customer's choice $X$ **is** $U_M$, by construction. Observing $X=0$ reveals $U_M=0$ with certainty. That pins down $f_Y$ to a single branch ("$x=u_M$" → $Y=U_=$), so no averaging is needed.

1. Second calculation — interventional
$$
P(Y_{X=0}=1) = P(U_M{=}0)P(U_={=}1) + P(U_M{=}1)P(U_-{=}1) + P(U_M{=}2)P(U_+{=}1)
$$
- Under intervention $do(X=0)$, $f_X$ is deleted — $X$ is forced to 0 regardless of mood. $U_M$ is still random and unconstrained, so **all three** values of $U_M$ must be considered and averaged.

Checking each branch:
- $U_M=0$: $x{=}0{=}u_M$ → "$x=u_M$" branch → $Y=U_=$
- $U_M=1$: $(1{-}1)\%3=0=x$ → "$x=(u_M{-}1)\%3$" branch → $Y=U_-$
- $U_M=2$: $(2{+}1)\%3=0=x$ → "$x=(u_M{+}1)\%3$" branch → $Y=U_+$

$$
P(Y_{X=0}=1) = \tfrac{1}{3}(0.4) + \tfrac{1}{3}(0.2) + \tfrac{1}{3}(0.6) = \tfrac{1.2}{3} = 0.4
$$

How come both land on 0.4, and why that matters?
- **Conditional** collapses to one term — observing $X$ reveals $U_M$ exactly
- **Interventional** averages over *all three* moods — forcing $X$ reveals nothing about $U_M$ — yet the weighted average coincidentally still equals 0.4 ($\tfrac{1}{3}(0.4+0.2+0.6)=0.4$)

==$M_0$ perfectly reproduces the true model's layer-1 **and** layer-2 behavior== — $P(y_x)=P(y\mid x)$ holds for every value → $M_0$ is $G^{(L_2)}$-consistent. ==Looks like a faithful model if you only have observational/interventional data.==

---
==Why does $M_0$ Fail at Layer 3?==

Layer-3 constraint being tested:
- $P(y_x \mid x') = P(y_x)$
In plain words:
- The counterfactual outcome "what would $Y$ be if we forced $X=x$" shouldn't depend on what $X$ was *actually, factually* observed to be ($x'$). If the true graph is a simple $X \to Y$ arrow with no hidden confounding, this should hold — ==the counterfactual world where $X$ is forced to 0 shouldn't care whether the customer actually picked machine 0, 1, or 2.==

## Computing $P(Y_{X=0}=1 \mid X=2)$ in $M_0$

Asks: given the customer **actually** chose machine 2, what's the probability $Y=1$ would have been true had they instead been **forced** to choose machine 0?

> [!important] Step by step
> 1. **Use the observation to pin down $U_M$**: $f_X(u_M)=u_M$, so observing $X=2$ means $U_M=2$ with certainty
> 2. **Evaluate the counterfactual $Y_{X=0}$ under $U_M=2$** — check $f_Y$'s branches with $x=0$, $u_M=2$:
>    - $x=u_M$? $0=2$? No
>    - $x=(u_M-1)\%3$? $0=1$? No
>    - $x=(u_M+1)\%3$? $0=(2+1)\%3=0$? **Yes** → $Y=U_+$
> 3. **Result**: $P(Y_{X=0}=1 \mid X=2) = P(U_+=1) = 0.6$

Here is the mismatch:
- $P(Y_{X=0}=1 \mid X=2) = 0.6 \quad \neq \quad P(Y_{X=0}=1) = 0.4 \text{ (computed earlier)}$
- These should be equal but aren't
- By the layer-3 constraint $P(y_x \mid x')=P(y_x)$, these two quantities are supposed to match. $0.6 \neq 0.4$ — violation confirmed.

What does this mean in casino terms?
- If the customer chose machine 2, they'd have had a *higher* payout chance (0.6) had they chosen machine 0 instead — but this contradicts the baseline counterfactual probability of 0.4 computed without conditioning on their actual choice.
- **The counterfactual answer changes depending on what the customer actually did**, even though the same hypothetical intervention (force $X=0$) is being asked about both times.

Why is this happening?
- ==$U_M$ is doing double duty==
- $U_M$ determines the customer's *actual* choice ($X=U_M$) **and** simultaneously determines which branch of $f_Y$ applies in *every* hypothetical world — including counterfactual worlds where $X$ is forced elsewhere.
- Even though $M_0$'s diagram looks like an innocent $X \to Y$ arrow (matching layers 1 and 2 perfectly), $U_M$ secretly acts like a hidden confounder between "what $X$ actually was" and "what $Y$ would be under any counterfactual intervention" — something the graph itself doesn't capture, <mark style="background: #FF5582A6;">only exposed by probing layer-3 quantities.</mark>

How do we connect this to the point of Definition 3?
- $M_0$ is genuinely, perfectly $G^{(L_1)}$- and $G^{(L_2)}$-consistent — matches $M^*$ exactly at those layers. But it **fails** $G^{(L_3)}$-consistency: one of the graph's implied layer-3 constraints breaks.
- ==This demonstrates that matching a graph's constraints at lower layers gives **no guarantee** of matching them at higher layers== — exactly the motivation for needing machinery (Theorem 1, G-NCMs) that guarantees $G^{(L_3)}$-consistency *by ==construction==*, rather than models like $M_0$ that look faithful but secretly aren't once probed deeply enough.
Summary for this example : $M_0$ **fails** $G^{(L_3)}$-consistency — a layer-3 constraint like $P(y_x \mid x') = P(y_x)$ breaks down. $M_0$'s internal structure (mood $U_M$ secretly driving both $X$ and $Y$'s branch) ==doesn't match the true mechanism, despite mimicking its lower-layer behavior perfectly.==

---
NCM Expressiveness Beyond the True SCM's Own Form
(clarification following **Theorem 2**)

What is the concern? ---> An NCM is a specific kind of SCM — functions are neural networks (MLPs), noise is uniform $(0,1)$. ==The true SCM $M^*$ that actually generated the data probably isn't built that way at all — different functions, different noise distribution entirely.==
- That mismatch doesn't matter. $M^*$ itself doesn't need to be an NCM. Theorem 2 guarantees: ==no matter what $M^*$ actually looks like internally, as long as it induces graph $G$, there exists **some** G-NCM $\hat M(\theta)$ (built from MLPs and uniform noise) that reproduces $M^*$'s distributions across **all three layers** (L1, L2, L3) exactly.==
- Comes down to expressiveness — MLPs can approximate any function to arbitrary precision, and a uniform $(0,1)$ variable can be transformed (via the probability integral transform) into any other distribution.
- So even though $\hat M$'s internal machinery looks nothing like $M^*$'s, ==it can still be tuned to match $M^*$'s external behavior — its full output distributions — at every layer.==

This is what allows treating the NCM as a valid **proxy** for the true SCM. You never get to see $M^*$ directly, but Theorem 2 says: whatever $M^*$ is, some NCM exists that behaves identically to it, all the way up to counterfactuals — <mark style="background: #ABF7F7A6;">so training an NCM to match observable data is a legitimate way to stand in for the unreachable true model.</mark>

---
#### Definition 4 (Neural Counterfactual Identification)

> Given true SCM $M^*$, causal diagram $G$, and available data $Z = \{P(\mathbf{V}_{z_k})\}$ (observational/interventional). Query $P(y_* \mid x_*)$ is **neural identifiable** from $\Omega(G)$ and $Z$ iff:
> $$P^{\hat M_1}(y_*\mid x_*) = P^{\hat M_2}(y_*\mid x_*)$$
> for **every** pair of G-NCMs $\hat M_1, \hat M_2 \in \Omega(G)$ that both match $M^*$ on all of $Z$.

- If *every* NCM that fits the observed data the same way also agrees on the query's answer, the query is identifiable — its value is forced by $G$ + data, regardless of which compatible NCM you happened to train.

#### Theorem 3 (Counterfactual Graphical-Neural Equivalence / Dual ID)

> $Q$ is neural identifiable from $\Omega(G)$ and $Z$ **if and only if** $Q$ is identifiable from $G$ and $Z$ in the classical (symbolic) sense.
-  Bridges the new "neural" identifiability (Def. 4) to established non-neural results — searching within the NCM space gives the *same* identifiability verdict as classical symbolic methods (e.g. do-calculus). Justifies performing identification via NCM optimization (Alg. 1) instead of symbolic derivation, with no loss of correctness.

---
#### Algorithm 1 - Neural ID
```
Input : query Q = P(y*|x*), L2 datasets Z(M*), causal diagram G
Output: P^M*(y*|x*) if identifiable, FAIL otherwise

1  M̂ ← NCM(V, G)
2  θ*_min ← argmin_θ P^M̂(θ)(y*|x*)  s.t. Z(M̂(θ)) = Z(M*)
3  θ*_max ← argmax_θ P^M̂(θ)(y*|x*)  s.t. Z(M̂(θ)) = Z(M*)
4  if P^M̂(θ*_min)(y*|x*) ≠ P^M̂(θ*_max)(y*|x*) then
5      return FAIL
6  else
7      return P^M̂(θ*_min)(y*|x*)
```

 - **Line 1** — construct a G-constrained NCM (Def. 2); this is the ==search space== $\Omega(G)$
 - **Lines 2–3** — the core move: search for two parameter settings of the *same* NCM, both constrained to match the real data ($Z(\hat M(\theta)) = Z(M^*)$). Among all valid settings: $\theta^*_{\min}$ makes query $Q$ as *small* as possible, $\theta^*_{\max}$ as *large* as possible
 - **Lines 4–7** — compare $Q$ under the two extremes. Disagree → FAIL (non-identifiable). Agree → that shared value is the correct answer

Why do we minimize AND maximize, not just train one model?
- Directly implements Definition 4
- $Q$ is identifiable only if **every** NCM matching the data on $Z$ agrees on $Q$. Checking one arbitrary NCM tells you nothing about whether other equally-valid NCMs would disagree.
 **The trick**: instead of checking infinitely many NCMs, check only the two **extremes**. ==If the min possible value of $Q$ (over all valid $\theta$) equals the max, then $Q$ must be *constant* across the entire feasible set== — every NCM matching the data gives the same answer, without checking them individually. If min ≠ max, that alone witnesses two data-matching NCMs that disagree — exactly what non-identifiability means.

Why is this approach efficient?
- Definition 4 technically quantifies over *every pair* of NCMs in $\Omega(G)$ — <mark style="background: #FF5582A6;">intractable to check exhaustively</mark>.
- Min/max optimization collapses that infinite verification into solving two optimization problems.
- Another angle : Same idea as checking whether a continuous function is constant on an interval by comparing its min and max, rather than checking every point.
- Without requiring $\theta^*_{\min}$ and $\theta^*_{\max}$ to still match $Z(M^*)$, the search is meaningless — you'd just find an NCM that trivially sets $Q$ to 0 or 1 by ignoring the data. The constraint keeps the search within "==models actually plausible given what's observed==" — only within that set does agreement/disagreement on $Q$ mean anything.

This min/max structure is exactly what Corollary 2 proves correct: $Q$ is truly identifiable iff the procedure doesn't FAIL, and when it doesn't, the returned value equals the true $P^{M^*}(y_*\mid x_*)$. Any NCM contradicting the optimality of $\theta^*_{\min}$/$\theta^*_{\max}$ (inducing a $Q$ outside that range) would violate their being the true min/max over the whole feasible set.

Summary : Minimizing and maximizing simultaneously turns an intractable "check that infinitely many NCMs all agree" requirement into a tractable "find the two extreme cases and see if they collapse to one value" check.

---
Simultaneously Fitting Multiple Datasets in Z

1. The easy case
	- Single dataset (e.g. just observational $P(V)$) → training is standard: sample from NCM, compare to real data, adjust parameters to reduce the difference. One dataset, one thing to fit.
2. The harder case
	- $Z$ can contain **several** datasets at once — e.g. observational $P(V)$ **and** interventional $P(V_x)$. The NCM must satisfy **all** of them simultaneously, using the *same* underlying parameters $\theta$ (same functions $\hat F$, same noise $P(\hat U)$).


Why isn't this obviously easy?
- These datasets aren't independent training targets to fit one after another — they're all generated by the *same* single underlying model, just observed under ==different conditions== (no intervention vs. $do(X=x)$). Improving the fit to one dataset can throw off the fit to another, since changing $\hat F$ or $P(\hat U)$ affects **every** layer's distributions at once, not just the one currently being focused on.
- No straightforward "just add more data" recipe exists — need a training procedure that balances fitting all datasets in $Z$ together, without one improving at the expense of another.
---
Algorithm 2 — Sampling a Counterfactual from an NCM

![[IMG-20260816175341649.png|311]]
- Answers: "if I want $m$ samples from the counterfactual distribution $P(Y_* \mid x_*)$, how do I actually get them out of a trained NCM?"
> [!note] Step by step
> 1. **Draw random noise** — sample $\hat u$ from the NCM's noise distribution $P(\hat U)$ (uniform $(0,1)$ for each exogenous variable)
> 2. **Check the condition** — using that *same* $\hat u$, evaluate what $X_*$ would be. If it doesn't equal $x_*$, discard this sample and try again
> 3. **If it matches, evaluate the query** — compute $Y_*$ using that *same* noise value $\hat u$
> 4. **Repeat** until $m$ accepted samples are collected
- **rejection sampling**: repeatedly draw random noise, keep only draws where the "condition" part comes out right, and for those kept draws, read off the "query" part — since both were computed from the *same* underlying noise, they're automatically consistent with each other.

<mark style="background: #FF5582A6;">Super important:</mark>
Why does it matter to use the same û for both?
- Because $X_*$ and $Y_*$ are both computed as functions of the *same* sampled $\hat u$, you get a properly joint, coherent sample — not two unrelated pieces glued together.
- This is exactly how counterfactuals are supposed to work: <mark style="background: #ABF7F7A6;">imagine one full "world" (one setting of the noise), then ask what multiple things would be true in that one world simultaneously</mark> — including what $Y$ would be under an intervention, given that the same world's noise also happens to produce $X=x_*$ naturally.
---
Important definitions:
1. $P^{M^∗}(V_{z_k}​​)$ is: **the full joint probability distribution over all our variables, as generated by the true model $M^∗$, under one specific interventional setting $z_k$​.**

2. $P^{M^∗}(X,Y)$ being "exact" means we know, with total precision:
$P(X=0,Y=0)=0.64,P(X=0,Y=1)=0.16,P(X=1,Y=0)=0.04,P(X=1,Y=1)=0.16$
	- We know every single cell of the joint probability table, exactly, with no error or uncertainty. Not "approximately 0.64 based on some samples" — genuinely, exactly 0.64.

3. "A collection of" these"
	- $Z = \{P^{M^*}(\mathbf{V}_{z_k})\}_{k=1}^{\ell}$ just means: we're not necessarily given just **one** such exact distribution — you might be given **several**, indexed by $k = 1, \ldots, \ell$. Each one corresponds to a different interventional setting.

4. Example:
	- $Z = \{P^{M^*}(\mathbf{V}),\ P^{M^*}(\mathbf{V}_{X=0}),\ P^{M^*}(\mathbf{V}_{X=1})\}$ — three exact distributions: the plain observational one, plus what everything looks like under $do(X{=}0)$, plus what everything looks like under $do(X{=}1)$. That's $\ell=3$ distributions bundled together as the available data $Z$
---
Optimizing the Query — The Second Challenge of Algorithm 1
- Want NCM parameters $\theta$ to push $Q = P(y_* \mid x_*)$ as large (or small) as possible — but $Q$ is a probability, not directly differentiable unless turned into a loss.

The trick:
> [!note] Three steps
> 1. Get samples from the NCM via **Algorithm 2** — call this $\hat Q$ (model's current samples for $Y_*$ under condition $x_*$)
> 2. Compare $\hat Q$ against the *target*: **1** if maximizing the probability of $y_*$, **0** if minimizing
> 3. Penalize the model based on how far its samples are from that target

> [!example]
> Maximizing $P(Y{=}1)$: sample from NCM → $\hat y = 0.6$ (sigmoid output, real number in $[0,1]$, not hard 0/1). Want this close to 1, so penalize the gap: $(1-0.6)^2$. Minimizing this squared-error loss pushes $\hat y$ toward 1 — i.e. pushes the model toward assigning higher probability to $Y{=}1$.

- $\mathbb{D}_Q$ = whatever distance function measures this gap. Squared error is one option, but the paper uses **log loss** in real experiments, since variables are binary — same reasoning as cross-entropy in DAS, better-behaved for binary outcomes than squared error.

Summary : To maximize/minimize a query, sample the model's current output for that query via Algorithm 2, then use a loss that pushes those sampled outputs as close as possible to the extreme value (1 or 0) being optimized toward — turning "maximize a probability" into an ordinary loss you can back-propagate through.

---
Equation 5 — The Full Training Objective
$L\left(\hat M, \{\hat P^{M^*}(\mathbf{V}_{z_k})\}_{k=1}^\ell\right) = \left(\sum_{k=1}^\ell \mathbb{D}_P\left(\hat P^{\hat M}_{z_k}, \hat P^{M^*}_{z_k}\right)\right) \pm \lambda\, \mathbb{D}_Q(\hat Q, Q)$
- Glues together "fit the data" and "push the query toward its extreme" into a single number to minimize.

1. Fit the data
 $\sum_{k=1}^\ell \mathbb{D}_P(\hat P^{\hat M}_{z_k}, \hat P^{M^*}_{z_k})$
- For every dataset in $Z$, measure how different the NCM's own samples are from the real data, and sum all those differences. Same $\mathbb{D}_P$ divergence as before (measured via a GAN discriminator in practice) — one term per dataset, summed, so the model is pressured to match **all** of them at once.

1. Push the query
 $\pm \lambda\, \mathbb{D}_Q(\hat Q, Q)$
-  Penalize the gap between the model's current sampled query value $\hat Q$ and the target extreme $Q$ (0 or 1). ==The $\pm$: subtract if *maximizing* the query, add if *minimizing* — so minimizing the overall $L$ pushes $Q$ the intended direction.==

What λ does?
- A weight controlling how much to care about pushing the query vs. fitting the data.
- Per Appendix B: $\lambda$ starts **high** and **decreases** during training — ==early on, focus on pushing the query to its extreme; later, as $\lambda$ shrinks, faithfully fitting the real data takes over and dominates.==

Equation 5 = (how badly the model fails to match the real data) plus-or-minus (how far the query still is from its target extreme, weighted by $\lambda$) — one combined loss that simultaneously trains the NCM to be data-consistent **and** pushes the query toward its min/max, exactly what Algorithm 1's lines 2–3 need.

---
Algorithm 3 — Training Model (GAN-NCM)
- Implements the optimization from **Equation 5**, running the min-model and max-model training **side by side** in the same loop — this is the concrete procedure behind Algorithm 1's lines 2–3.
```
Input : Data {P̂^M*(V_zk)}, query Q, causal diagram G,
        # Monte Carlo samples m, regularization constant λ,
        learning rate η, training epochs T

1  M̂ ← NCM(V, G)                        // from Def. 2
2  Initialize parameters θ_min and θ_max
3  for t ← 1 to T do
4      L_min ← 0, L_max ← 0
5      for k ← 1 to ℓ do
6          P̂_min(V_zk) ← M̂(θ_min).sample(V_zk, n_k)   // via Alg. 2
7          P̂_max(V_zk) ← M̂(θ_max).sample(V_zk, n_k)
8          L_min ← L_min + D_P(P̂_min(V_zk), P̂^M*(V_zk))
9          L_max ← L_max + D_P(P̂_max(V_zk), P̂^M*(V_zk))
10     Q̂_min ← M̂(θ_min).sample(Y*, m)
11     Q̂_max ← M̂(θ_max).sample(Y*, m)
12     L_min ← L_min − λ D_Q(Q̂_min, Q)
13     L_max ← L_max + λ D_Q(Q̂_max, Q)
14     θ_min ← θ_min − η∇L_min
15     θ_max ← θ_max − η∇L_max
```

Two separate models, one loop
- $\theta_{\min}$ and $\theta_{\max}$ are **two independent copies** of the same NCM architecture, trained in parallel every epoch — one being pushed toward the smallest possible query value, the other toward the largest. This is exactly the min/max pair from **Algorithm 1**, now with an actual training procedure attached.

Step by step:

> [!note] Lines 1–2 — setup
> Build the G-constrained NCM architecture (Def. 2), initialize two separate parameter sets $\theta_{\min}$, $\theta_{\max}$.

> [!note] Lines 5–9 — the data-fitting term (Eq. 5's Piece 1)
> For each dataset $k$ in $Z$: draw samples from both models via **Algorithm 2**, compare each to the corresponding real (empirical) dataset using divergence $\mathbb{D}_P$ (a GAN discriminator in practice), and accumulate this into $L_{\min}$ and $L_{\max}$ separately. Summing over $k$ pressures both models to match **every** dataset in $Z$ simultaneously — ==the "multiple datasets" challenge discussed earlier.==

> [!note] Lines 10–13 — the query-pushing term (Eq. 5's Piece 2)
> Sample the query itself ($\hat Q_{\min}$, $\hat Q_{\max}$) from each model via Algorithm 2. Then adjust the accumulated loss:
> - $L_{\min}$ gets $-\lambda\,\mathbb{D}_Q(\hat Q_{\min}, Q)$ — **subtracted**, since this model is being pushed toward the *minimum*
> - $L_{\max}$ gets $+\lambda\,\mathbb{D}_Q(\hat Q_{\max}, Q)$ — **added**, since this model is being pushed toward the *maximum*
>
> This is exactly the $\pm\lambda\,\mathbb{D}_Q$ term from Equation 5, split across the two models with opposite signs.

> [!note] Lines 14–15 — gradient descent
> Standard update: nudge each model's parameters in the direction that reduces its own loss. Both models train simultaneously but independently — they never share gradients, only the same data and query definition.

<mark style="background: #FFF3A3A6;">Summary : Algorithm 3 is Equation 5's loss, applied twice per epoch — once to a model being pushed toward the query's minimum, once toward its maximum — using Algorithm 2's sampling to estimate both the data-fit term and the query term, so that after training, comparing the two models' query values tells you whether the query is identifiable (Algorithm 1).</mark>

---
#### Summary on this paper's goals and experiments:

The big question the paper is trying to answer :
- **Can you use neural networks to answer "what if" questions (counterfactuals) about a system, when you only have regular data (observations) and maybe some experiments — not the true underlying causal model itself?**

The assumption the authors add: a causal diagram
- Instead of assuming nothing, they assume you have a **causal diagram GG** — just the qualitative structure of what causes what (including which variables might share hidden confounders, drawn as bidirected edges). Not the exact mechanisms, not the exact noise — just the shape of the causal story.

The tool the authors build in this paper: the NCM
- A **Neural Causal Model** is just an SCM where the functions are neural networks. When you constrain an NCM to match a specific graph G (a "G-NCM"), two things are true simultaneously — (proving these two things is the paper's first big contribution):
- **Theorem 1**: any G-NCM automatically satisfies _every_ counterfactual-level constraint that G implies (this is "$G^{(L_3​)}$-consistency" — the thing Definition 3 formalized)
- **Theorem 2**: G-NCMs are still expressive enough to represent the true model's behavior at all three layers, even though the true model probably isn't itself built from neural networks

**Algorithm 1** answers the identification problem cleverly: train two NCMs, one pushed to minimize the query, one pushed to maximize it, both constrained to still fit the real data. If they agree, the query is identifiable (and that's the answer). If they disagree, it's not — and trying to estimate it would be meaningless.

The engineering problem: making Algorithm 1 actually trainable
- **Algorithm 2** — how do you get a sample of a counterfactual quantity out of a trained NCM? (Rejection sampling on shared noise $\hat{u}$.)
- **Equation 4** — how do you turn those samples into an actual probability estimate? (Simple counting/ratio, Monte Carlo style.)
- **Equation 5 / Algorithm 3** — how do you train the two competing NCMs to simultaneously (a) fit the real data and (b) push the query to its extreme? (A GAN-based loss, with $λ$ decaying over training to shift emphasis from "chase the extreme" to "stay faithful to data.")

