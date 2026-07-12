"M′ is L2-consistent with M" :  
	$L_2(M') = L_2(M)$
- Recall $L_2(M)$ = "the whole bucket of interventional distributions M can produce." So this equation just says: for every possible intervention,  and M′ produce the exact same probability table.
Why this breaks down for $V_L$ vs. $V_H$​?
- Example: $M_L$ is defined over **low-level variables** $V_L$ (e.g. carbs, fat, protein, dish, BMI), while $M_H$ is defined over **high-level variables** $V_H$​ (e.g. calories, dish, BMI-category).
- These are **different variable spaces** — $M_L$'s distribution table has columns for carbs/fat/protein; $M_H$​'s distribution table has a column for "calories" instead. We cannot line up the two tables row-by-row and ask "do these match?" — the columns don't even correspond to the same things. $L_i(M_L) = L_i(M_H)$  itself is undefined, not just false.
- **Fix**: use $\tau$ to translate $M_L$​'s output into $V_H$​'s terms first, then compare — this translated comparison is what Q-τ consistency formalizes.
---
**Motivating Problem: Not All Low-Level Quantities Have High-Level Counterparts**

- $\tau$ was built (Definition 4) to work ==**cluster by cluster**== — it takes a whole cluster like $\{C,F,P\}$ and turns it into one high-level variable, $Z$.
- Problem: what if you want to ask a counterfactual question about ==**just carbs ($C$) alone**==, without fat and protein?
- You can't. $\tau$ only knows how to translate **whole clusters at a time**. Asking about a *partial piece* of a cluster is undefined — the abstraction was never built to handle that.
- **Conclusion:** only questions that align exactly with whole clusters (or unions of whole clusters) can be translated through $\tau$. This section defines *which* low-level counterfactual questions qualify.

---
**Symbol 1: $\mathbf{Y}_{L,*}$ — a general low-level counterfactual query**

$$\mathbf{Y}_{L,*} = (Y_{L,1}[x_{L,1}], Y_{L,2}[x_{L,2}], \ldots)$$

- Same object as $Y^*$ from **Definition 1** — a bundle of "potential outcome" statements, possibly across different hypothetical worlds, for the **low-level** model.
- **$Y_{L,i}[x_{L,i}]$** : "the outcome variable(s) $Y_{L,i}$, in the hypothetical world where we intervened and forced $X_{L,i} = x_{L,i}$." (bracket = intervention, same as Def. 1)
- Subscript **$L$** : marks "this is happening at the ==low level==," to distinguish from the high-level version built later.
- The $*$ : a generic marker meaning "the whole bundle of these terms," not a specific number.

> Example: $\mathbf{Y}_{L,*}$ could be "what would BMI have been if diet had been healthy, **and** what would BMI have been if diet had been unhealthy" — two potential-outcome statements bundled together.

---
**Symbol 2: The Crucial Restriction — $Y_{L,i}$ and $X_{L,i}$ Must Be Unions of Clusters**

> "Each $Y_{L,i}$ and $X_{L,i}$ must be unions of clusters from $\mathbb{C}$ (i.e. $Y_{L,i} = \bigcup_{C \in \mathbb{C}'} C$ for some $\mathbb{C}' \subseteq \mathbb{C}$)"

- $Y_{L,i}$ : whichever *set of variables* is the outcome in term $i$.
- $X_{L,i}$ : whichever *set of variables* is intervened on in term $i$.
- $\mathbb{C}' \subseteq \mathbb{C}$ : ==some subset of the clusters== already defined (e.g. $\mathbb{C}' = \{C_1, C_3\}$).
- $\bigcup_{C \in \mathbb{C}'} C$ : "union of all clusters in that subset" — glue those clusters' variables together.
- **Requirement:** $Y_{L,i}$ must *exactly equal* one of these unions — built entirely out of **whole clusters**, ==never a fragment.==

**Allowed vs. not allowed:**
- ✅ $Y_{L,i} = \{B\}$ — the whole cluster $C_1$
- ✅ $Y_{L,i} = \{B, D\}$ — union of clusters $C_1$ and $C_3$
- ✅ $Y_{L,i} = \{C, F, P\}$ — the whole cluster $C_2$
- ❌ $Y_{L,i} = \{C\}$ alone — just a fragment of cluster $C_2$, breaks $\tau$'s cluster-by-cluster design

---
**Symbol 3: Why This Restriction Makes $\tau$ Well-Defined**

> "such that $\tau(Y_{L,i})$ and $\tau(X_{L,i})$ are well-defined (i.e. $\tau(Y_{L,i}) = (\tau_C(C) : C \in \mathbb{C}')$)"

- Since $Y_{L,i}$ is guaranteed to be a **clean union of whole clusters**, $\tau$'s per-cluster subfunctions ($\tau_C$, from Definition 4) can be applied to each cluster in $\mathbb{C}'$ separately, then stacked together.
- ==$\tau(Y_{L,i}) = (\tau_C(C) : C \in \mathbb{C}')$ : "run the translator $\tau_C$ on each cluster $C$ in the chosen subset $\mathbb{C}'$, and list the outputs side by side."==

> Example: if $Y_{L,i} = \{B, D\}$ (clusters $C_1$ and $C_3$), then $\tau(Y_{L,i}) = (\tau_{C_1}(B), \tau_{C_3}(D)) = (B_H, D_H)$.

---
**Symbol 4: Building the Full High-Level Counterpart, $\mathbf{Y}_{H,*}$**

$$\mathbf{Y}_{H,*} = \tau(\mathbf{Y}_{L,*}) = \big(\tau(Y_{L,1}[\tau(x_{L,1})]),\ \tau(Y_{L,2}[\tau(x_{L,2})]),\ \ldots\big)$$

Two separate translations happen inside each term:

1. **The intervention value gets translated**: $x_{L,1}$ becomes $\tau(x_{L,1})$ — its high-level counterpart.
	- e.g. intervening with $D=$"pizza" becomes $\tau(D=\text{pizza}) = D_H=\text{unhealthy}$
2. **The outcome variable/value gets translated**: the whole term $Y_{L,1}[\ldots]$ gets passed through $\tau$ too.

> ==$\mathbf{Y}_{H,*}$ is just: take the entire low-level counterfactual statement, and rewrite every piece of it — both the "what we intervened on" and "what we're asking about" — in high-level language.==

> Example: "what would BMI have been, had diet been pizza" becomes "what would BMI-category have been, had diet-category been unhealthy."

---
**Symbol 5: The "Bucket" of Low-Level Values Matching One High-Level Answer**

$$\mathcal{D}_{\mathbf{Y}_{L,*}}(\mathbf{y}_{H,*}) = \{\mathbf{y}_{L,*} \in \mathcal{D}_{\mathbf{Y}_{L,*}} : \tau(\mathbf{y}_{L,*}) = \mathbf{y}_{H,*}\}$$

- $\mathbf{y}_{H,*}$ : one *specific* high-level counterfactual outcome (e.g. "BMI-category = overweight, had diet-category been unhealthy" — one concrete answer).
- $\mathcal{D}_{\mathbf{Y}_{L,*}}$ : the domain — the set of *all possible* low-level counterfactual outcomes.
- $\mathcal{D}_{\mathbf{Y}_{L,*}}(\mathbf{y}_{H,*})$ : restrict that domain to only the low-level outcomes that, once translated through $\tau$, land exactly on the chosen high-level answer $\mathbf{y}_{H,*}$.

> Example: many different exact BMI numbers (26.1, 27.8, 31.4, ...) all translate to the same high-level bucket "overweight." $\mathcal{D}_{\mathbf{Y}_{L,*}}(\mathbf{y}_{H,*})$ is ==exactly the list of all those low-level values that get grouped into that one high-level answer== — same intravariable clustering idea from Definition 3, now applied to a whole **counterfactual outcome**.

---
**Why This Was All Necessary → Connects to Definition 5**

- This passage exists to make precise: ==*"If I want the probability of a high-level counterfactual answer, add up the probabilities of every low-level scenario that would produce that same high-level answer."*==
- Exactly equal to the sum in **Definition 5's Equation 6**: 
	$$\sum_{\mathbf{y}_{L,*} \in \mathcal{D}_{\mathbf{Y}_{L,*}}(\mathbf{y}_{H,*})} P(\mathbf{Y}_{L,*} = \mathbf{y}_{L,*})$$
- This passage is exactly what **defines the summation set** used there.

---
**Summary**
> Before comparing a low-level counterfactual question to a high-level one, we need: (1) a rule for *which* low-level questions have valid high-level translations (only whole-cluster unions), (2) a precise way to translate a full counterfactual statement through $\tau$, and (3) a way to identify *every* low-level answer that collapses into one high-level answer — so their probabilities can be summed.

---
**Definition 5 : Q-τ Consistency**

- The formal payoff of everything set up in the previous section (unions of clusters, $\mathbf{Y}_{H,*}$, $\mathcal{D}_{\mathbf{Y}_{L,*}}(\mathbf{y}_{H,*})$).
- Answers the question: ==**"When can a high-level SCM $M_H$ be trusted to give the right answer to a specific counterfactual question, compared to the true low-level SCM $M_L$?"**==

---
**The Setup**

- $M_L$, $M_H$ : SCMs over $V_L$ and $V_H$ respectively.
- $\tau$ : the constructive abstraction function (Def. 4), built from clusters $\mathbb{C}$ and $\mathbb{D}$.
- **Low-level quantity of interest**:
	$$Q = \sum_{\mathbf{y}_{L,*} \in \mathcal{D}_{\mathbf{Y}_{L,*}}(\mathbf{y}_{H,*})} P(\mathbf{Y}_{L,*} = \mathbf{y}_{L,*})$$
	- In words: ==add up the probability of **every low-level outcome that maps to the same high-level answer** $\mathbf{y}_{H,*}$== (this summation set is exactly what was built in the previous passage).
- **Its high-level counterpart**:
	$$\tau(Q) = P(\mathbf{Y}_{H,*} = \mathbf{y}_{H,*})$$
	- Just: "the probability of that one high-level answer, as computed directly from the high-level model."

---
**The Consistency Condition**

$$\sum_{\mathbf{y}_{L,*} \in \mathcal{D}_{\mathbf{Y}_{L,*}}(\mathbf{y}_{H,*})} P^{M_L}(\mathbf{Y}_{L,*} = \mathbf{y}_{L,*}) = P^{M_H}(\mathbf{Y}_{H,*} = \mathbf{y}_{H,*})$$

- **$M_H$ is Q-τ consistent with $M_L$** if this equality holds.
- In plain English: ==**"the total probability mass of all the low-level scenarios that collapse into this one high-level answer, according to $M_L$, must equal the probability $M_H$ assigns to that same high-level answer directly."**==
- This is basically a **push-forward measure** check — translating $M_L$'s distribution through $\tau$ and seeing if it lands exactly on $M_H$'s distribution.

---
**Extending to a Whole Layer**

- If $M_H$ is Q-τ consistent with $M_L$ for **every** query $Q$ of this form drawn from $L_i(M_L)$, then $M_H$ is said to be ==**$L_i$-τ consistent**== with $M_L$.
- This lets us talk about matching on:
	- **$L_1$-τ consistency** — matches on all observational quantities
	- **$L_2$-τ consistency** — matches on all interventional quantities
	- **$L_3$-τ consistency** — matches on all counterfactual quantities (the strongest form)

---
**Why This Matters**

- $M_H$ can only be treated as a trustworthy **abstraction** of $M_L$ **for the specific quantities where they are τ-consistent** — not automatically for everything.
- Matching on lower layers does **not** guarantee matching on higher layers.
- **Proposition 1** connects this back to prior literature: if $M_H$ is **$L_3$-τ consistent** with $M_L$, this coincides with the older, more rigid notion of a "constructive τ-abstraction" (Beckers & Halpern 2019) — but ==Q-τ consistency is more flexible since it lets you check consistency **query by query**, rather than requiring the entire models to align at once.==

---
**One-line summary**
> ==Definition 5 gives a precise, checkable condition for "the high-level model gives the right answer to this counterfactual question" — by requiring that the summed probability of all matching low-level scenarios equals the probability the high-level model assigns directly.==

---
**Definition 6 : Abstract Invariance Condition (AIC)**

- Answers a *different* question than Definition 5: not "does $M_H$ match $M_L$ on some query," but ==**"is it even *possible* for *any* $M_H$ to be a valid abstraction of $M_L$, given the clusters we chose?"**==
- This is a **property of $M_L$ and $\tau$ alone** — it doesn't reference any candidate $M_H$ at all.

---
**The Core Problem It's Addressing**

- $\tau$ maps many different low-level values to the **same** high-level value (that's the whole point of intravariable clustering — e.g. many (carbs,fat,protein) combos all map to "$Z=200$").
- But what if two low-level values that get **squashed together** by $\tau$ actually behave **differently** in terms of their downstream effects?
- If so, ==no high-level model can correctly summarize that cluster with a single value== — information necessary for correct predictions would be lost by the abstraction itself, no matter how good $M_H$ is.

---
**Breaking Down the Formal Statement**

$$\tau_{C_i}\Big(\big(f^L_V(pa^{(1)}_V, u_V) : V \in C_i\big)\Big) = \tau_{C_i}\Big(\big(f^L_V(pa^{(2)}_V, u_V) : V \in C_i\big)\Big)$$

- **Setup**: take two low-level value-tuples $v_1, v_2 \in \mathcal{D}_{V_L}$ such that $\tau(v_1) = \tau(v_2)$ — i.e., ==they get mapped to the **same high-level value**== (they're in the same intravariable bucket).
- **$pa^{(1)}_V$ and $pa^{(2)}_V$** : the parent-values corresponding to $v_1$ and $v_2$ respectively — i.e., "the inputs feeding into variable $V$'s mechanism," under scenario 1 vs. scenario 2.
- **$f^L_V(pa_V, u_V)$** : run $V$'s actual low-level mechanism, given its parents and its own noise $u_V$ — this produces $V$'s *next* value.
- **$\big(f^L_V(pa^{(1)}_V, u_V) : V \in C_i\big)$** : do this for *every* variable $V$ inside cluster $C_i$, and collect the outputs — i.e., "run the whole cluster's mechanisms forward one step, starting from scenario 1's parent-values."
- **$\tau_{C_i}(\ldots)$** : then translate *that whole output tuple* into its high-level counterpart.
- **The condition**: doing this starting from $v_1$'s parent-values must give the ==**same high-level result**== as doing it starting from $v_2$'s parent-values — **for every possible value of the noise $u$**, and for every cluster $C_i$.

---
**In Plain English**

> ==If two low-level situations look identical from the high-level perspective (same $\tau$ value), then **everything downstream** that depends on them must *also* look identical from the high-level perspective — regardless of the hidden noise.==

- Intuitively: two values in the same intravariable cluster must have the **same functional effect** in the high-level setting.
- This is exactly what makes intravariable clustering meaningful as an **invariance**: if $\tau$ groups (10,5,8) and (5,5,13.75) together as "$Z=200$" calories, the AIC demands that *whatever effect calories has on downstream variables* must be the same for both — otherwise "$Z=200$" isn't really telling the whole story, and collapsing them into one value was invalid.

---
**Why This Matters — Connecting Forward**

- **Proposition 2** shows the AIC is *exactly* the condition needed: an appropriate $M_H$ exists (that is $L_3$-τ consistent with $M_L$) **if and only if** $M_L$ satisfies the AIC w.r.t. $\tau$.
- So the AIC is the ==**dividing line**== between:
	- Clusters that are "safe" to abstract (AIC holds → some valid $M_H$ exists)
	- Clusters that **throw away causally relevant information** (AIC fails → no $M_H$, however cleverly built, can be a faithful abstraction)
- From this point on, the paper **assumes the AIC holds**, and treats it as a prerequisite check before attempting to build/learn an abstraction at all.

---
**One-line summary**
> ==Definition 6 (AIC) says: if two low-level scenarios are indistinguishable at the high level, they must *stay* indistinguishable after being run forward through the system — otherwise the chosen clustering is throwing away information no abstraction could recover.==

---
#### Definition : Identifiability
> Identification (in ordinary causal inference, without abstractions) asks: _"given a causal graph and some data, is there only ONE possible value the query could take — or could two different SCMs, both consistent with the graph and data, give different answers?"_ If there's only one possible answer, the query is **identifiable**; if two SCMs could disagree, it's **not identifiable**.

---
**Definition 8 : Abstract Identification (τ-ID)**

- Answers: ==**"Given only the coarse graph $\mathcal{G}_\mathbb{C}$ and the data we actually have, which queries can we pin down a *definite* answer for — even though we never see the true low-level model $M_L$?"**==
- Direct extension of ordinary causal identification, but spanning ==**two variable spaces at once**== ($V_L$ and $V_H$), connected through $\tau$.

---
**Recap: Why This Comes After Definition 7 (C-DAG)**

- Def. 7 gave us a **coarser graph** $\mathcal{G}_\mathbb{C}$ over clusters instead of raw variables (e.g. over $\{D_H, Z, B_H\}$ instead of $\{R,D,C,F,P,B\}$) — much easier to specify by hand.
- Def. 8 asks the natural next question: given only this coarse graph + some data, what can we actually determine?

---
**The Big Idea (Ordinary Identification, Recapped)**

- Ordinary identification asks: ==*"given a causal graph and some data, is there only ONE possible value the query could take — or could two different SCMs, both consistent with the graph and data, disagree?"*==
- One possible answer → **identifiable**. Disagreement possible → **not identifiable**.
- Definition 8 does the exact same thing, but now the two SCMs being compared live in ==**different variable spaces**== ($V_L$ vs $V_H$), connected via $\tau$.

---
**Breaking Down the Setup, Piece by Piece**

- **$\mathbb{Z} = \{P(V_{L[z_k]})\}_{k=1}^{\ell}$** : the "shopping list" of **available data** — e.g. observed $P(V_L)$, or a handful of real interventional studies.
- **$\Omega_L$** : the space of ==**every conceivable SCM**== over $V_L$ — every possible way reality could actually work at the low level, totally unconstrained.
- **$\Omega_H$** : same idea, but for every conceivable high-level SCM over $V_H$.
- **$\Omega_L(\mathcal{G}_\mathbb{C})$** : narrow $\Omega_L$ down to only the SCMs whose *true* causal diagram, once clustered, gives exactly $\mathcal{G}_\mathbb{C}$ — i.e. $\mathcal{G}_\mathbb{C}$ is a valid C-DAG *for* $M_L$.
- **$\Omega_H(\mathcal{G}_\mathbb{C})$** : narrow $\Omega_H$ down to only the SCMs whose own causal diagram **is** $\mathcal{G}_\mathbb{C}$ directly.
- ==This is the graph doing its job as an assumption== — we stop considering *every* possible model, and only keep the ones compatible with the causal structure we assumed.

---
**The Actual Definition of τ-ID**

> "$Q$ is τ-ID from $\mathcal{G}_\mathbb{C}$ and $\mathbb{Z}$ **if and only if** for every $M_L \in \Omega_L(\mathcal{G}_\mathbb{C})$, $M_H \in \Omega_H(\mathcal{G}_\mathbb{C})$ such that $M_H$ is $\mathbb{Z}$-τ consistent with $M_L$, $M_H$ is also $Q$-τ consistent with $M_L$."

**In plain English:**
> ==Take *any* possible low-level model $M_L$ compatible with our graph, and *any* possible high-level model $M_H$ compatible with our graph. If $M_H$ happens to match $M_L$ on the data we actually have (𝕫-τ consistent), then — no matter which $M_L, M_H$ pair we picked — $M_H$ is *guaranteed* to also match $M_L$ on the query $Q$ we care about.==

- Key move: we never see the true $M_L$. All we know is the graph + observed data $\mathbb{Z}$.
- τ-identifiability says: ==**even in total ignorance of the true $M_L$**, as long as our high-level model fits the available data, it is *automatically forced* to also give the right answer for $Q$== — because *every* possible low-level model consistent with graph + data would agree on $Q$ anyway. No room for disagreement → no need to know the true $M_L$.

---
**What Non-Identifiability Means, Concretely**

> "τ-nonidentifiability implies there exist $M_L$ and $M_H$ such that they match in both $\mathcal{G}_\mathbb{C}$ and $\mathbb{Z}$, yet still do not match in $Q$."

- Just the negation: ==you can find at least one pair of models where everything checkable (graph, available data) matches perfectly, but $Q$ still comes out differently.==
- Since you can never tell from the data alone which model is "true," you cannot trust *any* single answer to $Q$ — genuinely ambiguous, however well your high-level model fits the visible data.

---
**Concrete Intuition (Nutrition Example)**

- Suppose we only observe $P(V_L)$ and the C-DAG $D_H \to Z \to B_H$. We want $Q = P(B_{H, D_H=\text{unhealthy}})$ — causal effect of unhealthy diet on BMI-category.
- **No hidden confounding** between diet and calories → causal effect is nailed down uniquely by observed data → **τ-ID**: any $M_L, M_H$ fitting the data agree on $Q$.
- **Hidden common cause** (e.g. genetics affecting both diet-craving and metabolism) not shown in the graph → two different $M_L$'s could produce *identical* observed data but imply *different* true effects → **non-identifiable**. No cleverness in building $M_H$ can rescue this — the ambiguity is baked into what data + graph can distinguish.

---
**Why This Matters?**

- Deliberately phrased over *both* spaces ($\Omega_L$ and $\Omega_H$) at once because this is a genuinely new, ==**cross-level**== identification problem.
- Payoff (**Theorem 1**, right after): this cross-level problem turns out to be ==**exactly equivalent**== to ordinary identification carried out entirely within the high-level space $V_H$ — meaning in practice you never need to reason about $\Omega_L$ at all, only the much smaller, more tractable $\Omega_H$.

---
**One-line summary**
> ==Definition 8 (τ-ID) says: a query is safely answerable across abstraction levels if *every* low-level/high-level model pair consistent with our assumed graph and available data is forced to agree on that query — meaning we don't need to know the true low-level model to trust the high-level answer.==

---
**Theorem 1 : Dual Abstract ID**

- Answers: ==**"Do I really need to reason about every possible hidden low-level truth, or can I just work with the simple high-level diagram instead?"**==
- A **reduction**: turns a hard, cross-level identification problem into an ordinary, already-solved, single-level identification problem.

---
**The Statement**

$$Q \text{ is τ-ID from } \mathcal{G}_\mathbb{C} \text{ and } \mathbb{Z} \iff \tau(Q) \text{ is ID from } \mathcal{G}_\mathbb{C} \text{ and } \tau(\mathbb{Z})$$

---
**Two Ways to Ask "Is This Computable?"**

- **Way 1 — the hard way (τ-ID, Definition 8):**
	- Imagine ==**every possible way the true low-level world could work**== (every hidden mechanism, every possible confounder).
	- Check: no matter which of these true worlds we're actually in, does the high-level model always give the correct answer?
	- Impossible to check directly — infinitely many hidden possibilities.

- **Way 2 — the easy way (ordinary ID):**
	- ==Forget the low-level world exists.== Just treat the simple high-level diagram $\mathcal{G}_\mathbb{C}$ as the *whole story*.
	- Ask the standard, well-established causal inference question: "given this diagram and this data, can I compute the answer with a formula?" (do-calculus / classical ID algorithms — a **solved problem**.)

---
**What the Theorem Guarantees**

> ==**These two questions always give the same answer.**==

- If the easy way says "yes, computable" → the hard way is *guaranteed* to also say yes.
- And vice versa.
- **Practical payoff:** you never have to do the hard way. Just draw the simple high-level diagram, run standard identification on it, and the result is *guaranteed correct* with respect to the real, unknown, low-level world underneath — ==even though you never looked at that low-level world at all.==
---
**Why This Works (Intuition)**

- $\mathcal{G}_\mathbb{C}$ (the C-DAG) does **double duty**:
	- it's a legitimate abstraction of the true low-level graph (by Def. 7's construction), **and**
	- it can be treated as an ordinary causal diagram in its own right, over $V_H$.
- Nothing relevant to τ-translated queries is lost by working only with $\mathcal{G}_\mathbb{C}$ over $V_H$ — so swapping "reason about $M_L$" for "reason about $\mathcal{G}_\mathbb{C}$ as a normal graph" costs nothing.

---
**Where This Leads Next**

- **Corollary 1**: τ-ID ⟺ **Neural-ID** (identification using NCMs) on $V_H$.
- Since $|V_H| \ll |V_L|$, training an NCM over the small high-level space is **tractable** — unlike training one directly over $V_L$ (e.g. raw pixels).
- This tractable procedure is exactly what **Algorithm 2** implements: build a $\mathcal{G}_\mathbb{C}$-NCM over $V_H$, fit it to $\tau(\mathbb{Z})$, and check if $\tau(Q)$ comes out the same regardless of max/min fitting — all without ever needing $M_L$.

---
**One-line summary**
> ==Theorem 1 turns an intractable cross-level identification problem (reasoning about every possible low-level model) into an ordinary, already-solved identification problem (reasoning about one graph over the small high-level space) — with the exact same answer guaranteed either way.==

---
**Visualizing the NCM: What It Physically Is**
![[IMG-20260712204728121.png|391]]
- An NCM is **not one big network** — it's ==**one small neural net per variable**==, wired together exactly like the causal graph says.
- For each variable, the architecture stacks three pieces:
	1. **Noise (gray)** : a plain random number, sampled from a simple uniform distribution — no structure baked in.
	2. **Neural net (teal)** : the trainable part. Takes in the noise *plus* the output of that variable's parents, and produces a value. Gradient descent adjusts the weights inside this box.
	3. **Output (purple)** : the resulting value, which becomes an input to the next variable's neural net downstream.
- "Training the NCM" = adjusting the weights inside the teal boxes until, when fed random noise, the outputs statistically match the real observed data.
- This is exactly **Definition 2 (G-NCM)** made visual — the graph dictates the wiring; the neural nets are the trainable mechanisms.

---
**Algorithm 2 : NeuralAbstractID**

- Purpose: ==**identify and estimate a query across abstraction levels using NCMs**== — implements Corollary 1 (τ-ID ⟺ Neural-ID on $V_H$) in practice.

**The Core Trick: Train Two Rival Copies**
- Build **two separate NCMs**, both with:
	- the exact same architecture (same graph $\mathcal{G}_\mathbb{C}$, same wiring)
	- fit to the exact same available data ($\tau(\mathbb{Z})$)
- The only difference is the **training objective**:
	- **Copy 1** : fit the data, and *among all weight settings that fit it equally well*, push the query $\tau(Q)$ as **high** as possible.
	- **Copy 2** : fit the data, but push $\tau(Q)$ as **low** as possible.

**Why Two Copies?**
- There may be ==**multiple different sets of weights**== that all fit the observed data equally well, but **disagree** on the query we actually care about — this is exactly the non-identifiability scenario from Definition 8 / Theorem 1.
- Training one network to push the answer up and another to push it down is a practical way to **search for the worst-case disagreement**.

**The Final Check**
- Compare the two extreme answers (max-trained vs. min-trained):
	- ==**If they match**== → no room left for disagreement. Every possible model consistent with the data would give this same answer → **identifiable**, trust the value.
	- ==**If they don't match**== → concrete proof of ambiguity. The data genuinely doesn't pin down a unique answer, no matter how the model is trained → **FAIL**, not identifiable.

**Algorithm Steps, Mapped to the Idea**
1. Build $\tau$ from clusters $\mathbb{C}, \mathbb{D}$ (Def. 4).
2. Construct a $\mathcal{G}_\mathbb{C}$-NCM over $V_H$ (Def. 2) — the shared architecture.
3. Train copy 1 (θ*min) and copy 2 (θ*max), both constrained to match $\tau(\mathbb{Z}(M_L))$, one minimizing and one maximizing $\tau(Q)$.
4. If the two resulting values of $\tau(Q)$ differ → return FAIL.
5. If they agree → return that shared value as $Q(M_L)$.
---
**One-line summary**
> ==An NCM is a small neural net per variable, wired to match the causal graph. Algorithm 2 trains two copies of this same architecture with opposing goals (max vs. min the query) on the same data — if they land on the same answer, the query is safely identifiable; if not, the data is ambiguous and the algorithm fails honestly rather than guessing.==

---
**Proposition 5 : When Is the AIC Automatically Satisfied?**

- Recap of the problem: checking whether a *chosen* clustering satisfies the AIC (Def. 6) normally requires knowing $M_L$'s actual mechanisms — which we usually **don't have**.
- Proposition 5 gives an escape hatch:

	> ==**$M_L$ is guaranteed to satisfy the AIC w.r.t. $\tau$ if and only if $\mathbb{D}_{C_i} = \mathcal{D}_{C_i}$ for all $C_i \in \mathbb{C}$.**==

**What $\mathbb{D}_{C_i} = \mathcal{D}_{C_i}$ Means**

- $\mathcal{D}_{C_i}$ : the **full domain** of a cluster — every possible value-combination the low-level variables in that cluster could take.
- $\mathbb{D}_{C_i}$ : the **intravariable clustering** we chose — how we grouped those values together.
- Setting them equal means: ==**don't group anything together at all**== — every individual low-level value gets its own singleton bucket.
- In other words, $\tau_{C_i}$ isn't *compressing* anything — it's just ==**relabeling**== each low-level value with a new high-level name, one-to-one (a bijection).

**Why This Guarantees the AIC**

- If $\tau_{C_i}$ is bijective (never merges two distinct low-level values into the same high-level value), the AIC's premise — "two values $v_1, v_2$ map to the same high-level value" — **can never actually occur** (except trivially, $v_1 = v_2$).
- No case to check → the condition **holds automatically**.
- ==**You buy safety by giving up compression.**==

**The Catch**
- This bijective choice keeps $|V_H|$ the same size as $|V_L|$ — no distinctions are thrown away.
- But you're **not required to keep $V_H$ looking like $V_L$** — you can choose *any* relabeling of the same size.
- Meaning: you're free to pick whichever **representation** of the data is most useful for your task, as long as it's a faithful one-to-one repackaging — rather than being stuck with the raw low-level variables (e.g. pixels) themselves.
---
==**Definition 9 : Representational NCM (RNCM)**==
**Motivating Problem**
- For low-dimensional data (calories, BMI), a human can hand-specify sensible intravariable clusters.
- For **images** (thousands of pixels), you ==**cannot hand-write**== a rule saying which raw pixel-arrays count as "the same digit." No way to manually enumerate that clustering.
- Solution: instead of *hand-specifying* $\tau$, **learn it** — using the safe bijective strategy from Proposition 5.

**The Definition**
> A Representational NCM is a pair $\langle \hat{\tau}, \hat{M} \rangle$, where $\hat{\tau}(v_L; \theta_\tau)$ is a **trainable, parameterized** function mapping $V_L \to V_H$, and $\hat{M}$ is an NCM defined over $V_H$.
- **$\hat{\tau}$** : instead of a fixed hand-written rule (like $Z = 4C+9F+4P$), this is now itself ==**a neural network with its own trainable weights $\theta_\tau$**==. The clustering is *learned*, not decided.
- **$\hat{M}$** : same idea as before — an NCM built over whatever high-level space $\hat{\tau}$ ends up mapping into.
- **$\mathcal{G}_\mathbb{C}$-RNCM** : adds the structural constraint — $\hat{\tau}$ splits into per-cluster subfunctions $\hat{\tau}_{C_i}$ (mirroring Def. 4), and $\hat{M}$ respects the graph $\mathcal{G}_\mathbb{C}$, same as an ordinary $\mathcal{G}_\mathbb{C}$-NCM.

**How Training Works (Colored MNIST Example)**
1. Train $\hat{\tau}$ like an ==**autoencoder**==: one network compresses an image into a small representation, a second network tries to reconstruct the original image from it.
2. Good reconstruction ⇒ evidence $\hat{\tau}$ is (close to) **bijective** — no information thrown away.
3. This is exactly the condition **Proposition 5** says guarantees the AIC holds automatically — no manual verification needed.
4. Separately train the NCM $\hat{M}$ over the resulting representation space (via **Algorithm 2**), letting it learn causal structure in the clean, learned space instead of raw pixels.
---
**Why This Whole Chain Matters**
- **Def. 6 (AIC)** : what makes an abstraction *valid*.
- **Prop. 5** : a specific, ==always-safe== way to satisfy that validity condition (bijective relabeling).
- **Def. 9 (RNCM)** : how to **learn** that bijective relabeling automatically via representation learning, when hand-specifying it (as in the nutrition example) is infeasible — exactly the situation with image data in the paper's experiments.

---
**One-line summary**
> ==Proposition 5 shows that a purely bijective (non-compressing) clustering always satisfies the AIC for free; Definition 9's RNCM exploits this by *learning* such a bijective mapping with an autoencoder-style network, letting abstractions be discovered automatically for high-dimensional data like images instead of hand-specified.==

---
Summary of the experiments performed:
- Goal: ==empirically evaluate the effects of using abstractions in causal inference tasks==.
- Two experiments, testing the theory at two very different scales:
	1. **Nutritional Study** — small, low-dimensional, hand-specified clusters.
	2. **Colored MNIST** — high-dimensional image data, clusters learned via **RNCM**.

---
**5.1 Nutritional Study**
- Recap of the toy example (Ex. 1): $V_L = \{R, D, C, F, P, B\}$, clusters $\mathbb{C} = \{D_H=\{D\}, Z=\{C,F,P\}, B_H=\{B\}\}$.
- **Query**: $Q = P(B_{D=d} \geq 25)$ — the ==causal effect of diet on the probability of being overweight==.
- **Why abstraction helps here**: $R$ and $D$ are 32-dimensional one-hot vectors, and the others are real-valued — high-dimensional enough that estimating $Q$ directly in $V_L$ is difficult. Working in the abstracted $V_H$ space (using $\tau$ from $\mathbb{C}$ and binary $\mathbb{D}$) is far more tractable.
- **Comparison**: NCMs trained in the original low-level space $V_L$ vs. in the abstracted space $V_H$ (via Alg. 2).

**Results (Fig. 6)**
- **Fig. 6a — identifiability gap**: since $Q$ is identifiable, the gap between the max-trained and min-trained NCM (Alg. 2's two copies) should shrink to ~0.
	- ==The abstracted approach (blue) converges quickly to a near-zero gap.==
	- Competing approaches (raw data = red, normalized data = yellow) **fail to close the gap** even after many training iterations.
- **Fig. 6b — estimation error**: mean absolute error (MAE) vs. dataset size.
	- ==The abstracted approach achieves significantly lower error== than working directly with raw or normalized low-level data.
- **Takeaway**: abstraction isn't just theoretically valid — it makes both **identification** and **estimation** *practically* easier, even in a fairly small/simple setting.
---
**5.2 Colored MNIST Digits**
- Dataset: colorized MNIST digits. Each image $I$ has a digit label $D$ and color label $C$, related via C-DAG $\mathcal{G}_\mathbb{C}$ (Fig. 7a): $C \to D$, $C \to I$, $D \to I$ (color and digit both cause the image; color also influences digit — a built-in **confound**).
- Color and digit are **highly correlated** in the data (e.g. 0s tend to be red, 5s tend to be cyan) — this spurious correlation is the thing a good causal model needs to see through.

**Three Competing Approaches**
1. **Conditional GAN** — no causal structure, pure pattern-matching.
2. **GAN-NCM** — causally structured, but trained directly on **raw pixels**.
3. **GAN-RNCM (this paper)** — causally structured, but trained on a **learned representation** (via Def. 9's autoencoder-style $\hat\tau$) instead of raw pixels.

**The Four Query Types Tested (Fig. 5)**

| Column                | Query               | What it tests                                                                                   |
| --------------------- | ------------------- | ----------------------------------------------------------------------------------------------- |
| Image samples         | —                   | basic generation ability                                                                        |
| $P(I \mid D=0)$       | observational (L1)  | reproduce natural correlation (mostly red 0s expected)                                          |
| $P(I_{D=0})$          | interventional (L2) | ==break the spurious correlation== — 0s of **all colors** expected                              |
| $P(I_{D=0} \mid D=5)$ | counterfactual (L3) | disentangle: **keep** this individual's color (cyan, from being a 5), **change** the digit to 0 |

**Results**
- **Column 1-2 (generation, observational)**: all three methods do fine — easy cases, just reproducing what's in the data.
- **Column 3 (interventional)**: this is where methods diverge.
	- Conditional GAN and GAN-NCM ==**struggle to disentangle color from digit**== — samples remain color-biased (still mostly red 0s) because they can't cleanly separate the confound.
	- GAN-RNCM correctly produces 0s of **all colors**.
- **Column 4 (counterfactual)**: hardest test — requires holding one attribute fixed for a specific instance while varying the other.
	- GAN-RNCM correctly produces **cyan 0s** (preserves the individual's original color, changes only the digit) — closely matching ground truth.
	- The other methods fail to properly disentangle.
- **Def. 6 (AIC)**: abstraction is only valid if no causally-relevant information is thrown away.
- **Prop. 5 + Def. 9 (RNCM)**: gives a way to *learn* such a valid, information-preserving abstraction automatically (via autoencoder-style training), instead of requiring a human to hand-specify it.
- **Figure 5 shows**: ==being causally structured (like GAN-NCM) is *not enough* on its own for high-dimensional data.== Without first learning a clean, disentangled representation, a causal model still struggles to separate spurious correlations from real causal structure when working directly on raw pixels.
- **The RNCM's representation-learning step is what makes causal reasoning actually *work* in practice**, not just in principle, once data moves beyond small, interpretable variables.

>[!Conclusion] Conclusion:
>The experiments show abstraction pays off at both ends of the spectrum: in the small nutrition example, it makes identification and estimation dramatically more reliable; in the high-dimensional Colored MNIST example, it's the *only* thing that lets a causally-structured model actually disentangle spurious correlations (color vs. digit) and get interventional/counterfactual queries right.

