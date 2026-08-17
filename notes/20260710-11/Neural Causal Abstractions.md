By : Kevin Xia, Elias Bareinboim

---
- Open problem the paper discusses : how to best leverage ==abstraction theory== in real-world ==causal inference tasks==, where the true model is not known, and limited data is available
- **Quote from the paper :** we focus on a family of causal abstractions constructed by clustering variables and their domains, redefining abstractions to be amenable to individual causal distributions.
	**1. "Constructed by clustering variables and their domains"** : 
	This refers to the two ways the authors in this paper simplify a "messy" low-level model (like pixels in an image) into a clean high-level model (like the concept of a "dog").
	- **Intervariable Clustering (Clustering Variables):** You take several individual low-level variables and group them into one high-level variable. For example, in an image, you cluster thousands of individual **pixels** into a single variable called **"Object"**. From the paper's nutrition example, the low-level variables **Carbohydrates, Fat, and Protein** are clustered into a single high-level variable called **"Calories"**.
	- **Intravariable Clustering (Clustering Domains):** You take the different _values_ those variables can have and group them into single high-level categories. ==For example, many different combinations of (carbs, fat, protein) might all result in exactly 200 calories; those different low-level "tuples" are clustered into one high-level value: "200".== This captures **invariances** in the data, where different low-level states are "functionally identical" at a higher level.
	**2. "Redefining abstractions to be amenable to individual causal distributions"**
	- the "relaxed" part of the paper's theory. Everything I've seen till now was **"declarative"**, meaning I had to have both the full low-level and high-level models already in hand to see if they matched.
	- What's the problem with this? In the real world, we almost never have the "ground truth" low-level model; we only have **data** (like images or observational records).
	- **The Solution:** Instead of requiring the _entire_ structure of both models to perfectly align, this paper focuses on **"Q-τ Consistency"**. This means they check if a **specific query** or "individual distribution" (like "What happens to BMI if I change the diet?") matches across the two levels.
---
**1. Generative Models as "==Representations of Processes=="**
	Most AI systems are built on **generative models**, which are mathematical attempts to ==replicate the "underlying processes" that generated the data==. Ex.: a model of the economy is a representation of millions of individual purchasing decisions. Goal of a generative model is to create a digital "proxy" of reality.
	
**2. Standard Generative Models: "Joint Density" (Layer 1)**
	A **standard generative model** focuses on ==**joint density**, which is a way of saying it learns how different variables are likely to appear together.==
	- **The Limit:** restricted to the first layer of the Pearl Causal Hierarchy, known as **"Seeing"** / Observation.
	- Ex.: A standard model might learn that "Rain" and "Wet Sidewalks" often happen at the same time, but it cannot tell _why_ they are related or which one causes the other.

**3. Causal Generative Models: Interventions and Counterfactuals (Layers 2 & 3)**
	A **causal generative model** goes further by modeling the =="internal gears" of the system==. This is typically formalized using **Structural Causal Models (SCMs)**, which describe the actual mechanisms and functions that produce the data.
	==These models allow the AI to move up the Ladder of Causation:==
	- **Interventions ("Doing" - Layer 2):** The model can predict what will happen if you **change** a variable. Ex.: it can predict if the sidewalk will still be wet if you hold up an umbrella, effectively "changing" the original process to see the result.
	- **Counterfactuals ("Imagining" - Layer 3):** The model can answer "What if?" questions about the past. It can imagine a world where the past was different—for instance, "Would the sidewalk be dry now if it hadn't rained an hour ago?".

---
Quote from paper : In typical tasks of causal inference, the goal is to obtain a quantity from a higher layer when given data only from lower layers (e.g. inferring interventional quantities from observational data). It is understood that ==this is generally impossible without additional assumptions== since higher layers are underdetermined by lower layers.

---
Definition 1 : Layer 3 valuation
- Formally defines how a Structural Causal Model produces the highest level of information in the Pearl Causal Hierarchy : **Counterfactuals**.
**Joint Counterfactuals**
- SCM M induces a set of distributions, L3​(M), which represent **joint counterfactual events**.
- **Notation:** P(Y1​[x1​],Y2​[x2​],…) represents the probability of several different outcomes ==happening under different hypothetical scenarios simultaneously==.
- small example : allows you to ask, "What is the probability that the sidewalk would be dry if it hadn't rained, **and** wet if it had rained?".
**Mathematical Mechanism (Equation 1)**
- The definition calculates these probabilities by integrating over the **exogenous variables (U)**,[specific context of a situation]:
![[IMG-20260711191456080.png|521]]
Super important : Breaking down the formula into little pieces
- The Indicator Function (1): checks if the predicted counterfactual outcome $Y_i[x_i]$ matches a specific value $y_i$ for a given individual context $u$.
- The Integration: By summing these across all possible contexts (U), the model determines the overall probability of the event.
==**The "Mutilation" Procedure**==
- To evaluate these counterfactuals ---> model uses a **mutilation procedure**.
- In the hypothetical world where we intervene on a variable X we "mutilate" the model by **deleting its natural causal mechanism** $F_X$ and replacing it with a constant value $x$.
- Rest of the system's mechanisms remain the same, allowing the model to trace how that one hypothetical change ripples through the other variables.
---
**Hierarchy of Layers**
- In Definition 1, they establish a ==**containment hierarchy**== ---> lower layers are just special, simplified cases of this Layer 3 definition:
	- **Layer 2 (Interventional):** This is a subset of Layer 3 where you only look at one hypothetical scenario at a time (all $x_i$ are the same).
	- **Layer 1 (Observational):** This is the most basic subset where you perform **no interventions** at all (the set of changed variables is empty, $X_i$ = $∅$).
---
Some confusing notation :
1. Li​(M)**: The "Bucket" of Causal Layers**
	- The notation Li​(M) refers to the entire set of possible probability distributions that a specific Structural Causal Model (M) can generate for a given layer (i) of the causal ladder.
	- L1​(M) **(Observational):** Everything the model knows about **"Seeing"**—the patterns and correlations in raw data where no variables are changed.
	- L2​(M) **(Interventional):** Everything the model knows about **"Doing"**—what happens to the system when we actively intervene and change a variable (the "mutilation procedure").
	- L3​(M) **(Counterfactual):** Everything the model knows about **"Imagining"**—hypothetical "What if?" scenarios where we can imagine a different past for a specific individual context.
2. **Z**: The "Shopping List" of Queries**
	- Z is defined as a specific set of **quantities** or "questions" you want to ask from **Layer 2**.
	- Instead of looking at _every_ possible intervention the model could perform, Z is a curated list of specific interventions we care about for our task (e.g., "What happens if we change the digit to 0?" or "What happens if we increase the calorie count?").
3. Z(M)**: The "Answer Key"**
	- While Z is the list of questions, Z(M) represents the actual **numerical answers** (the distributions) those questions produce when they are run through a specific model M.
	- If we have two different models—a low-level one (ML​) and a high-level one (MH​)—we use this notation to see if their answer keys match.
---
![[IMG-20260711195616598.png|150]]
 **The "Shopping List" Formalized**
- represents ==a collection of interventional (Layer 2) probability distributions== — i.e., a batch of "what happens if we do X" answers, not just one.
- **V** : denotes ==the entire set of endogenous variables== in the model (all the variables you care about, bundled together — not just one).
- **Subscript $z_k$** : signals an **intervention**.
	- some variables have been "mutilated" — their natural mechanism ripped out and replaced with a fixed constant $z_k$ .
	- So $V_{z_k}$ = "V, in the hypothetical world where we forced certain variables to equal $z_k$."
- $P(V_(z_k)​)$ : the **interventional distribution** itself.
	- In words: "the probability distribution over all the variables in V, *after* performing intervention $z_k$."
	- This is a single Layer 2 quantity — one entry on the shopping list.
- **{ … }_{k=1}^ℓ​** : curly braces = ==a set==.
	- The subscript/superscript ($k=1$ to $ℓ$) just means: "collect $ℓ$ of these," i.e. don't just take one intervention — take a whole list of $ℓ$ different ones, indexed $k = 1, 2, \dots, ℓ$.
	- In other words, "the set containing $P(V_{z_1})$, $P(V_{z_2})$, … all the way up to $P(V_{z_ℓ})$."

---
**Contextual Meaning**
- Collectively, this set is named **Z** in the paper.
- ==Z = the "available data" / "shopping list" of causal queries== that a researcher has access to, and uses to train or evaluate an abstraction.
- **Z(M)** = the **"answer key"** — plug a specific SCM $M$ into each of these queries, and you get the actual numerical distributions that $M$ produces for each one.
	- This lets you compare: does the low-level model's answer key $Z(M_L)$ match the high-level model's answer key $Z(M_H)$ (once passed through $\tau$)?
	- This comparison is exactly what shows up in **Algorithm 2**, where $\mathbb{Z}(M_L)$ ("Layer 2 datasets") is given as *input*, so the algorithm can check whether the high-level GC-NCM is consistent with it.

---
>[!Summary] **Summary**
> This notation is the mathematical way of saying: ==**"Here is a list of ℓ different things we can 'do' to the system, and the resulting data for each of those actions."**==
> - $\mathbb{Z}$ itself is just the **list of questions** ("what happens under intervention z1?", "...under z2​?", etc.) — it doesn't contain any actual numbers yet.
> 
>- $\mathbb{Z}(M)$ (plugging in a specific SCM M) is what turns this into the **"answer key"** — actually running each of those ℓ interventions through model M and getting real distributions out.
>
>- This is exactly what feeds into **Algorithm 2** as the available data: you're told the answer key for the low-level model, $\mathbb{Z}(M_L)$, and the algorithm searches for a high-level NCM whose own answer key (after passing through τ) matches it.

---
#### Small refresher on Probability distributions

- A probability distribution is just a list of every possible outcome, paired with how likely each one is. 2 main components:
	1. All the possible things that could happen
	2. A number (between 0 and 1) attached to each one, saying how likely it is. All the numbers must add up to 1
--- 
Example: rolling a die
- Possible outcomes: 1, 2, 3, 4, 5, 6.
- The probability distribution is something like this:

| outcome | probability |
| ------- | ----------- |
| 1       | 1/6         |
| 2       | 1/6         |
| 3       | 1/6         |
| 4       | 1/6         |
| 5       | 1/6         |
| 6       | 1/6         |

This table **is** the distribution ---> "what can happen, and how often."

Same example with more than one variable at once:
- Instead of one die, we roll two dice: a red one (R) and a blue one (B). Now an "outcome" is a _pair_ of numbers, like (R=3, B=5). The distribution is a bigger table — one row for every possible pair:

|R|B|probability|
|---|---|---|
|1|1|1/36|
|1|2|1/36|
|...|...|...|
|6|6|1/36|

- This is written as P(R,B) — "the joint distribution over R and B." Same idea, just over a bigger table because there are two variables instead of one.

Connecting this to $P(\mathbf{V})$ :
- $\mathbf{V}$ is _all_ the variables in the model bundled together (like R and B bundled together above, but potentially many more variables).
- $P(\mathbf{V})$ just means: "the giant table listing every possible combination of values that all the variables could take, together with the probability of each combination."
- If $\mathbf{V}$ = {R, D, C, F, P, B} (from the nutrition example earlier — restaurant, dish, carbs, fat, protein, BMI), then $P(\mathbf{V})$ is one enormous table with a row for every possible combination of (restaurant, dish, carbs, fat, protein, BMI), and a probability attached to each row.

Now: what changes with an intervention, $P(\mathbf{V}_{\mathbf{z}_k})$?
- It's the **exact same kind of table** — same structure, same idea (every possible combination of values, with a probability for each) — but the numbers in the table are **different**, because we've forced some variable to a fixed value before generating the rest.

Example:
- P(X,Y) (no intervention) might look like:

|X (drug)|Y (recovers)|probability|
|---|---|---|
|0|0|0.3|
|0|1|0.2|
|1|0|0.1|
|1|1|0.4|

- This reflects what naturally happens — including the fact that maybe sicker people are more likely to take the drug (X and Y are correlated through other factors).
- $P(X, Y)_{X=1}$ — i.e. $P(\mathbf{V}_{z})$ with the intervention z = "force X=1" — is a **different table**:

| X (drug) | Y (recovers) | probability |
| -------- | ------------ | ----------- |
| 1        | 0            | 0.3         |
| 1        | 1            | 0.7         |

- Now X is _always_ 1 (we forced it), and the probabilities for Y reflect _only_ the causal effect of the drug, with none of the "who tends to take the drug" bias mixed in. It's a genuinely different distribution — different numbers — because the underlying scenario generating the data has changed.

So, finally, what is $\{P(\mathbf{V}_{z_1}), P(\mathbf{V}_{z_2}), \ldots\}$?
- It's just: **a collection of several such tables — one full table of outcome-probabilities per intervention you're interested in.**
- $P(\mathbf{V}_{z_1})$ = the full table of "what combinations of values are how likely, if we perform intervention #1"
- $P(\mathbf{V}_{z_2})$ = a _different_ full table, for intervention #2
- ...and so on, up to ℓ of these tables.

$\mathbb{Z}$ is just "here's my binder of ℓ such tables, one per intervention I care about."

---
# Definition 2 : G-Constrained Neural Causal Model
-  **G-NCM** is just an SCM where the mechanism functions are replaced with trainable neural networks, but the neural networks are only allowed to look at the inputs that the causal diagram says they're allowed to look at.
- SCM has endogenous variables V, exogenous "contexts" U, and mechanisms F — one function $f_{V_i}$​​ per variable, saying how to compute $V_i$​ from its parents and its own noise.

A G-NCM takes this same shape, but with two changes:
1. The mechanisms $f_{V_i}$​​ become **neural networks** instead of arbitrary/unknown functions — meaning they're trainable, and you can fit them to data with gradient descent.
2. The **noise variables U** are handled differently — instead of one totally separate noise source per variable, noise is shared across variables that are already connected by dashed bidirected edges (unobserved confounding) in the graph G.
---
Notation in definition:

1. $\hat{M}^θ$ over V with parameters $θ$ = {$\theta_{V_i}$ : $V_i$ ∈ V}
	- $\hat{M}$ — the hat symbol (^) means "this is the _learned/neural_ version," as opposed to the true unknown model.
	- $θ$ — the collection of **all trainable weights**, one bundle $\theta_{V_i}$​​ per variable, since each variable gets its own little neural network.
2. $\hat{U}$ = {$\hat{U}_C$ ​: C ∈ C(G)}
	- $\mathcal{C}(G)$ = "the set of all maximal cliques over the bidirected edges of G." In plain English: **group together every set of variables that are all mutually connected by dashed double-arrows** (i.e., they all share some unobserved confounding with each other).
	- For each such group C, you create **one shared noise source** $\hat{U}$​.
	- **Why?** Because a dashed bidirected edge $V_j$ ⇠⇢ $V_i$​ means "there's some unmeasured background factor affecting _both_ $V_i$​ and $V_j$​. If you gave $V_i$​ and $V_j$​ each their own _independent_ private noise, you couldn't represent that shared influence at all. So instead, variables that are confounded together **draw from the same noise variable**, which is exactly what lets the model represent that correlation.

3. $\hat{F} = \{\hat{f}_{V_i} : V_i \in V\}$ — The Neural Mechanisms
	- Each variable $V_i$ gets ==its own neural network== $\hat{f}_{V_i}$, parameterized by its own weights $\theta_{V_i}$.
	- Inputs to this network: $U_{V_i} \cup Pa_{V_i}$
		- **$Pa_{V_i} = Pa_G(V_i)$** — the parents of $V_i$ in the graph
		- **$U_{V_i} = \{\hat{U}_C : \hat{U}_C \in \hat{U} \text{ s.t. } V_i \in C\}$** — whichever ==shared noise groups== $V_i$ belongs to.

Causal Diagram as Inductive Bias :
- The neural net computing $V_i$ is ==**architecturally forbidden**== from looking at anything that isn't:
	- a parent of $V_i$ in $G$, **or**
	- noise it shares confounding with.
- The graph **constrains what information is even allowed to flow into each network** —  ==you're structurally preventing the network from seeing irrelevant variables.==
---
- $\mathbb{C}$ — the Intervariable clusters: a grouping of the low-level variables into buckets (e.g. {C,F,P} grouped into one bucket called "calories").
- $\mathbb{D}$ — the Intravariable clusters: a grouping of the _values_ within each bucket (e.g. all combinations of carbs/fat/protein that add up to 200 calories get grouped together).
#### Definition 4 : Constructive Abstraction Function
High level example of what $\tau$ does:
- τ does exactly two jobs, applied one after another:
	1. **Collapse variables**: take a big list of low-level variable values, and squash each cluster of variables down into one high-level variable.
	2. **Collapse values**: take the value that cluster produced, and figure out _which bucket_ it falls into (from the intravariable clustering), then output the _label_ of that bucket as the high-level value.
- Using the running example: $\tau$ takes (carbs=10, fat=5, protein=8), computes calories = 4(10)+9(5)+4(8) = 117, checks which intravariable bucket 117 falls into (say, "the 100–150 calorie bucket"), and outputs the high-level value for that bucket.


Breaking down the three conditions of Definition 4
1. τ : $\mathcal{D}_{V_L} \to \mathcal{D}_{V_H}$
	- Read this as: "τ is a function whose input is a low-level value-tuple, and whose output is a high-level value-tuple." $\mathcal{D}_X$​ always means "the domain of X" — i.e., the set of all possible values X could take.

2. Condition 1: "There exists a bijective mapping between $V_H$​ and $\mathbb{C}$ such that each $V_{H,i} \in V_H$ corresponds to $C_i \in \mathbb{C}$"
	- "Bijective mapping" just means a perfect one-to-one pairing — every high-level variable corresponds to exactly one cluster, and every cluster corresponds to exactly one high-level variable. No leftovers on either side.
	- In plain English: **each intervariable cluster you formed _becomes_ one high-level variable.** If $\mathbb{C} = \{\{D\}, \{C,F,P\}, \{B\}\}$, then $V_H$​ literally has three variables: one named for the dish-cluster, one for the calorie-cluster (Z), one for the BMI-cluster.

3. Condition 2: "For each $V_{H,i} \in V_H$, there exists a bijective mapping between $\mathcal{D}_{V_{H,i}}$​​ and $\mathbb{D}_{C_i}$"
	- Same one-to-one idea, but now for _values_ instead of variables.
	- Each **intravariable cluster** ( $D^j_{C_i}$​, one specific bucket of low-level value-combinations) is paired one-to-one with a **specific value** of the high-level variable ($v^j_{H,i}$​).
	- Plain English: **each bucket you formed becomes one possible value of the corresponding high-level variable.** The bucket "all (c,f,p) combos totaling 200 calories" becomes literally the value "Z=200."

4. Condition 3: "τ is composed of subfunctions $\tau_{C_i}$ for each $C_i \in \mathbb{C}$ such that  $v_H = \tau(v_L) = (\tau_{C_i}(c_i) : C_i \in \mathbb{C})$ , where $\tau_{C_i}(c_i) = v^j_{H,i}$​ if and only if $c_i \in D^j_{C_i}$​"
	- $\tau_{C_i}$ — a **mini-function**, one per cluster, that only handles that one cluster's job. τ overall is just "run all these mini-functions side by side."
	- $c_i$​ — the actual observed low-level values belonging to cluster $C_i$​ (e.g., the specific tuple (c, f, p)=(10, 5, 8)).
	- $\tau_{C_i}(c_i) = v^j_{H,i}$​ **if and only if** $c_i \in D^j_{C_i}$ — this is the actual rule: "look up which bucket $D^j_{C_i}$​ the value $c_i$​ falls into, then output the high-level value $v^j_{H,i}$​ that corresponds to that bucket" (exactly what Condition 2 set up).
	- $v_H = \tau(v_L) = (\tau_{C_i}(c_i) : C_i \in \mathbb{C})$ — the _overall_ output $v_H$ is just the tuple of all these mini-function outputs, one per cluster, stacked together.
---
Putting the whole thing together: the calorie example
1. **Low-level input**: $(R, D, C, F, P, B)$ — restaurant, dish, carbs, fat, protein, BMI.
2. **Cluster them per $\mathbb{C}$**:
	- $D$ alone
	- $\{C, F, P\}$ together as one bucket
	- $B$ alone
	- ==$R$ gets dropped== — not in any cluster.
3. **For the $\{C, F, P\}$ cluster**:
	- Compute $c_i = (10, 5, 8)$
	- Look up which intravariable bucket it lands in ($D^j_{C_2}$ = "totals to 200")
	- Output $\tau_{C_2}(c_i) = Z{=}200$
4. **Do the same (trivially, since they're singleton clusters)** for $D$ and $B$.
5. **Stack the results**:
	$$\tau(v_L) = (D_H,\ Z{=}200,\ B_H)$$
	— this is $v_H$, ==the high-level output.==
