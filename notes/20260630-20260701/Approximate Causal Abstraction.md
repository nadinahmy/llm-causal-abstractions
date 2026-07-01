- From definition 2.2: Formally, the equation F_x maps ℛ(𝒰∪𝒱−X) to ℛ(X), so F_x determines the value of X, given the values of all the other variables in 𝒰∪𝒱. 
### Causal Dependency in Context (u)
- What does it mean for Y to "depend on" X? ---> Y **depends on** X in a context `u` (= the setting of all exogenous/external variables) if X has the power to change Y while everything else in the system is held fixed.
- ==Formal requirement== : dependency exists if there's at least one setting `z` of all the *other* internal variables such that flipping X from `x` to `x′` forces Y's structural equation to produce a different result.
$$F_Y(x, z, u) \neq F_Y(x', z, u)$$
- ==Key intuition== : even if Y is pushed around by tons of factors, X still counts as a cause as long as there's ==at least one scenario== where X makes a difference to Y's outcome.
- In recursive (acyclic) models ---> once the context `u` is fixed, these dependencies pin down the unique value of every variable in the system.
### Recursive (Acyclic) Models
- The paper restricts attention to ==recursive== (= acyclic) models ---> models where there's a partial order `⪯` on the variables such that if Y depends on X, then `X ≺ Y` [i.e. causes come before their effects in the ordering].
- Why does this matter? ---> given a context `u`, the values of all the other variables are ==fully determined==. We just solve for the endogenous variables one at a time, in the order given by `≺`.
- Notation : we write an endogenous variable's equation as `X = f(Y)` ---> means the value of X depends *only* on the variables in `Y`, and `f` is the rule connecting them.
- Climate example is recursive ---> `T = f_L(W, U)`.
### Interventions & Allowed Interventions
- What's an intervention? ---> written `X ← x` [the "do" operation], it ==forces== the variables in the set X to take the values x, regardless of what their equations would normally say.
- What does it do to the model? ---> setting X to x in a model `M = (𝒮, ℱ)` gives a ==new== model `M_{X←x}`, identical to M *except* its equations ℱ get swapped for ℱ_{X←x}:
	- every variable `Y ∉ X` ---> keeps its original equation, `F_Y` unchanged
	- every variable `X′ ∈ X` ---> its equation is thrown out and replaced by the constant `X′ = x′` [the matching value from x]
- ==Allowed interventions==:
	- old assumption (Halpern & Pearl 2005, Halpern 2016) ---> *any* intervention can be performed
	- this paper follows Rubenstein et al. 2017 + Beckers & Halpern 2019 and adds a set ℐ of ==allowed interventions== instead
	- why? ---> to capture cases where not all interventions interest the modeler, and/or some just aren't ==feasible== [needed later for defining abstraction]
- Updated definition ---> a causal model is now a tuple `M = (𝒮, ℱ, ℐ)`, where `(𝒮, ℱ)` is the basic model and ℐ is the allowed-intervention set.
	- sometimes written `M = (M′, ℐ)` with `M′ = (𝒮, ℱ)`, when we want to stress the role of ℐ.
#### Syntax note
- `⊨` is the "satisfies" or "is true in" symbol. `(M, u) ⊨ ψ` reads: _"the formula ψ is true in model M, given context u."_
	- tells you how to decide if a causal statement is true
---
A causal formula isn't true or false on its own — we need _both_ the model `M` (the equations) _and_ a context `u` (the external inputs) before we can ask whether it holds. That's why everything is phrased as `(M, u) ⊨ ψ` rather than `M ⊨ ψ`.
- `(M, u) ⊨ X = x` is true exactly when the variable `X` actually comes out equal to `x` once you solve the model's equations in context `u`. Because the model is recursive, there's a _unique_ solution (you just solve the variables one at a time in causal order), so there's no ambiguity — `X` has one definite value, and the formula is true iff that value is `x`
---
![[Pasted image 20260630105702.png|363]]
To check whether `φ` is true *under that intervention*, we don't evaluate `φ` in the original model `M` — we first build the surgically-altered model `M_{Y←y}` and *then* check whether `φ` holds in *that* model, using the same context `u`. So `[Y←y]φ` formalizes the statement *"φ would be the case if we intervened to set Y to y."*
- - `M(u)` = the single complete value-vector `v` that all the variables take in context `u`. Formally it's the unique `v` with `(M, u) ⊨ 𝒱 = v`. Think of it as **"run model M with external inputs u, and M(u) is the resulting state of the whole world."**
- `M(u, Y←y)` = the same thing, but in the _intervened_ world. It's the unique `v` such that `(M, u) ⊨ [Y←y](𝒱 = v)` — i.e. **"run the model with inputs u, but also forcing Y = y, and read off the full resulting state."**
See [[20260623-20260624#==Definition 3.12==]] for detailed definition of restriction set and omega-tau mapping
- A low-level intervention maps to a high-level one (ωτ​(X←x)=Y←y) if and only if: τ(Rst(VL​,x))=Rst(VH​,y).
- You take the "cloud" of all possible low-level states consistent with your micro-action and zoom them all out using τ. For the mapping to be valid, this zoomed-out cloud must **perfectly match** the set of high-level states consistent with a single macro-action (Y←y).
> [!IMPORTANT] Definition 2.3:
 > 1. The state mapping τ (how variables are grouped) must logically dictate the intervention mapping ω (how actions are grouped).
>  2. An action X←x is only valid at the high level if its "zoomed-out" results map **perfectly and uniquely** to a high-level action Y←y.
> 3. If a micro-intervention is too messy to "nail down" the macro-model to a single state, it is considered **undefined** in that abstraction. This ensures that the macro-model is a faithful, non-ambiguous representation of the micro-system.

-----
> [!ABSTRACT] Definition 3.1: dmax​ Distance
> - Measures the distance between two models (M1​,M2​) that share the same variables but have different equations [similar models].
> - It is a **worst-case metric**. It identifies the single largest discrepancy in predictions across all possible interventions and contexts.
> - **Alpha (**α**):** If the maximum discrepancy is ≤α, then M1​ is an α-approximation of M2​. It provides a mathematical guarantee on the maximum possible prediction error.

---
![[Pasted image 20260701213856.png|546]]
- **Pathway 1 (Micro-First):** Perform a micro-intervention (X←x) in a micro-context (uL​), calculate the micro-result (ML​), and then "zoom out" using the mapping function τ.
- **Pathway 2 (Macro-First):** "zoom out" the intervention and context first (using ωτ​ and τU​) and then run the high-level macro-model (MH​) to get a prediction.

Definition 3.2 calculates the ==**maximum distance** (d_V_H​​)== between these two pathways. If the M_H's prediction is always within a distance of α from the "zoomed-out" micro-reality, the abstraction is faithful within that error bound.

- max : We look for the **worst-case scenario**. We test every possible low-level intervention (X←x) and every possible context (uL​) to find the ==single largest discrepancy== between the micro-reality and the macro-prediction.
- min : We choose the **best possible mapping** for the exogenous noise (τU​). Since the function that "lifts" low-level noise to high-level noise is a theoretical link rather than a physical part of the system, we assume the abstraction is as good as the most optimal τU​ we can find.

> [!IMPORTANT] Definition 3.2: Approximate Abstraction (dτ​) 
> - Measures the "faithfulness" of a macro-model by comparing micro-results (zoomed out) against macro-predictions.
> - By taking the max over all interventions and contexts, it guarantees that the macro-model's error will never exceed α.
> - In the climate example, MH​ is a τ-α abstraction if the expected difference between the predicted high-level temperature (e′) and the actual zoomed-out micro-temperature (e) is less than or equal to α.
> - If α=0, then MH​ is a perfect, exact τ-abstraction of ML​.

As mentioned in the paper : **We are identifying the degree to which MH approximates ML with the worst-case distance between the two ways of lifting the context-intervention pairs, for an optimal choice of τ𝒰.**

---
>[!TIP] The "Sanity Check" Propositions
> - **Prop 3.3:** Shows that **Approximate Abstraction (with** α=0**) = Exact Abstraction**. This ensures the new math doesn't break the old, perfect-case rules.
> - **Prop 3.4:** Shows that **Approximate Abstraction (with** Id **mapping) = Model Approximation (**dmax​**)**. This ensures that comparing two models at the same level is just a specific, "non-zoomed" version of the broader abstraction framework.

---
> [!DEFINITION] **Definition 4.2: Likelihood of Serious Error (**dβ​**)**
> -  used when you don't care about the average error, but you are very worried about **major failures** or "outliers".
> ![[Pasted image 20260701221204.png]]
>- β : **Tolerance Threshold** ---> It represents what you consider a "serious" or "significant" difference (e.g., an error of more than 5 degrees in a weather model).
>- Pr({u:⋯≥β}): **Probability of Failure** ---> It calculates the total likelihood of all environments where the distance between the two models' predictions is greater than or equal to your threshold β.
>- max…​: Again, we evaluate this for the intervention that causes the most frequent "major" errors.
>- "For the worst-case action, what is the **probability** that my model will give me a result that is **significantly wrong** (off by at least β)?"

---
> [!IMPORTANT] Definition 4.8: Probabilistic Distance with Intervention Distributions
> - **The Goal:** Measuring abstraction error when macro-actions have multiple possible micro-level implementations.
> -  **Key Strength:** It prevents "rare boundary cases" at the micro-level from unfairly disqualifying a high-level model that is accurate the vast majority of the time.
> - **The Two Uncertainties:**
>	1. **Context Uncertainty (**PrL​**):** We don't know exactly what the environment is.
>	2. **Implementation Uncertainty (**PrI​**):** We don't know exactly which micro-action triggered the macro-action.

---
Some notes on the notion of "Distance" and what it means to compare "macro level distance" to "micro level distance" :
- **"distances"** here refer to the mathematical way we measure ==how different two states are from each other within a specific model==. 
- Distances at the micro and macro levels are often **unrelated** ---> the "zoom out" rule (τ) can drastically change the scale of an error when moving between them.
**1. Micro-Level Distance (dVL​​)**
- measures the difference between two highly detailed states in the low-level model (ML​).
- metric defined on the low-level state space, comparing the values of every single micro-variable.
**2. Macro-Level Distance (dVH​​)**
- measures the difference between two simplified states in the high-level model (MH​).
- metric defined on the macro-level state space, which has much fewer variables and a different range of values.
Problem seen in **Proposition 5.2** is that the magnitude of an error at the low level does not tell the magnitude of the error at the high level, because abstraction mapping (τ) can be either **highly robust** or **highly sensitive**:
- **Robustness :** You could have a **massive micro-distance** (e.g., changing the wind speed in thousands of grid squares) that results in a **zero macro-distance** because those changes didn't affect the final average temperature enough to change the state from "Normal" to "El Niño".
- **Sensitivity:** You could have a **tiny micro-distance** (e.g., a tiny change in one neuron or one voter) that happens to be the "tipping point" that flips a macro-variable from 0 to 1. In this case, a small micro-error is magnified into a large macro-error.
> [!SOLUTION] **The Factor k** 
> - Because these two distances are measured on different scales and in different variable spaces ----> the constant k in **Proposition 5.2** was introduced.
> - k represents the maximum possible "magnification" that the mapping τ can perform.
> - If k is high, the macro-model is very sensitive to micro-errors; if k is low, the macro-model is very robust and ignores most micro-noise.

