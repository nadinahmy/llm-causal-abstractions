> [!Theorem] Theorem 21
> *Suppose $Ψ_0$​ is a set of atomic soft interventions with signature Σ. Let Ψ be the closure of $Ψ_0$​ under function composition. Then $(Ψ,∘)$ is an intervention algebra.*

Unpacking the theorem term by term:
- **Soft intervention** : a family ${I_X}_{X \in \mathbf{X}}$ where each $I_X : \mathrm{Val}_{\mathrm{V}} \to \mathrm{Val}_X$ _replaces_ the mechanism $F_X$ directly. Replacement is a fixed, pre-chosen function — it does **not** look back at the old mechanism.
- **Atomic** : targets exactly one variable. So each element of $\Psi_0$ is a single mechanism-replacement "$F_X := I_X$".
- **Closure under composition**: $\Psi$ = all finite composites $I^{(1)} \circ \dots \circ I^{(n)}$ of elements of $\Psi_0$ (including, implicitly, the empty composite). Why is this needed at all? Because Definition 15 [[JMLR - Causal Abstraction, A Theoretical Foundation for Mechanistic Interpretability#Definition 15 Intervention Algebra]] demands an _algebra_ — the operation $\circ$ must not lead outside the set. $\Psi_0$ alone isn't closed; $\Psi$ is closed by construction.
- **Two signatures are now in play**:
    - $\Sigma$ — the original signature whose models the soft interventions operate on;
    - $\Sigma^\ast$ — the manufactured signature that the proof will conjure, on which everything becomes hard interventions.
    
**The two laws — and why soft interventions obey them:**
- The only real work in Theorem 21 Theorem 20 already proved: anything obeying the two laws _is_ hard interventions on some signature. So Theorem 21 just has to check that soft interventions obey them.

1. Law 1 — commutativity (different variables → order doesn't matter). Say $I$ swaps the mechanism in slot $X$ and $I'$ swaps the mechanism in slot $Y$. These touch different slots. Doing $I$ then $I'$, versus $I'$ then $I$ — either way you end with the new part in slot $X$ and the new part in slot $Y$. Order is irrelevant because they never interfere. ✔

2. Law 2 — left-annihilativity (same variable → the second overwrites the first). Now $I$ and $I'$ both target slot $Y$. $I$ installs part $P$; then $I'$ installs part $P'$ into the same slot. The second install obliterates the first — slot $Y$ ends holding $P'$, exactly as if $I$ had never run. So $I \circ I' = I'$. ✔

- Where "chosen in advance" earns its keep: $I'$ installs $P'$ **no matter what's currently in the slot** — it doesn't check. So it can't matter that $I$ ran first. The first intervention leaves no fingerprint. That blindness is the entire reason left-annihilativity holds.

- Conclusion ---> Both laws check out $\Rightarrow$ soft interventions are eligible $\Rightarrow$ Theorem 20's construction applies to them $\Rightarrow$ they form an intervention algebra.

---
Decoding the manufactured signature $\Sigma^\ast$ (end of the Theorem 21 proof)
> [!quote] Consequently, there exists a signature $\Sigma^\ast$ such that hard interventions on $\Sigma^\ast$ are isomorphic to the soft interventions $\Psi$ with respect to function composition. Specifically, where $\Psi_0^X \subseteq \Psi_0$ is the set of atomic soft interventions that target $X$, the variables of $\Sigma^\ast$ are $\mathrm{V}^\ast = {X^\ast : \Psi_0^X \neq \emptyset}$ and, for each $X^\ast \in \mathrm{V}^\ast$, the values are $\mathrm{Val}_{X^\ast} = \Psi_0^X$.

Running example throughout: $\Psi_0 = {J, I_1, I_2}$ with $J$ targeting $X$, and $I_1, I_2$ targeting $Y$.
The pieces of notation
- **$\Psi_0^X$** — A _subset_ of $\Psi_0$: keep only the atoms aimed at variable $X$. The superscript is a **filter, not an exponent**.
    - $\Psi_0^X = {J}$, and $\Psi_0^Y = {I_1, I_2}$.
- **$\Psi_0^X \neq \emptyset$** — "there is at least one atom targeting $X$." A non-emptiness test; it skips any original variable no intervention ever touches.
- **$\mathrm{V}^\ast = {X^\ast : \Psi_0^X \neq \emptyset}$** — for each original variable $X$ that gets targeted, mint a new symbol $X^\ast$. In the example both $X$ and $Y$ are targeted, so $\mathrm{V}^\ast = {X^\ast, Y^\ast}$.
- **$\mathrm{Val}_{X^\ast} = \Psi_0^X$** — the punchline. The **values** of $X^\ast$ _are the soft interventions themselves_: $\mathrm{Val}_{Y^\ast} = {I_1, I_2}$, a value-set whose elements are mechanism-replacements.

* A "value" being an entire intervention seems wrong — values are usually numbers. But Def. 2 only requires each variable to come with _some_ non-empty range; it never says values must be numbers. So letting them be interventions is allowed.

What the construction is actually saying:
- $X^\ast$ = "**which mechanism is currently installed in slot $X$?**"
- and its values are exactly the mechanisms you could install there, i.e. $\Psi_0^X$. Then the isomorphism is almost obvious:

| Soft world ($\Sigma$)                                                  | Hard world ($\Sigma^\ast$)                                                 |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| composite $J \circ I_2$: "install constant-3 at $X$, increment at $Y$" | partial setting ${X^\ast = J,\ Y^\ast = I_2}$: "record those as installed" |
| _edits_ mechanisms                                                     | _pins variables to values_                                                 |

Why the star, and why it's needed
- The star just keeps the two levels typographically apart: $X$ = original variable (values: numbers); $X^\ast$ = meta-variable about $X$'s slot (values: mechanisms).
- The proof **has** to name this signature because Definition 15 says "intervention algebra" = "isomorphic to hard interventions on **some** signature".

Sanity check (counting)
- $\mathrm{V}^\ast = {X^\ast, Y^\ast}$; $\mathrm{Val}_{X^\ast} = {J}$; $\mathrm{Val}_{Y^\ast} = {I_1, I_2}$.
- Hard interventions on $\Sigma^\ast$ = each variable set-or-left-alone = $2 \times 3 = 6$.
- Soft composites in $\Psi$: $X$'s slot (original or $J$) $\times$ $Y$'s slot (original, $I_1$, or $I_2$) = $6$.
- Equal finite counts + operation-preserving map = the asserted isomorphism.
- ----
#### The proof of Theorem 21
The proof that this algebra is isomorphic to a set of hard interventions follows exactly as in the proof of Theorem 20, relying on the fact that soft interventions also satisfy the key properties (a) and (b) in Definition 16:
- If soft interventions target different variables, $∘$ is commutative: if X $\neq$ Y, then $I_X$ $∘$ $I_Y$ = $I_Y$ $∘$ $I_X$.
	- $I_X$ replaces slot $X$'s mechanism, $I_Y$ replaces slot $Y$'s. Disjoint slots $\Rightarrow$ order irrelevant $\Rightarrow$ the two composites are the same function on models.
- If interventions target the same variable, $∘$ is left-annihilative: if X=Y, then $I_X$ $∘$ $I_Y$ = $I_Y$.
	- Both target slot $X$. Do $I_X$ (installs a mechanism), then $I_Y$ (installs another into the *same* slot); the second overwrites the first, net effect $= I_Y$.
- With (i) and (ii) established, Equation (2) from Theorem 20 replays unchanged: composing a sequence of soft interventions gives the same function whether you use the raw sequence or its cleaned-up normal form. So the evaluation map "sequence" ↦ "its composite" is constant on each ≈-bucket, descends to a well-defined isomorphism, and — exactly as in Theorem 20 — the quotient of sequences is isomorphic to hard interventions on a manufactured signature. 
Consequently, there exists a signature $\Sigma^\ast$ such that hard interventions on $\Sigma^\ast$ are isomorphic to the soft interventions $\Psi$ with respect to function composition.
- where $\Psi_0^X \subseteq \Psi_0$ is the set of atomic soft interventions that target $X$, the variables of $\Sigma^\ast$ are $\mathrm{V}^\ast = \{X^\ast : \Psi_0^X \neq \emptyset\}$ and, for each $X^\ast \in \mathrm{V}^\ast$, the values are $\mathrm{Val}_{X^\ast} = \Psi_0^X$.
- $\Psi_0^X \subseteq \Psi_0$** — atoms targeting $X$ (superscript = filter, not power). Example: $\Psi_0^X = \{J\}$, $\Psi_0^Y = \{I_1, I_2\}$.
- $\mathrm{V}^\ast = \{X^\ast : \Psi_0^X \neq \emptyset\}$** — one new variable $X^\ast$ per targeted original $X$. Read $X^\ast$ = "*which mechanism is installed in slot $X$?*" Example: $\mathrm{V}^\ast = \{X^\ast, Y^\ast\}$.
- $\mathrm{Val}_{X^\ast} = \Psi_0^X$ — its values *are the swaps themselves*. $\mathrm{Val}_{Y^\ast} = \{I_1, I_2\}$. Setting $X^\ast = I$ means "$I$ is the mechanism in slot $X$." ==Legal because Def. 2 allows any non-empty value-range, including a set of interventions.==
The isomorphism reads: a soft composite on $\Sigma$ (edit these mechanisms) $\leftrightarrow$ a hard intervention on $\Sigma^\ast$ (record which mechanisms are installed). Swapping-mechanisms downstairs $=$ setting-a-value upstairs, where the value *is* the mechanism.
---
#### Short description and definition of Isomorphism with a small example

- Two structures are **isomorphic** when they are _the same thing wearing different labels_ — there's a perfect dictionary between them that translates not just the objects but the operations, with nothing lost and nothing left over.
- The cleanest everyday example: your left hand and a mirror image of your left hand. Same structure exactly — same number of fingers, same joints, same angles. The mirror is a perfect relabeling. But they're not _identical_ (you can't superimpose them). They're isomorphic: structurally indistinguishable, physically distinct.

A more mathematical one. Consider two systems:

- **System A:** the numbers {0,1,2,3} under "add, then wrap around at 4" (so 3+2=1, since 5 wraps to 1).
- **System B:** the four rotations of a square {0°,90°,180°,270°} under "do one rotation, then the other."
- These _look_ like completely different subjects — arithmetic versus geometry. But match them up:   0 $\leftrightarrow$ 0°, 1↔90°, 2↔180°, 3↔270°. Now check the operations correspond. In A: 1+1=2. In B: 90° then 90°=180°. Match. In A: 3+2=1. In B: 270° then 180° = 450° = 90°. Match. _Every_ equation in A becomes a true equation in B under the dictionary, and vice versa. ==They're isomorphic — one is arithmetic, one is geometry, but structurally they're the same four-element system.==
### The two ingredients
To claim "P is isomorphic to Q" you need a map ϕ : P → Q (the dictionary) with two features:
1. **Bijection** — perfect pairing. Every element of P maps to exactly one element of Q, every element of Q is hit exactly once. No two things in P collapse to the same thing in Q (nothing lost), and nothing in Q is missed (nothing extra). This is what makes it a faithful relabeling rather than a lossy summary.
2. **Operation-preserving (homomorphism)** — the dictionary respects the structure. If you combine two things in P and _then_ translate, you get the same answer as translating both _first_ and combining in Q. In symbols: ϕ(a ∗ b) = ϕ(a) ⋆ ϕ(b), where $∗$ is P's operation and $⋆$ is Q's.

A map with both properties is an **isomorphism**, written P≃Q. The payoff: anything true about P's structure is automatically true about Q's. They're interchangeable for all structural purposes.

> [!Question] Why do we even care about isomorphism throughout this paper?
> - Definition 15 _defines_ "intervention algebra" as "isomorphic to hard interventions on some signature." So proving something is an intervention algebra _just is_ exhibiting an isomorphism to hard interventions. Theorem 20 did it for sequences-of-atoms; Theorem 21 does it for soft interventions; later, the same word will let the authors of the paper say a rotated linear subspace of a neural net "is" a variable.
> - Each time, the claim is: _this exotic-looking thing is, structurally, nothing more than ordinary hard interventions on a well-chosen variable space_ — same structure, different labels. Isomorphism is the precise sense of "same structure."

---
Definition 24:
The Formal Rule:
- The definition states that for an intervention algebra (Λ,⊕), and two interventions λ and λ′ within that set:
- λ≤λ′ **if and only if** λ′⊕λ=λ′.
- 
To understand why this formula defines an "ordering," it helps to think about what the operation ⊕ (combining interventions) actually does:
- The formula says: "Intervention λ is smaller than or equal to λ′ if performing λ _after_ λ′ changes absolutely nothing".
- If λ′ is a large intervention that already fixed the values of certain neurons, and λ is a smaller intervention that tries to fix a subset of those same neurons to the same values, then λ is redundant. The system is already in the state λ wants it to be in.

Application to Hard Interventions:
- The definition notes that for standard hard interventions (Φ,∘), this ordering matches the **natural partial order** [covered this in previous papers]
- x ≤ y if the set of variables changed by x is a **subset** of those changed by y, and they are set to the same values.
- **Example:**
    - **Intervention** y **(Larger):** Set (Voter 1, Voter 2) to (Yes, No).
    - **Intervention** x **(Smaller):** Set (Voter 1) to (Yes).
    - Because x is "contained" within y, x ≤ y. If you perform y and then perform x, the final state is still just y (Voter 1 is still Yes, Voter 2 is still No).

Why the Order Matters
- The definition points out that ≤ is defined like a **semi-lattice**, but with one crucial warning: ==the operation ⊕ is **not commutative** in general.==
- This means the sequence matters: λ′⊕λ is not necessarily the same as λ⊕λ′.
- In the rule λ′⊕λ=λ′, the "smaller" intervention **must come last** to prove it is redundant and thus "lower" in the ordering.