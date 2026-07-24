---

---

---
- Main problem the paper discusses : The graphical and parametrical conditions under which a causal model can abstract another are not known.
	- another open problem is learning causal abstractions from data
- Target of the paper : Tackle these issues for linear causal models with linear abstraction functions
- ==Abs-LiNGAM== : a method that leverages the constraints induced by the learned high-level model and the abstraction function to speedup the recovery of the larger low-level model, under the assumption of non-Gaussian noise terms.
- Causal Abstraction has also found wide interest in ==explainable AI== to align machine rep- resentations with human-interpretable concepts in feed-forward neural networks, concept- based neural networks, and Large Language Models.
#### Formula (1) :
${x ∈ D(X) ∣ x_V ​= v}$
This mathematical notation describes the "filter" applied to the system:
- $x∈D(X)$ : We are looking at a configuration (a vector of values) for every single variable in the model.
- **The vertical bar ∣ :** This means "such that" or "given the condition that."
- $x_V​ = v$ : This is the condition. It states that for the specific subset of variables you manipulated (V), their values in the vector (x) must exactly match the intervention value (v).
#### Figure 1
**Part (a): T-Abstraction (The Theory)**
Visualizes the mathematical relationship between a **concrete model (L)** and an **abstract model (H)**.
- **The Mapping (T):** The large matrix T is the "bridge" that transforms a high-dimensional concrete state into a low-dimensional abstract state (e.g., mapping thousands of neurons to a single concept).
- **Relevant Variables (Dashed Circles):** These are the specific low-level variables ($X_i$​) that the abstraction function directly depends on to calculate the value of an abstract variable ($Y_j$​).
- **Concrete Blocks ($Π(Y)$):** key contribution of this paper, A "block" is a group of low-level variables that ==includes the relevant variables **plus** any irrelevant variables that sit on a "T-direct path" to them.==
- **The Ordering Rule:** Figure 1(a) shows that if $Y1$​ causes $Y2​$ in the abstract world, then the entire block $Π(Y1​)$ must causally precede the block $Π(Y2​)$ in the concrete world.
==**Causal Discovery** :== Finding the causal "gears" in a system with hundreds of variables is computationally exhausting; Abs-LiNGAM uses high-level knowledge to "cheat" and find the answer faster.
**Part (b): Abs-LiNGAM**
The **Abs-LiNGAM** part of the figure illustrates a new 4-step algorithm designed to solve a major problem in AI: ==**Causal Discovery**==. Finding the causal "gears" in a system with hundreds of variables is computationally exhausting; Abs-LiNGAM ==uses high-level knowledge to "cheat" and find the answer faster.==
The steps of the abs-LiNGAM method:
1. **Step (i) - T-Reconstruction:** The algorithm takes a small set of paired samples ($D_J$​) to learn the mapping $T$. It figures out which concrete variables are "relevant" to which abstract concepts.
2. **Step (ii) - Abstract Discovery:** It applies the learned mapping T to a much larger dataset of concrete-only observations ($D_L$​). This creates a "synthetic" abstract dataset ($D_\hat{H}$​), which it uses to quickly find the **abstract causal graph** $\hat{H}$ using a standard algorithm like DirectLiNGAM.
3. **Step (iii) - Infer Constraints (K):** This is the clever part. The algorithm looks at the abstract graph it just found. If there is no path between abstract variable $Y_i$​ **and** $Y_j$​, it knows there cannot be a causal path between their low-level counterparts. It collects these "forbidden paths" into a set of constraints called K.
4. **Step (iv) - Concrete Discovery:** Finally, it runs causal discovery on the original, massive low-level dataset ($D_L$​). However, it tells the algorithm: _"Don't even bother checking these forbidden paths in_ K." This drastically reduces the search space, allowing the computer to find the complex low-level structure ($L$) in a fraction of the time.
---
#### Small example of Abs-LiNGAM in action
- We are trying to map the wiring of a massive, complex computer (**Low-Level Model** $L$). This computer has **1,000 individual wires** $(X_1,\ldots,X_{1000})$, and determining which wire causes another is an overwhelming task.
- We also possess a **high-level manual** describing the computer's major functional blocks (**Abstract Model** $H$). Instead of thousands of wires, the manual contains only five components:
	- Power ($Y_1$)
	- Processor ($Y_2$)
	- Memory ($Y_3$)
	- Graphics ($Y_4$)
	- Storage ($Y_5$)
- ==Abs-LiNGAM uses this simple high-level description to **guide** the search for the much more complicated low-level causal graph.==
#### Step (i): T-Reconstruction (Finding the "Plugs")
- we begin with a **small paired dataset** $D_J$ where we can observe **both** the individual wires $(X)$ and the high-level blocks $(Y)$.
- Goal : Determine which wires belong to each high-level block
Example: 
- After analyzing the data, we discover:
	- Wires $X_1,\ldots,X_{10}$ all belong to the **Power** block ($Y_1$).
	- Wires $X_{11},\ldots,X_{50}$ belong to the **Processor** block ($Y_2$).
	- ...
- These groups are the **Relevant Variables**: $\Pi_R(Y_1)=\{X_1,\ldots,X_{10}\}$ ---> these are the wires responsible for implementing the abstract concept **Power**.
#### Step (ii): Abstract Discovery (Learning the "Manual")
- Next, we obtain a **very large dataset** $D_L$ containing observations of only the 1,000 wires.
- Since we already know the mapping $T$, we can compress this dataset into a much smaller abstract dataset: $\hat{D}_H$
- Instead of analyzing 1,000 variables, we now analyze only the five abstract variables: $Y_1,\ldots,Y_5$
- Goal : Learn the causal graph among the five high-level components.
- Example
	- Running a standard causal discovery algorithm reveals:
	- Power ($Y_1$) → Processor ($Y_2$)
	- Processor ($Y_2$) → Memory ($Y_3$)
	but
	- Power ($Y_1$) has **no causal influence** on Graphics ($Y_4$).
#### Step (iii): Infer Constraints ($K$) (Creating the "Forbidden List")
- This is the key insight of Abs-LiNGAM.
- If the abstract model tells us that $Y_1 \nrightarrow Y_4$ --->  none of the wires implementing Power can causally influence any wires implementing Graphics.
- Example:
	- Since $\Pi_R(Y_1)=\{X_1,\ldots,X_{10}\}$ and $\Pi_R(Y_4)=\{X_{500},\ldots,X_{510}\}$
	- Conclusion: No wire from $\{X_1,\ldots,X_{10}\}$ can have a directed causal path to any wire in $\{X_{500},\ldots,X_{510}\}$.
	- These forbidden relationships become a set of constraints: $K.$
	- Rather than searching every possible connection, the algorithm is now told which ones are impossible.
#### Step (iv): Concrete Discovery (Recovering the Full Wiring Diagram)
- Finally, we return to the original dataset containing all 1,000 wires.
- A causal discovery algorithm (such as **DirectLiNGAM**) is run **with the constraint set $K$**.
#### Result
- Instead of testing every possible wire-to-wire connection, the algorithm ignores thousands of impossible edges ---> dramatically reduces the search space and makes learning the complete low-level causal graph much faster.
> [!Visualization] Visualization
> Without abstract knowledge, solving the causal graph is like solving a **1,000-piece puzzle** while checking every piece against every other piece.
> With Abs-LiNGAM, we already know:
> - Pieces from **Pile A** can never connect to **Pile D**.
> - Pieces from **Pile B** can never connect to **Pile E**.
> - The algorithm therefore skips huge portions of the search and focuses only on plausible connections.

---
#### Quick recall : Causal Sufficiency and Faithfulness
1) ==Causal Sufficiency==
	- Causal sufficiency is the assumption that the model includes all relevant variables necessary to explain the observed causal relationships.
	- Implies the **absence of hidden confounding** and selection bias.
	- Requires the **mutual independence of exogenous terms** (the noise/background factors). For any two exogenous variables $E_1$​ and $E_2$​, it must hold that $E_1 ​⊥ E_2$​.
	- If a high-level model lacks causal sufficiency, its variables become **confounded**. This often happens if the "blocks" of low-level variables used to create abstract variables overlap, causing the abstract noise terms (U) to become dependent.
2) Faithfulness
	- Faithfulness is the assumption that every statistical dependency observed in the data is actually a result of the model's causal structure, rather than a coincidence.
	- Implies the **absence of "canceling paths"** across variables.
	- **The "Zero-Effect" Trap:** In an unfaithful model, a variable could have a physical path to another variable, but the weights of the intermediate edges might perfectly cancel each other out (e.g., a +1 path and a −1 path), making the net causal effect look like zero.
	- Example 1 in the Paper: authors demonstrate an **unfaithful concrete model** where the effect of X1​ on X4​ is canceled out. Even though a path exists at the low level, the abstract model shows no connection between the corresponding variables ($Y_ 1 ​↛  Y_2$​) because the causal influence was "hidden" by the cancellation.
> [!Summary] Summary:
Sufficiency ensures we aren't seeing "ghost" relationships caused by things we didn't measure, Faithfulness ensures we aren't "missing" relationships that actually exist because the math accidentally zeroed them out

---
**Surjectivity and Full Column-Rank**
The transformation T is the matrix that maps concrete states (x) to abstract states (y). For this to be "surjective," it means that ==**every possible state in the abstract world must be reachable** from at least one state in the concrete world.==
- **Full Column-Rank:** Mathematically, for the matrix T (which has d concrete variables and b abstract variables) to be surjective, it must have **full column-rank**. This ensures that each abstract variable is "doing something unique"—there are no redundant abstract variables that are just linear combinations of the others.
	- full column-rank ---> columns of a matrix are linearly independent, meaning no abstract variable can be represented as a linear combination of the others.
- **Non-Empty Relevant Sets:** If a column in T were empty (all zeros), the corresponding abstract variable would always be zero regardless of the low-level state. This would violate surjectivity because you could never "reach" any other value for that variable.
**The Requirement for Interventional Consistency**
The core of Causal Abstraction is **Interventional Consistency** (Equation 4). This rule states that if you "do" something in the abstract world (like setting a concept to "Positive"), the result must be identical to what happens if you perform the corresponding "implementation" in the messy concrete world.
#### Lemma 1
**Why Relevant Variables Must Be Disjoint**
- relevant variables must be **mutually disjoint** ---> a single concrete variable (like a specific neuron) cannot be "relevant" to two different abstract variables.
Proven by contradiction in **Appendix B.1**:
- **The Conflict:** Imagine one concrete variable, $X_s$​, is part of the "Relevant Set" for both Y1​ and Y2​.
- **The Action:** You decide to perform an intervention on Y1​ in the abstract model. Because $X_s$​ is relevant to Y1​, your low-level implementation **must** fix the value of $X_s$​ to satisfy the intervention.
- **The Failure:** Because $X_s$​ is _also_ relevant to Y2​, the act of fixing $X_s$ will instantly change the value of Y2​ as well.
- **The Contradiction:** ==If the abstract model says that Y1​ does not cause Y2​, then Y2​ **should not change** when we intervene on Y1​. However, the shared variable forced it to change anyway.==
To prevent these "accidental" causal effects and ensure the abstract model's logic holds true for **all possible interventions**, each abstract variable must have its own private, unique set of concrete variables.
---
Proof of corollary 1:
- **Corollary 1 (Constructive Abstraction)** establishes that any linear T-abstraction between two linear Structural Causal Models (SCMs) is automatically a **constructive abstraction**.
- In simple terms, ==a constructive abstraction is one where each high-level variable depends on its own unique, non-overlapping set of low-level variables.== 

**Dependence via Linear Mapping**
- In a linear T-abstraction, the relationship between concrete variables (X) and abstract variables (Y) is defined by a matrix T. By definition, the set of low-level variables that an abstract variable $Y_j$​ "depends on" is exactly its set of **relevant variables** $Π_R​(Y_j​)$—those variables with non-zero coefficients in the j-th column of the matrix T.

**Reliance on Lemma 1 (Disjointness)**
- **Lemma 1** states that for any two distinct abstract variables, their sets of relevant concrete variables must be **mutually disjoint**.
- If the relevant sets are disjoint, it means each high-level variable is "constructed" from a private pool of low-level variables that no other high-level variable uses. ==This matches the definition of a constructive abstraction.==

---
**Sufficient Abstract Connectivity (Lemma 2)**
This lemma links a specific type of low-level path to a **direct edge** in the abstract graph.
- **The Condition:** There must be a **T-direct path** $(X_1 \xrightarrow{T} X_2)$ between two relevant variables. A T-direct path is a low-level directed path where every intermediate variable is **irrelevant** (it doesn't directly contribute to any abstract concept).
- **The Result:** This guarantees an **edge** (Y1​→Y2​) exists in the high-level graph.
- **The Logic:** Because the influence isn't "interrupted" by other relevant variables at the low level, the high-level model must represent this as a single, direct causal step.
**Sufficient Directed Paths (Corollary 2)**
This is a broader generalization that links any low-level path to an **abstract path**.
- **The Condition:** There is **any directed path** ($X_1​ \rightsquigarrow X_2$​) between relevant variables in the low-level model. This path could be T-direct, or it could pass through other relevant variables (like $X_3$​) along the way.
- **The Result:** This guarantees a **directed path** ($Y_1 ​\rightsquigarrow Y_2$​) exists in the high-level graph.
- **The Logic:** If the low-level path is not T-direct, it means it is "interrupted" by a variable $X_3$​ that belongs to another abstract concept $Y_3$​. By applying Lemma 2 to the segments of that path, you get a chain of abstract edges ($Y_1 ​→ Y_3 ​→ Y_2$​), which forms a high-level path.
---
#### Theorem 1
- For a direct edge to exist between two abstract variables ($Y_1 ​→ Y_2$​), **every single concrete variable** in the relevant set of the source ($Π_R​(Y_1​)$) must have a path to **at least one** relevant variable in the target set ($Π_R​(Y_2​)$).
- If $Y_1$​ causes $Y_2$​ at the high level, it means that ==**any change** to $Y_1$​ must result in a change to $Y_2$​.==
- Since $Y_1$​ is defined by its relevant concrete variables, ==a change to **any one** of those variables ($X ∈ Π_R​(Y_1​)$) effectively changes the value of $Y_1$​.==
- Therefore, ==if even one variable in the $Y_1$​ cluster has no way to influence the $Y_2$​ cluster, you could manipulate that variable, change the value of $Y_1$​, and see **zero change** in $Y_2$​.== This would break "interventional consistency," meaning your abstract model would be a lie.
#### Example: Invalid Abstraction (Violation of Theorem 1)
- Consider two abstract variables:
	- **Diet** ($Y_1$)
	- **Weight** ($Y_2$)

- Relevant variables:
	- $Π_R(Y_1)=\{X_{\text{Sugar}},X_{\text{WaterIntake}}\}$
	- $Π_R(Y_2)=\{X_{\text{BMI}}\}$

- Low-level causal structure:
	- $X_{\text{Sugar}} \rightarrow X_{\text{BMI}}$
	- $X_{\text{WaterIntake}} \not\rightsquigarrow X_{\text{BMI}}$

- Suppose the abstract graph contains:
	- $Y_1 \rightarrow Y_2$

- Now intervene only on:
	- $X_{\text{WaterIntake}}$

- Result:
	- $Y_1$ changes because $X_{\text{WaterIntake}} \in Π_R(Y_1)$.
	- $Y_2$ does **not** change because there is **no directed path** from $X_{\text{WaterIntake}}$ to $X_{\text{BMI}}$.

- ==This violates Theorem 1.==

- Why?
	- Theorem 1 requires **every** relevant variable in $Π_R(Y_1)$ to have a directed path to **at least one** relevant variable in $Π_R(Y_2)$.
	- Here, $X_{\text{WaterIntake}}$ has no path to $X_{\text{BMI}}$.

- ==The abstract edge $Y_1 \rightarrow Y_2$ is therefore inconsistent with the true low-level causal system.==
---
