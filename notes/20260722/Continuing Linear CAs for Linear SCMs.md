---

---

---
**Linear Transformations**
- mathematical mapping between two sets of variables (vectors) that ==preserves the operations of addition and multiplication by a constant.== 
- In this paper, any linear transformation is represented by a **matrix**.
- If we have a set of low-level variables x, a linear transformation T produces abstract variables y by performing the operation $y = T^⊤x$.
- **The Intuition:** Every output variable is simply a "weighted recipe" of the input variables. For example, if $Y_1 ​= 2X_1 ​+ 0.5X_2$​, this is a linear transformation because $Y_1$​ is just a sum of $X_1$​ and $X_2$​ multiplied by fixed weights.
#### What is a Linear Causal Model (SCM)?
- A causal model is **linear** when the ==structural equations governing how variables interact are **linear combinations** of their parent variables==.
- **The Formula:** $X = W^\top X + E$
- **The Meaning:**
	- Every variable $X$ is determined by adding together the values of its **parent variables** (scaled by the weights in the matrix $W$) and adding a unique piece of **background noise** ($E$).
- **Total Effect:**
	- Because the model is linear, we can calculate the **total effect** of the exogenous noise on the entire system using the matrix $F=(I-W)^{-1}.$
	- This allows the model to be solved in **closed form**, meaning we can compute the final values directly without requiring iterative simulations or repeated causal propagation.
---
**Corollary 4 (Exogenous Abstraction)** 
- provides the bridge that links the background noise of the complex model to the background noise of the simple model.
- proves that if the high-level and low-level models are both linear and share a linear abstraction function (T), then the way their **exogenous (unobserved) variables** relate is also unique and linear.

**1. The Formula:** $γ(e) = S^⊤e$
- The corollary defines the **exogenous abstraction function ($γ$)** as a linear transformation represented by a matrix $S$. The matrix is calculated as: $S = FTG^{−1}$
- F **(Low-Level Engine):** The "reduced form" matrix of the complex model $L$, representing how all its internal noise ripples through the system.
- T : The endogenous abstraction matrix that tells us which low-level variables correspond to which high-level concepts.
- $G^{−1}$ **(High-Level Inverter):** The inverse of the "reduced form" matrix for the abstract model $H$.
The proof for this corollary is based on ==**observational consistency**==. For an abstraction to be valid, the following must hold: $τ(L(e)) = H(γ(e))$ _(Translation: Mapping the low-level results to the high-level must equal the result of running the high-level model on mapped noise.)_

==Because L, H, and τ are all assumed to be linear transformations in this paper, their composition is mathematically forced to be a single, unique linear transformation (S).==
- Even if a variable is "irrelevant" (it doesn't directly map to a concept via $T$), it might still be part of the concept's **Block ($Π$)** if it <mark style="background: #BBFABBA6;">provides the noise or background context for that concept.</mark>
- By using the matrix $S$, researchers can ensure that abstract variables remain "clean" and independent (causally sufficient). <mark style="background: #FFB86CA6;">If the blocks defined by S were to overlap, it would mean two different concepts share the same noise source, leading to hidden confounding.</mark>
---
#### Summary of Concrete blocks (Example 3)

- In the framework of **T-Abstraction**, a high-level concept is not just a single point; it is a **Concrete Block ($Π$)** that organizes the underlying "causal plumbing" of the low-level system. **Example 3** in the sources provides the critical intuition for how "irrelevant" variables are bundled into these blocks to maintain causal integrity.

##### The "Causal Sandwich" Structure
The paper describes a specific topological arrangement where irrelevant variables are "sandwiched" between abstract concepts.
- **Layer 1: The Abstract Parents ($Π_R​(Yparents​$):** The concrete variables that define the "cause" (e.g., Y1​,Y2​).
- **Layer 2: The Irrelevant Conduit:** A chain of low-level variables that the abstraction function T does not directly measure. These form a **T-direct path**—a route that travels only through "noise" variables.
- **Layer 3: The Abstract Child ($Π_R​(Ychild​)$):** The concrete variables that define the "effect" (e.g., Y3​).

> [!TIP] **The "Between" Rule** 
> This result proves that the irrelevant part of a block necessarily lies **between** the relevant variables of the abstract variable and those of its abstract parents.

Intuitively, irrelevant variables are the **wires or pipes** of the system. While the high-level model only cares about the "Input" and the "Output," the system requires these internal conductors to function.
- **Noise Contribution:** These irrelevant variables are the source of the **abstract noise term (U)**.
- **Conduction:** They allow the causal signal from the parents to reach the child's relevant variables without being "intercepted" by any other abstract concept.

The "sandwich" grouping is mandatory to ensure **Causal Sufficiency** at the abstract level.

> [!DANGER] The Risk of Shared Conductors (Example 4)
If an irrelevant "conductor" were shared between two different abstract variables (e.g., Y2​ and Y3​ both relying on the same X2​ for their signal), the following occurs:
> 1. **Dependent Noise:** The abstract noise terms U2​ and U3​ would no longer be independent because they share the same low-level origin (E2​).
> 2. **Confounding:** The two concepts become **confounded**, meaning they will appear correlated in the data even if there is no causal arrow between them.
> 3. **Unfaithfulness:** The high-level model becomes a "lie" because it cannot account for the secret low-level connection, leading to a violation of the **Disjoint Block** requirement.

---
**Theorem 2** : **Block Ordering** 
- establishes a crucial rule for how the "timeline" of a complex system must align with its simplified high-level version.
- states that if you have a high-level model that is a valid abstraction of a low-level model, their **causal sequences must be synchronized**.
**1. The Rule of Block Consistency**
<mark style="background: #FFB8EBA6;">If an abstract variable Y1​ causally precedes Y2​ (meaning Y1​ causes Y2​ or is further "upstream"), then every single concrete variable in the **block** for Y1​ must also causally precede every variable in the **block** for Y2​.</mark>
- If "Diet" causes "Weight" at the high level, then the entire low-level "engine" that handles diet must finish its work before the low-level "engine" for weight starts. There cannot be a low-level diet variable being caused by a low-level weight variable; that would break the high-level logic.
**2. "Leftover" Variables**
Any low-level variables that are **not part of any block** (meaning they are completely ignored by the high-level concepts) must come **last** in the causal order.
- Because these variables don't influence any of the high-level concepts, they are effectively "downstream noise." To keep the math clean, the theorem forces them to the end of the chain so they don't accidentally interfere with the primary causal flow.
This theorem proves that a valid abstraction isn't just a random clustering of variables; it is a **structural alignment**. It <mark style="background: #ADCCFFA6;">ensures that the "gears" of the low-level model turn in the exact same logical direction as the "arrows" in your high-level diagram</mark>. This is what allows the **Abs-LiNGAM** algorithm to "cheat"—it uses the high-level order to instantly know the required low-level order.
---
> [!CHECK] **Summary** **Lemma 5** allows you to "shrink" a complex model down to its essential causal components (the blocks) while guaranteeing that the high-level abstraction remains perfectly accurate.

---
Equation (14): The Concrete Weight Matrix $\mathbf{W}$
$$
W=
\begin{pmatrix}
W_{11} & W_{12} & \cdots & W_{1b}\\
0 & W_{22} & \cdots & W_{2b}\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & W_{bb}
\end{pmatrix}
$$

- This is still just the ordinary $d \times d$ matrix of edge weights between concrete variables — nothing new is being introduced except a way of **slicing it up**.
- Each block $W_{hk}$ is a sub-matrix of size $N_h \times N_k$ (where $N_h = |\Pi(Y_h)|$ is the number of concrete variables in block $h$). It contains **the edge weights from every variable in block $\Pi(Y_h)$ to every variable in block $\Pi(Y_k)$**.
- $W_{hh}$ (diagonal blocks) = the edges *within* block $h$, i.e., how the concrete variables belonging to the same abstract variable $Y_h$ causally affect each other.
- $W_{hk}$ for $h<k$ (off-diagonal, upper-triangular blocks) = the edges *from* block $h$ *to* block $k$, i.e., how variables belonging to an "earlier" abstract variable influence variables belonging to a "later" one.
- The zero blocks below the diagonal exist because the <mark style="background: #ABF7F7A6;">blocks are ordered consistently with the DAG: nothing in a later block can point back into an earlier block</mark> (that would violate the topological order established in **Theorem 2**).

> [!summary] Equation (14) in one line
> Take the full weighted adjacency matrix of the concrete graph, and partition its rows/columns according to which abstract variable each concrete variable belongs to.

---
Equation (15): The Abstraction Transformation $\mathbf{T}$

$$
T=
\begin{pmatrix}
t_1 & 0 & \cdots & 0\\
0 & t_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & t_b
\end{pmatrix}
$$

- $T$ is the $d\times b$ matrix defining $\tau(x) = T^\top x$, ---> it says how each abstract variable is a linear combination of concrete variables.
- Each column $t_k$ (a vector of length $N_k$) contains the coefficients that combine the concrete variables **in block $k$ only** to produce abstract variable $Y_k$.
- This is **block-diagonal** — not block-upper-triangular like $W$ — because of [[Learning Causal Abstractions of Linear SCMs#Lemma 1]] (Disjoint Relevant Variables): ==each abstract variable only depends on concrete variables in its own block, never on concrete variables belonging to a different block. So there's no "cross-block" contribution to $\tau$, hence all the off-diagonal blocks are zero.==
- Within $t_k$, some entries can still be zero — ==those correspond to concrete variables in block $k$ that are in the block (because they causally feed into the relevant variables) but are not themselves "relevant" (i.e., $\tau$ doesn't directly depend on them).== Note in paper : The **last entry** of each $t_k$ must be nonzero, since the block's ordering places all relevant variables after the irrelevant ones within the block.

> [!summary] Equation (15) in one line
> The abstraction function only "looks at" concrete variables within their own block — never mixing information across blocks — because relevant variable sets for distinct abstract variables are disjoint.

==Contrast with $W$ (Eq. 14)==
 - $W$ is **block-upper-triangular**: causal influence *can* flow between blocks (in abstract causal order).
- $T$ is **block-diagonal**: the abstraction mapping *cannot* mix variables across blocks — each abstract variable is built purely from its own block's concrete variables.
---
#### Special Case: Blocks Without Internal Causal Relations

> [!question] Statement
> "The exogenous and the endogenous transformations coincide whenever a block lacks internal causal relations and, consequently, all variables in the block are relevant."

Setup:
- $T$ (endogenous abstraction): maps concrete **values** → abstract values, $\tau(x) = T^\top x$
- $S$ (exogenous abstraction): maps concrete **noise terms** → abstract noise, $\gamma(e) = S^\top e$
- Related via Lemma 6: $s_k = F_{kk}\,t_k = (I - W_{kk})^{-1} t_k$

Condition: "block lacks internal causal relations" : $W_{kk} = 0$
- ==No concrete variable inside block $\Pi(Y_k)$ causes another variable in the same block.==
- Therefore --->  1 — $S = T$
$$
W_{kk}=0 \;\Rightarrow\; F_{kk}=(I-0)^{-1}=I \;\Rightarrow\; s_k = I \cdot t_k = t_k
$$
- ==With no internal edges, there's no noise propagation to account for within the block. Abstracting the *noise* and abstracting the *values* become the same linear operation.==
- As a consequence, all block variables are relevant
From Lemma 3 (Block Composition), $X \in \Pi(Y_k)$ iff:
- (a) $X$ is relevant, **or**
- (b) $X$ is irrelevant but has a **T-direct path** (an internal path) to a relevant variable

Why (b) becomes impossible?
- A T-direct path requires internal edges. If $W_{kk}=0$, no internal edges exist, so condition (b) can never be satisfied. Every variable in the block must satisfy (a) instead.

==Bottom line : When a block has **no internal structure**, it's the simplest possible case: every concrete variable in it is directly relevant to $\tau$, and the exogenous abstraction $S$ collapses to being identical to the endogenous abstraction $T$ — no correction term needed for internal noise propagation.==

