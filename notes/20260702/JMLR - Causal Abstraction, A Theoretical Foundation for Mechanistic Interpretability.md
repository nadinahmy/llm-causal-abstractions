Bit of a background on Interpretability first cause I know nothing about this field :')
- Modern AI models (like neural networks) are "black boxes."
- However, this is a bit misleading , cause technically, we know  _everything_ about what happens inside a neural network. Every number, every multiplication, every activation — it's all right there in the computer. So in one sense, nothing is hidden.
	- The problem is that this complete low-level knowledge is _useless to humans_. Knowing that "neuron 4,382 multiplied 0.23 by -1.7" tells you nothing about _why_ the model refused a loan application or translated a sentence a certain way.
	- Example: like trying to understand why your friend is sad by looking at a printout of every chemical reaction in their brain. Technically complete, practically meaningless.
- **explainable AI (XAI)** asks: how do we get explanations of model behavior that humans can actually understand?
### Two flavors of interpretability
1. **Behavioral interpretability** only looks at inputs and outputs. You poke the model with different inputs, watch what comes out, and try to summarize the pattern. You treat the inside as truly opaque.
2. **Mechanistic interpretability** is more ambitious: it wants to reverse-engineer what's happening _inside_ the model — to find the actual internal "algorithm" the network is running, described in human-understandable terms.
- Studying the human mind, you could go extremely low-level (biochemistry of neurons) or purely behavioral (what people do given stimuli). But there's a fruitful middle level: what _algorithm_ is the mind running?
### "Just so stories"
- Suppose we claim: "the model classifies this image as a dog because internally it first detects fur, then detects a snout, then combines them." That's a nice, intelligible story. But is it _true_? Is that actually what the network is doing, or did we just invent a plausible-sounding narrative?
- property we want is **faithfulness**: the simple explanation must genuinely reflect what the model does.
---
> [!Abstract] What the paper is trying to answer is : _under what precise conditions does a simple, human-friendly algorithm count as a faithful description of a messy neural network?_ The paper's answer: the theory of **causal abstraction** gives us the mathematical language to state and test this.

- Polysemantic neuron ---> one neuron participates in many unrelated concepts.
- Concepts in neural networks are spread across many neurons [Distributed Representations]
	- the standard tools of causality intervene on whole variables (neurons), but the concepts don't live in single variables — so you can't surgically poke one concept.
---
Main problem in AI: we have "perfect" information about a model (we know every weight and every vector), yet we have almost no understanding of how it actually thinks.
- **Low-level facts:** These are the "micro" details—the raw math, tensors, and activations. While we "know" them, they don't provide a **human-intelligible explanation** of a model's behavior.
- **Black Box:** A system where you can see the input and the output, but the internal "reasoning" process is hidden in complex math.
Mechanistic interpretability ---> process of **reverse-engineering** the internal math of a black box into a **transparent algorithm** that humans can understand.
New concept being introduced : **Interventionals**!
- While a "hard intervention" is like flipping an on/off switch on one variable, an **interventional** is a more complex manipulation used to isolate these "spread-out" concepts.
---
> [!DEFINITION] Definition 3.2
  1.  **Partial Settings (x)**
> - A **partial setting** is a way to describe the values of a ==specific subset of variables (X)== in your model, rather than the entire system at once.
> - **The Math:** If you pick a group of variables X, a partial setting x is a set containing exactly one value for each variable in that group.
> - **Plain English:** It is like focusing on just one part of the world. For example, in a voting model, a partial setting might be "Voter 1 voted YES and Voter 2 voted NO," without worrying about what the other 97 voters did.
   2. **Total Settings (v)**
> - A **total setting** is the special case where you define the value for **every single variable** (V) in the entire system.
> - **The Math:** It is represented as v∈Val^v, covering all variables in the model's signature.
> - **Plain English:** This is a "snapshot" of the entire system. It describes the state of every input, every internal neuron, and every output all at the same time.
   3. **The "Unique Values" Assumption**
> - The definition makes a specific technical assumption: ValX​∩ValY​=∅ **whenever** X != Y.
> - **What this means:** The authors assume that no two different variables share the exact same values. If two variables could both be "True" or "False," the paper "tags" them with the variable name (e.g., TrueX​ vs. TrueY​) to make them unique.
> - **Why they do this:** This allows them to represent a "setting" simply as a **set of values**. Because every value is unique to its variable, if you see the value "6," you immediately know which variable it belongs to without needing extra labels.

---
> [!DEFINITION] Definition 3.3
**1. Projection (**ProjX​(y)**)**
> - A projection takes a setting of many variables (y) and "crops" it down to only show the values for a specific subset (X).
> - **Plain English:** It is like having a spreadsheet of 1,000 neurons but choosing to only look at the column for **Neuron #5**. You are "projecting" the whole state of the network onto that one single unit.
> - **Causal Context:** In your research on interventions, if you have a total setting describing every part of the model, the projection allows you to isolate and talk about just the parts you are manually changing.
> 
**2. Inverse Projection (**ProjY−1​(x)**)**
> - The inverse projection is the opposite: you start with the values of a few variables (x) and find **every possible way** a larger set of variables (Y) could be configured while keeping those specific values fixed.
> - **Plain English:** It creates a "cloud" of all possible total states that are consistent with your specific micro-intervention.
> - **Example:** If you manually fix a high-level "Concept" variable to **True**, the inverse projection identifies every single configuration of the millions of underlying neurons that would result in that concept being True.


> [!NOTE] I think the above is some sort of formal representation of the concept of Restriction sets [[20260623-20260624#==Definition 3.12==]]

---
> [!abstract] Definition 8 — Solution Sets $\text{Solve}(\mathcal{M})$
> **Definition:** The set of all complete configurations of variables $v \in \text{Val}_\mathbf{V}$ where every internal mechanism/equation is satisfied:
> $$\text{Proj}_X(v) = F_X(v) \quad \text{for all } X \in \mathbf{V}$$
>
> **Key properties:**
> - For **acyclic models** (like most feed-forward neural networks), there is always exactly **one unique solution**.
> - It represents the **"final state"** of the system after all causal influences have been calculated.
>
> **Role in abstraction:** The core goal of [[Causal Abstraction]] is to prove that the solutions of a complex low-level model $\mathcal{L}$ map cleanly to the solutions of a simplified high-level model $\mathcal{H}$ *after an intervention is performed*:
> $$\tau\big(\text{Solve}(\mathcal{L}_I)\big) = \text{Solve}\big(\mathcal{H}_{\omega(I)}\big)$$

---
> [!important] Hard vs. Soft Interventions
> **Hard intervention** (Def. 9):
> - **Action:** Replace a variable's mechanism with a **constant value**: $F_X \mapsto (v \mapsto \text{Proj}_X(i))$
> - **Effect:** Breaks the causal link between the variable and its parents entirely.
> - **Analogy:** *Flipping a switch* to a fixed position.
>
> **Soft intervention** (Def. 10):
> - **Action:** Replace a variable's mechanism with a **new function**: $F_X \mapsto I_X$ where $I_X : \text{Val}_\mathbf{V} \to \text{Val}_X$
> - **Effect:** Modifies how the variable responds to the rest of the system without necessarily breaking the link.
> - **Analogy:** *Rewiring* a component so it follows a different logic.
>
> **Hierarchy:** every hard intervention is a special case of a soft intervention (a constant function), and every soft intervention is a special case of an [[Interventionals|interventional]] (a constant functional):
> $$\text{Hard} \subseteq \text{Soft} \subseteq \text{Func}$$
>
> **Interpretability context:** Hard interventions are the basis for [[Activation Patching]] (patching in a specific value), while soft interventions and interventionals allow for more general [[Model Transformations]].

---
### Definition 15 : Intervention Algebra
> [!abstract] Definition 15  
> **Goal:** Identify sets of complex neural manipulations that _act like_ simple, independent switches.
> 
> **Requirement:** The system must be **isomorphic to standard hard interventions**.  
> 
> An **intervention algebra** is a set of actions/manipulations that behaves like ordinary **hard interventions**.
> 
> This means it must obey two rules:
> 
> 1. **Commutativity**  
>     Changing different parts of the model in any order gives the same result.
>     
> 2. **Left-Annihilativity**  
>     The most recent action on a specific part always **overwrites** previous actions on that same part.
>     
> 
> **Why it matters:**  
> - This lets us define **modular features** in LLMs.  
> - Even if a feature is spread across many neurons, if we can find an _algebra of actions_ that obeys these rules, we can treat that feature as a single, high-level causal variable.
> - **The key idea:** Even if the actions are complicated neural manipulations, they count as an intervention algebra if they behave like simple causal switches.
>
> Formally:
> 
> $$(\Lambda, \oplus) \cong (\Phi, \circ)$$
>
> Where:
> - $\Lambda$ = the set of actions we are studying
> - $\oplus$ = how we combine those actions
> - $\Phi$ = the set of standard hard interventions
> - $\circ$ = doing interventions one after another

