
---
#### Transformers
- Each Transformer layer reads from the residual stream this like ==accumulated memory of the model== and its output is just added to the residual stream. 
- You can think of each layer as an incremental update : everything the model knows is stored in the residual stream
	- what the input is
	- what position you're at
	- whatever intermediate variables are computed ex. this text is positive and floppy. 
	- The model's best guess for the actual outputs
- Residual Stream : compressed mess of stuff.
- Point of this paper is figuring out the right units of analysis by finding the meaningful directions
	- Distributed ---> We do not know which direction things go in, we do not know ==what the meaningful directions are==
	- Pulsemanski proposed exactly this idea of there being sort of privileged non standard basis for interpreting neural networks
- Results of the paper : they are unable to find a basis aligned representation in a pure MLP just trained on their task and they are able to find a perfect representation of the causal variables under a rotated basis and that's just an MLP with relu units [at least in that model it seems like they're still not getting the privileged basis]
- Activation patching : neural network is provided a ‘base’ input, and sets of neurons are forced to take on the values they would have if different ‘source’ inputs were processed
- “The counterfactuals that these interventions create are the basis for causal inferences about model behavior.”
- The key insight is that ==viewing a neural representation through an alternative basis that is not aligned with individual neurons can reveal interpretable dimensions==
---
The results of the **Distributed Alignment Search (DAS)** experiment highlight a fundamental distinction in how neural networks process information, revealing two qualitatively different types of internal reasoning.

- **Shallow Reasoning (NLI Case):** In the natural language inference task, what appeared to be an abstract representation of "lexical entailment" was actually a **"data structure"**. ==DAS discovered that the neural subspace could be **decomposed** back into the specific identities of the individual words.== This suggests the model was simply packaging and passing raw data forward rather than computing a pure relationship.
- **Deep Abstraction (Hierarchical Equality):** In contrast, the hierarchical equality task resulted in representations that were ==**entirely abstracted** from the entities involved==. These representations **could not be decomposed** into the identities of the input objects. DAS proved that the network in this case was not just retrieving data, but was **truly implementing a symbolic, tree-structured algorithm**.

**Key Conclusion for Mechanistic Interpretability:** These results prove that even when two models achieve perfect accuracy, ==one may be acting as a "causal parrot" (storing data), while the other has learned a **pure symbolic relationship** that mirrors human-like algorithmic logic.==

---
Interchange Interventions (Activation Patching)

- In the context of the **Distributed Alignment Search (DAS)** paper, **Definition 2** formalizes the core operation used to test the causal link between internal neural states and high-level symbolic concepts.
- An **Interchange Intervention** is defined as an operation where a model M is provided with a **base input**, but specific internal variables (neurons or subspaces) are forced to take on the values they would have had if the model had instead been processing a different **source input**.
- Mathematically, given a set of source inputs s and target variables X, the intervention replaces the mechanisms of the target variables with the values retrieved from the source.

This operation creates a **counterfactual scenario** to ==observe how the model's behavior changes when a single "thought" or "concept" is transplanted from one context to another.==

1. Base Input (The Context)
- **Role:** Acts as the "shell" or background environment for the experiment.
- **Function:** It provides the default values for every variable in the model that is _not_ explicitly being targeted for a swap.
- **Result:** It establishes what the model would "normally" do in this specific scenario before the intervention.

1. Source Input (The Donor)
- **Role:** Acts as the "donor" of a specific conceptual value.
- **Function:** The model is run on the source input to =="capture" the exact activation values of the internal concept being tested (e.g., the representation of "Equality" or a specific word).==

1. Target Variables (The Site)
- **Definition:** The specific neurons or rotated subspaces where the swap occurs.
- **Interaction:** The values at this site are overwritten: **Base Values → Source Values**.

Purpose in Research : The Causal Litmus Test
- Unlike simple "probes" which only show where information is located (correlation), interchange interventions prove **causal responsibility**. ==If swapping the internal representation of a concept flips the model's output to match the high-level algorithm, you have proven that those neurons are actually "performing the work" of that concept.==

<mark style="background: #FF5582A6;">Metric: IIA (Interchange Intervention Accuracy)</mark>
- The success of these interventions is measured by **Interchange Intervention Accuracy (IIA)**.
- **IIA = 100%:** The model is a **perfect constructive abstraction** of the high-level symbolic algorithm.
- **IIA < 100%:** The model is an **approximate causal abstraction**, and the IIA score represents the degree of "faithfulness".
