
---
#### Research Question
- Can causal abstraction theory be used to validate whether internal LLM representations faithfully encode human-level semantic concepts under intervention, rather than just observationally?
- ==Can the internal representations of an LLM be viewed as causal abstractions of high level semantic variables?==
- Under what conditions can internal LLM representations be considered causal abstractions of high-level semantic variables?
	- More formally:
	- Can causal abstraction theory be used as a criterion for evaluating whether internal LLM representations faithfully encode human concepts?
		- Candidate methods:
			- Abs-LiNGAM
			- DAS
			- .........
- Can we use causal abstraction theory to prove that a high level concept is represented abstractly by low level LLM activations ?
- Can I learn an abstraction function , then apply activation patching at the low level LLM and an intervention on the abstracted model and use the commutativity diagram to discover whether this abstraction is a faithful one (interventionally not only observationally)
----
#### Motivation
- Current mechanistic interpretability methods often identify representations associated with human concepts. However, these representations are frequently ==validated observationally.== Causal abstraction theory offers a notion of when a high-level variable faithfully represents a lower-level system under interventions. This gives us a possible framework for validating semantic representations beyond correlation.
---
#### Method
- The high level idea I have for this method so far is :
```
Represent a "concept" ---> Learn abstraction ---> Intervene ---> Test commutativity ---> measure abstraction error [assuming I'm aiming for approximate abstraction if exact abstraction is too strong for an LLM]
```
---
#### Questions I need to answer
- What is DAS? Does it already do what I'm trying to do here? Is what I'm trying to do already out there or is this really a fresh untouched research area?
- What is linear probing? Why is it similar / different to what Im trying to do?
- The whole goal is to check whether the causal abstraction mapping is faithful under intervention rather than just being observationally correlated.
- Does this recovered representation obey causal abstraction conditions ? ---> Causal abstraction validity
---
#### Why isn't this already solved?
- It seems to me that mechanistic interpretability methods already include many causal intervention techniques.
- The question is whether causal abstraction theory offers a ==stronger== or more principled notion of what it means for a representation to correspond to a high-level concept.
	- Does it provide guarantees—such as intervention consistency, commutativity, or abstraction validity—that existing methods do not explicitly target?