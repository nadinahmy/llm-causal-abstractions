- Given that we are only provided with observational data , how can we measure quantities in interventional worlds?
	- We utilize our knowledge of the causal graph at hand, perform a set of operations that convert causal quantities including "do" operators, to expressions that can be evaluated directly from the data given.
	- This conversion is called a "identifying a causal estimand"
- ∑zP(y∣a′,z)⋅P(z) ---> to estimate the interventional distribution correctly, we need to evaluate the conditional distribution of Y given A and Z in every subgroup of our population corresponding to a specific value of Z.
	- Also called ==standardization==
- ![[Pasted image 20260628175033.png|537]]
	- loop over the three sickness levels, compute `P(z)` (the fraction of patients in that level) and `E[Y|A=a,Z=z]` (the mean survival of patients at treatment `a` in that level), multiply, and sum. Run it for `a=1` and `a=0`, subtract, and you get ≈ **+0.27** — matching randomization. The bias is gone.
- parent_adjustment_estimator(data, a=1) is the estimated survival rate in a world where _everyone_ is forced to take the drug, and `a=0` is the world where nobody does
	- The difference is the Average Treatment Effect!
- Naively calculating E[Y|A=1] - E[Y|A=0] secretly compares a population with mostly critical condition patients against another one with mostly "mild condition" patients. Sick people will die more no matter what, and this is what drags down the treated groups survival rate and makes the drug appear harmful. Here we're not truly isolating the drugs effect, we're partly measuring who was sick
	- In other, simpler words : The doctors gave the drug mostly to the _sickest_ patients. So the group of people who got the drug is stuffed full of very sick people — people who were likely to die no matter what. If we just put "got the drug" next to "didn't get the drug" and compare survival, the drug looks _harmful_. Not because it actually is, but because the people who got it were already much sicker. It's an unfair comparison — we're comparing sick people against healthy people.
	- Instead of one big unfair comparison, you compare drug-vs-no-drug _separately within each group of equally-sick patients_.
	- the loop re-weights each severity group by how common it is ==_overall_== rather than how common it is _among the treated/untreated_, and that single change is what removes the confounding effect/bias
- Second implementation in Exercise 2, model-based version
	- First, the survival-lookup step is pulled into an `OutcomeModel` object with `.fit()` and `.predict()`. `fit` builds a small table of survival rates, one per (treatment, sickness) cell; `predict(a, z)` reads a cell back. ==This sits in the place where, on a real problem, we'd drop in a trained model instead.==
	- Second, the average over sickness is done by **Monte-Carlo**: instead of writing out `P(z)` explicitly, you loop over _every patient_, look up the predicted survival for treatment `a` at that patient's sickness level, and take a plain mean. Because you iterate over real patients, common sickness levels automatically appear more often and so get more weight — which reproduces `P(z)` again.
- Exercise 3 - Back door adjustment:
	- What is the problem? Sometimes measuring/observing all confounders is infeasible / expensive, however we can smartly choose a small subset, i.e the backdoor criterion
	- Back-door paths are the source of confounding because they create a fake "association" between A and Y that isn't the treatment actually working. To get a ==clean estimate==, we need to _block_ all of them.
	- ![[Pasted image 20260628220012.png|454]]
		- Small refresher on 3 causal patterns observed in graphs:
		1. **the fork (common cause):** `A ← C → B`. Here `C` causes both, data shows A and B seem related _until_ you hold C fixed, then the relationship [dependency] vanishes.
		2. **the chain (mediator):** `A → C → B`. Here A affects B _through_ C. Same as above: related until you hold C fixed, then unrelated.
		3. **the collider (v-structure):** `A → C ← B`.   A and B are _unrelated_ to start with — but holding C fixed _creates_ a relationship that wasn't there.
		4. for forks and chains, conditioning on the middle variable _blocks_ the connection; for colliders, conditioning _opens_ it. [d-separation framework]
- ==Something that I do not understand at all, and need to research/ask Matej about more as it's very confusing : How is "A ← Z1 → Z2 → Y" a path? do we just ignore orientation here when discussing possible back door "paths"? does this mean that a path in this context is undirected?==
	- ![[Pasted image 20260628220657.png|453]]
- What is the conclusion of exercise 3? 
	- once you hold _both_ `A` and `Z2` fixed, changing `Z1` does nothing to `Y`. In symbols, `P(Y | A, Z1, Z2) = P(Y | A, Z2)`. 
	- The unmeasured variable `Z1` carries no extra information once you know `Z2`. So we can safely ignore it and save money instead of buying the expensive instrument mentioned in the exercise.
- ![[Pasted image 20260628221251.png|422]]
	- parent-adjustment formula (which needs _both_ `Z1` and `Z2`) collapses down to a small formula that needs _only_ `Z2`.
	- Line 1 : for every combination of `(z1, z2)`, take the outcome rate at that combination, weight it by how common that combination is, and add them up.
	- Line 2: we split the sum into 2 nested sums and use the chain rule, easy
	- Line 3: since Z1 and Y are independent, we can remove Z1 from the conditioning set, also easy
	- Reshuffling cause P(y|a',z2) no longer depends on z1, its just a constant w.r.t to z1 so we pull it out, clear
	- ∑z1​​P(z1​∣z2​)=1 because ![[Pasted image 20260628222729.png|444]]
	- finally we're left with parent adjustment using only Z2!
	
Memory check: the ∑  symbol [I keep getting confused by this, noted here for future reference]
- The little variable underneath ∑  indicates *what to loop over*.
- Tiny worked example:
	If z can be 0, 1, 2:
	$$\sum_{z} z^2 = 0^2 + 1^2 + 2^2 = 0 + 1 + 4 = 5$$

- In causal-inference adjustment formulas
$$\sum_{z} P(y \mid a, z)\cdot P(z) = \text{weighted average of the outcome over every value of } z$$
- Why must $\sum_{z} P(z) = 1$? → *the probabilities of all possible values of z must total 100%*
### Exercise 4 - Inverse Probability Weighting
- it keeps the whole population together but **gives each individual a different weight**, so that the confounding cancels out
- weight is built from one quantity, the **propensity score**: `P(A=a | Z)` — the probability that a person got _the treatment they actually got_, given their confounder values.
- The rule is: **weight each person by 1 divided by their propensity score**
- Why inverse?
	- From weight table ---> doctors treated 90% of critical patients but only 5% of mild ones
	-  A **mild patient who took the drug** is _rare_ (only 5% of mild patients did). Their propensity is 0.05, so their weight is 1/0.05 = **20** 
	- A **critical patient who took the drug** is _common_ (90% did). Propensity 0.9, weight 1/0.9 ≈ **1.1**
- What is this fixing? ---> treated group is overloaded with critical patients who get treatment a lot more than mild patients. If we assign a x20 weight to mild patients, IPW rebuilds the treated group so that it contains patients of all severity levels? [or atleast this is what I've understood]. Basically, by doing this, treatment no longer depends on Z anymore.
### Exercise 5
Main pipeline
![[Pasted image 20260628225944.png|283]]
- What's the problem here? ---> the confounder Z is now an image (the check-in photo of the animal). We can't stratify on pixels like before, every animal would basically be its own stratum. So the counting estimators from Ex 2-3 don't work here.
- The key idea [and honestly the whole point of the tutorial]: the estimators are ==modular==. The propensity score `P(A|Z)` is "a function that predicts treatment from the confounder" ---> so we train a CNN to BE that function, then run the exact same IPW estimator from Ex 4 on top of it.
- The dataset used:
	- uses real cat + dog images from CIFAR-10 as the confounder Z
	- the animal's ==species== is the hidden confounder ---> it secretly drives who got treated AND affects the outcome Y
	- but species is never handed to us directly, it's only buried in the pixels --->The CNN has to recover it from the raw image.
- Training the propensity model:
	- train the CNN to predict `A` (treatment) from `Z` (image) with normal cross-entropy loss
	- a model that predicts "how likely was this animal to be treated, given how it looks" ==IS== the propensity score `P(A|Z)`. Estimating propensity = just a supervised classification problem, nothing ==causal-specific== about the architecture.
	- train/test split matters here: fit the propensity model on one half, estimate the ATE on the other half [otherwise the model's overfitting leaks into the answer]
- ==Cool result== : when we plot the estimated propensity scores the distribution is ==bimodal== (two humps), and the humps line up with cats vs dogs. So the network figured out ON ITS OWN that species was the confounder, just from pixels + treatment labels. Nice sanity check.
- Plugging into IPW (ate_propensity_nn):
	- literally the Ex 4 formula, just with the CNN's propensity instead of an oracle
	- one new practical thing ---> ==clipping==. A model can spit out a near-zero propensity, and since IPW divides by it, one bad value can "explode" into a huge weight and dominate the whole estimate
	- result comes out reasonably close to the randomized ground truth. Not perfect cause there's a ==learned model== in the loop ---> getting the right sign + rough magnitude straight from raw images is already the win like mentioned
- Bonus exercise ---> the other half of the toolkit:
	- instead of modeling the treatment (IPW), model the ==outcome== with a CNN
	- the net takes TWO inputs now: the image Z + the treatment A, and predicts `E[Y|A,Z]`
	- then average its predictions over all images, once with A=1 and once with A=0, subtract = the Monte-Carlo parent adjustment from Ex 2, but with a neural outcome model instead of the lookup table
	- so between Ex 5 + bonus we've now neuralized BOTH estimator families
- ==Big takeaway== [this is the actual answer to "where does ML fit into causal inference?"]:
	- the structure of the estimators never changed across the whole tutorial. What changes is the box you slot the model into:
		- IPW ---> CNN estimates the propensity `P(A|Z)`
		- parent adjustment ---> CNN estimates the outcome `E[Y|A,Z]`
	- order of operations ---> causal graph tells you WHAT to estimate, ML gives you the muscle to actually estimate it when Z is an image / text / anything too complex to count.