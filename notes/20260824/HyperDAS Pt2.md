
---
#### Symmetry in Localization (Get/Set Consistency)

The intuition
- If a concept is genuinely **localized**, the **"get"** operation (reading the concept's value out of the base) and the **"set"** operation (writing a new value in from the counterfactual) should target the ==same features, same hidden representations==.
- If get and set land on different subspaces/positions, the model hasn't found "the concept" — it's found two correlated-but-different things.

Where asymmetry can creep in (Eq 6, token pairing)
- The hyper-network builds keys/values by **concatenating base + counterfactual hidden reps in a fixed order** (base-first).
- Risk: model learns a shortcut keyed to **position** ("slot 1 vs. slot 2") rather than to the **concept itself** — i.e., different rules for base vs. counterfactual roles.

The fix — random flipping
- During training, **randomly flip the concatenation order** in Eq 6 (sometimes base-first, sometimes counterfactual-first).
- Removes the positional shortcut entirely — model can't tell which prompt is "base" from its slot alone.
- Forces **one consistent localization rule**, regardless of role → enforces get/set symmetry.

Symmetric vs. Asymmetric — empirical, not assumed
- Paper trains **both variants** and reports full results for each.
- ==Open question worth checking when writing this up: does symmetry measurably help RAVEL/Iso-score, or is it close to a wash?==

Connects to:
- Same "shortcut vs. genuine mechanism" theme as the [[query-phrase masking fix]] earlier in these notes — another case of the model finding a cheap positional/contextual cue instead of solving the real localization problem.

Example (Vienna/Paris)
- **Base:** "Vienna is in..." (→ "Austria") | **Counterfactual:** "I love Paris"
- **Get** = read the country value out of Vienna's hidden rep ("Austria")
- **Set** = read the country value out of Paris's hidden rep ("France"), write it into base's slot

The loophole without flipping
- Eq 6 always concatenates `[base, counterfactual]` in the **same fixed order** → base always lands in slot 1, counterfactual always in slot 2.
- Model doesn't need to learn "find the country concept wherever it appears" — it can cheat with **two separate positional rules**: "slot 1 → rule A," "slot 2 → rule B."
- ==Nothing forces rule A and rule B to agree== — could get correct outputs from two disconnected mechanisms rather than one genuine "country" concept.

Why flipping closes it
- Randomly swap which prompt sits in slot 1 vs. slot 2 during training → "slot 1 rule" and "slot 2 rule" can no longer be learned separately (Vienna's sometimes in slot 1, sometimes slot 2).
- Only a rule based on **content** ("this hidden state = country"), not **position**, survives training on both arrangements.
- → forces get and set to converge on the same underlying mechanism.

One-line takeaway
- ==Fixing a loophole, not a bug: fixed concatenation order lets the model fake localization via position as a crutch instead of actually finding the concept.==

---
#### MDAS (Multi-task Distributed Alignment Search) — the missing link between DAS and HyperDAS

Where it comes from
- Introduced in the RAVEL paper (Huang, Wu, Potts et al.) — RAVEL's own proposed baseline, and what HyperDAS is directly benchmarked against.

The problem it solves (that vanilla DAS ignores)
- A single entity token (e.g., "Vienna") encodes **many attributes at once** — country, continent, language, latitude — all superposed in the same vector.
- Vanilla DAS trains **one `R` per attribute, in isolation** — loss only checks ==Cause== ("did intervening on this subspace correctly change the target attribute?"). ==Nothing prevents different attributes' subspaces from overlapping or bleeding into each other==, since DAS never checks.

What MDAS does differently
1. **Trains all attributes' subspaces jointly**, as disjoint chunks of one shared rotated space — rather than training each attribute's `R` separately, with no coordination between them.
2. **Adds a second loss term — Iso (isolation)**, alongside Cause:
   - **Cause:** did intervening on attribute A's subspace correctly change A?
   - **Iso:** did intervening on A's subspace leave every *other* attribute untouched?
   - Optimizing both together ==actively punishes cross-attribute leakage during training==, instead of just hoping it doesn't happen.

Concrete example (Vienna)
- MDAS tries to find one clean split of the residual stream into a country-slice, continent-slice, language-slice, etc., such that swapping in Paris's country value changes *only* country — verified during training via the Iso loss.

What MDAS still has in common with vanilla DAS
- ==Token position is still manually chosen== (e.g., always the entity's last token) — MDAS only automates **which dimensions** belong to which attribute, not **where** in the sentence to intervene.
- This is exactly the piece HyperDAS goes after next → explains why the DAS/MDAS/HyperDAS comparison table lines up the way it does (position: manual → manual → learned).

Where it sits in the lineage
- **DAS** → single attribute, subspace hand-trained, position hardcoded.
- **Boundless DAS** → automates subspace *size* only, still single-attribute, position still hardcoded.
- **MDAS** → multiple attributes jointly, Cause+Iso loss, disjoint subspaces — but position still hardcoded.
- **HyperDAS** → automates position too (Section 3.2), no retraining per concept.

#### What "manually selected token position" means, concretely (the "nna" example)

Tokenization, not concept
- "Vienna" isn't one token to the model — it's split into **sub-word pieces** (e.g., roughly "V" + "ienna," with "nna" as the paper's example of the last piece).
- There's no single vector for "the entity Vienna" — there's a **sequence** of vectors, one per sub-word piece.

What was manually fixed
- MDAS's designers picked, ahead of time, a **fixed rule**: always intervene on the entity's **last** sub-word token.
- For Vienna → that's "nna." Not searched, not learned — hardcoded by the researchers.

<mark style="background: #FF5582A6;">Why last token (the reasoning, not a guarantee)</mark>
- Causal attention only looks **backward** → by the last piece of "Vienna," its hidden state has had a chance to accumulate info from the earlier pieces too.
- ==A reasonable default guess for "where the entity's full info lives" — but still a guess baked in, not discovered==.

Why this matters for the DAS → MDAS → HyperDAS lineage
- This is the exact manual step **HyperDAS Section 3.2 (token pairing, matrix G) removes**.
- HyperDAS doesn't assume "always last piece" — it **learns, per example**, which base/counterfactual token pair actually carries the concept, via attention.
- ==Concrete, tokenizer-level version of the "position: manual vs. learned" row in the DAS/MDAS/HyperDAS comparison table.==
---
#### Section 4.1 — Layer-Specific Intervention Behaviors of HyperDAS

Setup
- Study 3 layers in detail: **shallow (L7), middle (L15), deep (L29)** (Figure 4).
- For each, categorize which base-sentence token HyperDAS picks for intervention: **BOS**, **Entity Token**, **JSON Syntax** (punctuation like `{`), or **Others**.

Counterfactual side — stable across all layers
- HyperDAS consistently picks the **entity token** (e.g., "Paris") at every layer studied.
- ==Attribute info is reliably in the entity token from early on — nothing layer-dependent here.==

Base side — highly variable, the interesting part
- **Shallow (L7):** "turbulent" — near-random choices, sometimes even the **BOS token**. Attribute info hasn't consolidated into one clean location yet this early.
- **Middle (L15):** converges on the **entity token** — matches the standard prior-work assumption (Meng et al., Geva et al.: "entity info lives in the entity token"). ==Also where disentanglement performance peaks== (ties to Fig 3b).
- **Deep (L29):** shifts to **syntax tokens** (e.g., JSON punctuation) — ==previously unknown to store attribute info==. <mark style="background: #BBFABBA6;">A genuinely novel finding, not something prior heuristics would have predicted.</mark>

Why this matters?
- ==Stronger argument for automation over hand-picked heuristics than efficiency alone==: HyperDAS doesn't just save labor over manually choosing "last entity token" (the MDAS/DAS heuristic) — it **discovers real structure a human wouldn't have hypothesized** (deep-layer syntax tokens mattering).
- The "last entity token" heuristic used by MDAS/DAS would have **completely missed** the deep-layer behavior.
- Connects to [[MDAS (Multi-task Distributed Alignment Search)]] — this is the empirical payoff of automating what MDAS still does by hand.

---
#### Householder Vector Analysis — "a window into attribute features"

The core idea
- `v` (from Step 3, `v = MLP(e_E^(N))`) is normally just plumbing — an ingredient for building `H` → `R`.
- Since a fresh `v` is generated **every forward pass, per concept**, you can collect many of them and treat them like embeddings:<mark style="background: #FF5582A6;"> do same-attribute vectors cluster, and different-attribute vectors separate?</mark>

What they did
1. Ran test examples, saved `v` each time, tagged by attribute (country, birth year, field of study, etc.).
2. 1,000 samples per attribute category.
3. **Figure 5:** PCA'd all vectors to 2D, plotted by attribute.
4. **Figure 6:** average cosine similarity — within-attribute pairs vs. cross-attribute pairs.

What they found
- Same-attribute vectors cluster **more tightly** than cross-attribute vectors (though overall similarity is high across the board — all from the same entity type, e.g. all "city" attributes).
- ==Notable exception: Latitude and Longitude vectors are highly similar to each other== — intuitively sensible, both are "numeric geographic coordinate" attributes.

Why this matters (not just a sanity check)
- `R` is **regenerated from scratch every forward pass** via the Householder trick — never directly trained/stored per-attribute like vanilla DAS's `R`. No obvious way to "go inspect the country-subspace" otherwise.
- ==This clustering is the evidence that fills that gap==: shows the network implicitly learns a **consistent geometric direction per attribute** in `v`-space, without ever being forced to store one.
- Supports the paper's core disentanglement claim: <mark style="background: #FF5582A6;">the Householder mechanism isn't improvising a different rotation by accident each time — it's steering `R` toward genuinely attribute-specific, reusable regions of space.</mark>

Connects to:
- [[Symmetry in Localization (Get/Set Consistency)]] and [[Masking of the Base Prompt]] — same broader theme of the paper: checking whether HyperDAS is doing something *real* vs. gaming the metric.

---
#### Figure 7 — Sparsity Loss Ablation (Intervention Patterns)

Setup
- Same base/counterfactual (Occupation-type) example, run through **3 HyperDAS variants**, each trained with a different amount of `L_sparse` (Eq 13).
- All 3 score ~94% Disentangle Score using **weighted (soft)** interventions — identical on the training-time metric.

What the heat map shows (matrix G, Eq 8/9)
- **Rows** = counterfactual tokens | **Columns** = base tokens.
- Color = `G_(b,c)` value, 0→1 (dark = ~0 contribution, bright pink/orange/cream = ~1, dominant contribution).
- **[SELF] row** = special "leave alone" option (Eq 7) — bright here means "not intervened on," opposite meaning from every other row.
- Softmax is **column-wise** → each column's values sum to 1, so the brightest cell per column is whichever option (a counterfactual token, or [SELF]) currently "wins" that base token.

The three regimes
1. **No sparsity loss:** influence spreads across **many** base tokens loosely (many-to-one blending) — still scores fine under soft eval, but never commits to one location.
2. **Too much sparsity loss:** pairwise attention **collapses entirely** — model blends nearly all hidden states together, no real selection pattern at all, yet still scores well under soft eval.
3. **Correctly-tuned sparsity loss (screenshot example):** one **clearly dominant peak per row**, not necessarily a single lit pixel — e.g. counterfactual "therapist" row has its brightest cell unambiguously at base "occupation," with weaker (but nonzero) activation elsewhere. [SELF] row lit broadly everywhere *except* dimmer at "occupation," where "therapist" is winning the tug-of-war.

Why this matters — the punchline
- All three variants look equally good on the **soft metric**.
- The difference only appears once **double-argmax snapping** (Eq 14–15) forces a hard 0/1 decision at test time:
  - No-sparsity model → breaks (influence too smeared to snap to one clear winner).
  - Over-sparsity model → breaks (never had a real preference to snap).
  - ==Correctly-regularized model → survives cleanly==, since it already had one dominant peak per row before snapping.
- Concrete post-snap result for the example: **base "occupation" ← replaced by counterfactual "therapist"; every other base token → [SELF]=1**.

Why it matters?
- ==Clean example of a metric giving false confidence== — 3 models score identically during training, but only one is doing genuine one-to-one localization.
- Same shortcut-vs-genuine-mechanism theme as [[Masking of the Base Prompt]] and [[Symmetry in Localization (Get/Set Consistency)]].

---
#### Faithfulness Concern — "Interpreting vs. Editing" (the umbrella framing)

The core tension
- Two very different things an intervention could be doing, indistinguishable from the outside:
  1. **Interpreting:** finding a feature that was **already there**, genuinely responsible for the concept.
  2. **Editing/steering:** having enough raw power to **force** the right output regardless of whether that spot actually mediates the concept.
- ==If the output changes correctly, that alone doesn't tell you which one happened.==

Why HyperDAS specifically is at risk
- It's a **learned, supervised** system — a full hyper-network trained end-to-end to maximize RAVEL score.
- Anything trained that way is incentivized to find **whatever exploit raises the score** ("hacking the evaluation"), not necessarily the intended genuine localization.
- ==More power/flexibility in the hyper-network = more room for such a shortcut, not less.==

"Out-of-distribution interventions"
- If the patched hidden state is a value the target model would **never naturally produce**, the model isn't being "read" — it's being pushed into an unnatural regime and told to output something. That's editing, not interpreting.

"Constraining optimization flexibility" — the umbrella over everything else in these notes
- **Sparsity loss** (Fig 7) — blocks the blend-everything shortcut.
- **Masking the query phrase** — blocks skipping hard mismatch cases.
- **Symmetry** (flipped concatenation order) — blocks a positional shortcut decoupling get/set.
- **Orthogonality of R** (Householder) — the intervention must be a rotation, not an arbitrary edit.
- ==Each fix deliberately removes flexibility the hyper-network would otherwise exploit to fake success rather than genuinely localize.==

Relevance to thesis
- ==Direct source-backed statement of the same problem behind my commutativity-criterion argument==: a supervised interpretability method's score alone can't distinguish "found real structure" from "learned to game the metric."

---
#### Figure 8 — Symmetric vs. Asymmetric Token Selection (empirical payoff of Symmetry)

Setup
- City-domain entities exactly **3 tokens long** (positions comparable across examples).
- Per example: record which token position gets picked in the **counterfactual** prompt (top) vs. **base** prompt (bottom).
- Two panels: **symmetric** (trained with random-flip) vs. **asymmetric** (no flip).

What the histograms show
- **Symmetric:** consistently favors the **last entity token** — for both base AND counterfactual. Same rule regardless of role.
- **Asymmetric:** counterfactual still favors **last entity token** (≥95%), but base shifts to favoring the **second-to-last entity token** instead. ==Different position depending on role.==

Why "different position depending on role" is a real problem, not a technicality
- The interpretability claim is: "found *the* place where country lives" — a claim about the model, implying one causal variable usable in both directions.
- If get needs position N and set needs position N-1, there isn't one shared location — there are **two disconnected mechanisms** that each happen to correlate with the right answer well enough to pass the metric.
- ==Analogy: claiming to find "the thermostat" but needing to read one dial and turn a totally different one to change the temperature — you haven't found the control, you've found two separate things.==
- Ties directly to the faithfulness section: passing the metric via two shortcuts ≠ uncovering real causal structure.

<mark style="background: #FF5582A6;">The performance cost</mark>:

| | Asymmetric | Symmetric |
|---|---|---|
| Average Disentangle | 84.7 | 76.9 |
| Verb domain | 93.0 | **42.3** |
| All-domains-joint | 80.7 | **54.8** |

How to interpret the drop:
1. **Shortcut-removal reading:** asymmetric's higher score may be partly inflated by the get/set decoupling shortcut; symmetric's lower score could be more honest.
2. **Harder-optimization reading:** forcing one rule to work both directions shrinks the solution space — could just be a harder objective, independent of any "cheating."
3. **Domain-mismatch confound:** verb domain's especially large collapse (93→42) may reflect base/counterfactual token structure mismatch specific to verbs (tense/conjugation), not a general shortcut story.

==Summary: the paper shows symmetry achieves its stated mechanistic goal (Fig 8), but leaves open whether the performance drop reflects removing an illegitimate shortcut vs. just a harder optimization problem — an honest gap, directly relevant to my thesis's own validation question (metric success ≠ genuine mechanism).==

---
#### Example — Get/Set Position Mismatch (Vienna/Paris, concrete)

Setup
- Base: "Vienna is in..." → `[Vi, en, na]` | Counterfactual: "I love Paris" → `[Par, i, s]`
- Target concept: country

What the asymmetric model does (per Fig 8)
- **Counterfactual side:** grabs **last token** → `s`
- **Base side:** grabs **second-to-last token** → `en` (NOT `na`)
- Actual intervention: patch whatever's in `s` into `en` — two different relative positions, not `na`→`s`.

The disconnected-shortcuts story
- **Rule A (read/get):** "the complete word's last sub-word token reliably holds the value" — plausible, could be genuinely true.
- **Rule B (write/set):** "the second-to-last token is a good injection point that doesn't immediately break the sentence's syntax" — possibly just a training-gradient artifact, not "country" at all.
- Both rules, independently, get "France" on the training distribution — but neither is really *the* country feature. Two different learned conveniences that happen to cooperate, not one shared mechanism.

The concrete tell that something's off
- ==Querying `na` (the standard "last entity token" assumption from MDAS/prior work) for a read-only lookup, outside RAVEL's exact patch format, might get nothing meaningful== — because the asymmetric model's real base-side location was `en`, only in the narrow "write" role, never verified as an honest "read" location.

What symmetric does instead
- Converges to using **`na` for both directions** — read Paris's value from `s`, but also read Vienna's *own* value from `na` (not `en`), and write into `na` too.
- One consistent claim: "country lives at the entity's final sub-word token, full stop" — same claim MDAS/prior heuristics assumed, now **verified by convergence under the flip constraint** rather than assumed by researchers.

Takeaway
- ==Concrete difference between "found one real thing" (symmetric, in principle) vs. "found two convenient but different things that cooperate" (asymmetric) — even though asymmetric currently wins on the benchmark numbers.==

---
### Appendix notes

#### Table 3a vs. Appendix A.1 — Per-Domain Specialists vs. Single Joint Model

Setup #1 — separate model per entity-type split (Table 3a, main results)
- Train **5 independent HyperDAS models**, one per RAVEL domain (cities, Nobel laureates, occupations, physical objects, verbs) — each sees only its own domain's data.
- ==Same setup MDAS was trained/evaluated with== → fair apples-to-apples comparison for the headline numbers.

Setup #2 — single model across all splits (Appendix A.1)
- Train **one HyperDAS model** on the combined data from all 5 domains at once, then evaluate on each domain's test set.

Why they run the second experiment
- Setup #1 only shows HyperDAS *can* beat MDAS when given the same narrow, domain-specialized advantage MDAS always had.
- ==Doesn't test whether the hyper-network's natural-language concept description actually generalizes==, vs. just memorizing domain-specific patterns.
- Real question: can **one model do the job of five** — using the concept description as its steering mechanism — rather than needing five separately-trained specialists (which MDAS structurally requires)?

Result
- Joint model: **80.7** vs. domain-specialized: **84.7** (only ~4.0% worse).
- Still beats MDAS's domain-specific baseline (**76.0** average).
- ==Generalization claim holds up reasonably well== — one HyperDAS model handling all 5 domains loses only a little vs. five specialists, and still outperforms MDAS.

