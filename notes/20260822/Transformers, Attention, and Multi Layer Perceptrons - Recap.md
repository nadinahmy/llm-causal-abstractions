3Blue1Brown video notes

---
## 1. Transformers
- GPT : Generative Pre-Trained Transformer
- Transformer : Specific kind of neural network, a machine learning model, core invention underlying current boom in AI
- Many different models can be built with transformers
	- Model that take audio and produces transcript (Voice to Text)
	- Model that produces synthetic speech from text (Text to Voice)
	- Model that take text description and produce images (Text to Image)

The full data flow, step by step
1. **==Tokenization==** — input text is broken into small pieces ("tokens": words, word-fragments, or common character combinations). For images/audio, tokens would be patches/chunks instead.
2. **Embedding** — each token becomes a **vector** (list of numbers) meant to encode its meaning. Picture these as coordinates in a high-dimensional space — words with similar meanings land near each other.
3. ==**Attention block** — lets the vectors "talk to each other" and update their values based on context.==
   - Example: "model" in *"a machine learning model"* vs. *"a fashion model"* — same starting token, different intended meaning. Attention is what figures out which words in context are relevant to updating which other words' meanings, and how.
   - =="Meaning" here is entirely encoded in the numbers inside each vector — nothing mystical beyond that.==
1. **MLP / feed-forward block** — a different operation where, unlike attention, **vectors do NOT talk to each other** — each one passes through the same operation independently, in parallel. Harder to interpret directly; loosely described as "<mark style="background: #ABF7F7A6;">asking a long list of questions about each vector, then updating it based on the answers.</mark>"
2. **Repeat** — attention block → MLP block → attention block → MLP block, alternating, many times over.
3. ==**Final prediction** — the hope is that by the end, all the essential meaning of the whole input has been "baked into" the very last vector in the sequence.== A final operation converts that last vector into a **probability distribution over all possible next tokens**.
4. **Autoregressive generation** — sample from that distribution, append the chosen token, then repeat the whole process to generate the next one, and so on (this is literally what early GPT-3 demos looked like — autocompleting text from a seed prompt).
5. **Turning this into a chatbot** — start with a **system prompt** establishing "a user is talking to a helpful AI assistant," append the user's actual question as the first turn of dialogue, then have the model predict what a helpful assistant would say next. (An additional training step is needed to make this work well in practice, not detailed in the video).

Two key things to remember:
- **Attention block** = vectors exchange information with each other, updating meaning based on context.
- **MLP / feed-forward block** = vectors are each processed independently, in parallel, with no communication between them.

Connecting to HyperDAS / prior work
- This "attention block" is the general concept underlying every Q/K/V formula seen in the HyperDAS paper (self-attention within the hyper network, cross-attention to the target model's hidden states).
- "All of the operations... look like a giant pile of matrix multiplications" — same spirit as the handcrafted MLP's W1/W2/W3 matrices worked through earlier, just at a much larger scale and with attention's added query/key/value structure.
- ==The "last vector in the sequence carries the essential meaning" idea directly parallels HyperDAS's own convention of using the **last token position** (`e_E^(N)`) as the concept-summary vector `v`.==

Deep learning : describes any model where you're using data to somehow determine how a model behaves. [Learn from Data]
	- Intuition and Pattern Recognition

---
What machine learning is, at its core
- ML = using **data** to determine how a model behaves, rather than explicitly coding the procedure by hand (the old, pre-ML approach to AI).
- Set up a flexible structure with **tunable parameters** ("knobs and dials"), then use many input/output examples to tune those parameters until the model mimics the desired behavior.
- **Simplest example — linear regression:** predict house price from square footage. The "model" is just a line, described by 2 continuous parameters (slope, y-intercept). Fitting the line to the data = the whole training process.

Deep learning specifically
- Deep learning models scale this idea up massively — GPT-3 has **175 billion parameters** (vs. linear regression's 2).
- Having many parameters isn't automatically useful — a huge model can easily **overfit** the training data or become **intractable to train**.
- What unifies deep learning models: they all use the same training algorithm, **backpropagation**, and they all follow a specific structural **format** that makes backprop work well at scale. Understanding this format explains many transformer design choices that would otherwise seem arbitrary.

The required format
1. **Input must be an array of real numbers** — a list, a 2D array, or (very often) ==higher-dimensional arrays, generally called **tensors**==.
2. Data flows through **many layers**, each layer itself just another array of real numbers, until the **final layer** = the output (e.g., <mark style="background: #FF5582A6;">for a text model: a probability distribution over all possible next tokens</mark>).
3. Parameters are called **weights** — the defining feature of deep learning models is that these parameters only ever interact with the data via **weighted sums** (plus some non-parameter-dependent non-linear functions sprinkled in).
4. Weighted sums are almost never written out explicitly one at a time — instead they're packaged as **matrix-vector products**. Multiplying a matrix by a vector produces an output where each component is itself a weighted sum — same math, cleaner notation. Practically: think of matrices full of tunable parameters that transform vectors drawn from the data.

GPT-3's numbers:
- 175 billion weights, organized into **~28,000 distinct matrices**.
- Those matrices fall into **8 categories**.
- GPT-3 specifically chosen as a reference point since it's the first LLM to capture mainstream attention (also: newer models' exact parameter counts are often kept private by companies now).

Concepts to hold onto going forward
- **Weights** = the model's actual "brain" — learned during training, determine behavior.
- **Data being processed** = whatever specific input is fed in for a given run (e.g., one snippet of text) — NOT learned, just encodes that particular input.
- Nearly all actual computation inside a model like ChatGPT is matrix-vector multiplication — easy to get lost in the scale of billions of numbers, but this weights-vs-data distinction is the anchor to keep track of what's "the model" vs. "what's currently being processed."

Connection to DAS
- This weights/data distinction maps directly onto DAS: the target model's own weights stay **completely frozen** throughout DAS training — only the rotation matrix `R` (itself just another set of weights, specific to the intervention) gets trained. The base/source/counterfactual inputs are the "data being processed," in this video's terms.
- "Each layer is an array of real numbers, transformed by matrices" is literally what was traced by hand in the handcrafted MLP (W1, W2, W3 matrices transforming the input vectors layer by layer).

The embedding matrix (W_E) — the model's first pile of weights
- Model has a fixed **vocabulary** (list of all possible tokens) — e.g., GPT-3's is 50,257 tokens.
- **Embedding matrix W_E** has **one column per vocabulary word/token** — that column is the vector that word turns into.
- Like all weight matrices: starts random, gets learned from data during training.
- **GPT-3 numbers:** vocab size 50,257 × embedding dimension 12,288 = **~617 million weights** — the first entry in the running tally toward GPT-3's full 175 billion.

Embeddings as points/directions in high-dimensional space
- <mark style="background: #BBFABBA6;">Think of each word's vector as **coordinates of a point** in a high-dimensional space</mark> (GPT-3: 12,288 dimensions).
- Key finding from training: ==the model tends to settle on embeddings where **directions in the space carry semantic meaning** — nearby points = similar meaning== (e.g., words near "tower" all give "tower-ish" vibes).

How Directions encode specific semantic relationships
- **Classic example:** `woman − man ≈ king − queen` (as vectors/directions). So `king + (woman − man)` lands near "queen" — i.e., one direction in the space seems to encode **gender**.
  - Noted in the video: the true "queen" embedding is actually a bit further off than this suggests, likely because "queen" isn't used in training data as simply "feminine king." Family-relation examples illustrated the idea more cleanly in the video's own experiments.
- **Country/leader example:** `Italy − Germany + Hitler ≈ Mussolini` — directions for "Italian-ness" and "WWII axis leaders" seem separately encoded.
- **Cuisine example:** `Germany − Japan + sushi ≈ bratwurst`.
- Nearest-neighbor check: "cat" landed close to both "beast" and "monster."

Dot product = a measure of alignment between vectors
- **Computationally:** multiply corresponding components, sum the results (same operation used throughout — weighted sums, matrix-vector products).
- **Geometrically:**
  - ==positive → vectors point in similar directions==
  - ==zero → perpendicular==
  - ==negative → point in opposite directions==

Worked example: a "plurality direction"
- Hypothesis: `cats − cat` might represent a general "plural" direction in the embedding space.
- Test: dot-product this direction against embeddings of singular vs. plural nouns — plural nouns consistently score higher, confirming the hypothesis.
- extension: dot-producting this same direction against "one," "two," "three," ... gives **increasing** values — as if the model has a quantitative internal sense of "how plural" a word is.

Summary of this section:
- This is the same underlying idea as `randvec()` from the DAS tutorial — assigning entities numeric vector representations — except here the vectors are **learned from real language data** (so directions become semantically meaningful) rather than randomly generated (where directions are arbitrary and meaningless).
- Dot products as "alignment scores" is exactly the mechanism behind attention's query/key comparison worked through in the toy "cat sat mat" example — same operation, same geometric interpretation, just applied there between Q and K vectors rather than between two word embeddings directly.
- This is also the conceptual ancestor of HyperDAS's Householder-transformation subspace-selection step: ==the idea that specific **directions** in a high-dimensional space can be dedicated to specific concepts is the same intuition underlying why a learned rotation matrix `R` can isolate a "country" or "birth year" subspace==.

Embeddings soak up context as they flow through the network
- A word's vector doesn't just represent that single word in isolation — <mark style="background: #FF5582A6;">it also encodes position information (covered later), and critically, it has the capacity to absorb context.</mark>
- Example: the vector for "king" might start generic, but by the end of the network could point in a direction encoding "a king who lived in Scotland, who took the throne by murdering the previous king, described in Shakespearean language" — all baked into one vector via the blocks it passes through.
- **Initially**, right after the embedding-lookup step, each vector encodes only its single word's meaning, with zero input from surrounding words. ==The **entire point of the rest of the network** (attention + MLP blocks) is to let each vector absorb progressively richer, more specific meaning from its context.==

Context size — a hard limit
- The network processes a **fixed number of vectors at a time** — the **context size**. GPT-3's was 2048.
- Data flowing through always looks like an array of `context_size` columns × `embedding_dim` rows (2048 × 12,288 for GPT-3).
- <mark style="background: #D2B3FFA6;">This limit is exactly why long chatbot conversations can feel like the bot "loses the thread" — once you exceed the context size, earlier parts of the conversation are no longer available to inform predictions</mark>.

Unembedding — turning the final vector into a prediction
- The output needed is a **probability distribution over all possible next tokens**.
- Two steps:
  1. **Unembedding matrix (W_U)** maps the very **last** vector in the context to a list of 50,000-ish values (one per vocabulary token). Same structure as the embedding matrix, just "swapped" — one row per vocab word instead of one column. Random at initialization, learned during training.
  2. **Softmax** normalizes that raw list into a valid probability distribution.
- **Why only the last vector, when thousands of others (with their own context-rich meanings) exist in that final layer?** Not wasteful — it turns out to be much more *training-efficient* to have every vector in the final layer simultaneously predict what comes right after it.
- **GPT-3 parameter count so far:** Embedding matrix (617M) + Unembedding matrix (617M) ≈ **just over 1 billion**, out of the eventual 175 billion total.

Softmax — the mechanics
- **Goal:** turn an arbitrary list of numbers (which can be negative, huge, and don't sum to 1) into a valid probability distribution (all values between 0 and 1, summing to 1).
- **How:** raise `e` to the power of each number (makes everything positive), then divide each by the sum of all of them (normalizes to sum to 1).
- **Behavior:** the largest input value dominates the output distribution (pushed close to 1), but it's "soft" — if multiple values are similarly large, they all get meaningful weight, and the whole thing changes continuously as inputs vary (not an abrupt jump like picking a strict max would be).

Softmax with temperature (T)
- Adds a constant `T` into the denominator of the exponent.
- ==**Larger T** → more weight given to lower values → distribution becomes more uniform/random.==
- ==**Smaller T** → bigger values dominate more aggressively.==
- ==**T = 0** (extreme case) → all weight goes to the single maximum value (fully deterministic, always picks the most likely token).==
- **Demonstrated with GPT-3 story generation** ("once upon a time there was A"):
  - T=0 → predictable, derivative Goldilocks-esque story.
  - Higher T → more original opening, but risks degenerating into nonsense.
- Note: the actual API caps temperature at 2 — an arbitrary product constraint, not a mathematical one, <mark style="background: #FF5582A6;">to avoid visibly nonsensical output.
</mark>

Terminology: logits
- The **raw, unnormalized** output of the final matrix multiplication (before softmax is applied) is called the **logits**
- Softmax's *output* = probabilities; softmax's *input* = logits.
---
- **This directly explains `.argmax(1)` and `preds[0]` from the DAS tutorial** — those "2 raw scores per example" being argmax'd were exactly this: **logits**, the pre-softmax output. The DAS tutorial's `classification_report` workflow used argmax (hard selection) rather than softmax (soft distribution) since it only needed a final hard prediction, not a probability distribution to sample from.
- The "context size" limit is conceptually why HyperDAS's task-specific setup (single flat vector inputs, no real token sequence) sidesteps this whole issue entirely — no context window to worry about when there's no sequence at all.
---
## 2. Attention
Recap from the embeddings chapter
- Attention first introduced in the 2017 paper *"Attention is All You Need."*
- Goal of the model: predict the next token given input text.
- Directions in the embedding space carry semantic meaning (e.g., a "gender" direction: masculine noun + that direction ≈ corresponding feminine noun).
- ==Transformer's overall aim: progressively adjust embeddings from encoding a single word to encoding rich **contextual** meaning.==

Motivating examples (why attention is needed)
- **"Mole" example:** "American shrew mole," "one mole of carbon dioxide," "biopsy of the mole" — after the pure embedding-lookup step, "mole" gets the **identical** vector in all three, since lookup has no context awareness. Only attention lets surrounding words update that vector toward one of several "mole" meanings.
- **"Tower" example:** generic "tower" embedding, preceded by "Eiffel" → should shift toward Eiffel-Tower-specific meaning (correlated with Paris, France, steel). Preceded further by "miniature" → shifts again, away from "large, tall" associations.
- **General principle:** <mark style="background: #FF5582A6;">attention lets the model move information from one embedding to another — potentially far apart in the sequence, and potentially much richer than a single word's worth of information</mark>.
- ==**Mystery novel example:** if input is most of a mystery novel up to "...therefore the murderer was," the **final vector** (which started life as just the embedding of "was") must, by the end, encode everything from the whole context relevant to predicting the next word — this is only possible because of repeated attention updates.==

The attention pattern — step by step
**Setup:** worked example sentence — "a fluffy blue creature roamed the verdant forest." Goal (for this single head): have adjectives ("fluffy," "blue") update the meaning of their noun ("creature"). ==Embeddings (denoted `e`) encode both word identity **and position**==.
### 1. Queries — "what am I looking for?"
- Each noun (e.g., "creature") effectively asks: *"are there any adjectives in front of me?"*
- This question = a **query vector** `q`, computed as `q = W_Q · e` (a learned matrix `W_Q` multiplied by the embedding).
- <mark style="background: #ABF7F7A6;">Query space is much smaller than embedding space</mark> (e.g., 128 dims vs. 12,288 for GPT-3).
- `W_Q` applied to **every** embedding in the context → one query vector per token (not just nouns — but conceptually, for this example, imagine it's mainly shaping the noun's query).
### 2. Keys — "what do I have to offer?"
- A second matrix `W_K` maps every embedding to a **key vector** `k`, same smaller dimensionality as queries.
- Keys are meant to potentially **answer** queries — e.g., <mark style="background: #FF5582A6;">"fluffy" and "blue"'s keys should end up closely aligned with "creature"'s query</mark>.
### 3. Dot products — measuring key/query match
- Compute the **dot product** between every possible key-query pair → a full grid of scores.
- <mark style="background: #BBFABBA6;">Large positive dot product = strong match ("fluffy" and "blue" attend strongly to "creature").
- Small/negative dot product = weak/no relevance (e.g., "the" vs. "creature").</mark>
- In ML jargon: "fluffy and blue **attend to** creature."
### 4. Softmax — normalizing into weights
- Raw dot-product scores range over all real numbers — not usable as weights directly.
- ==Apply **softmax down each column** of the grid → each column becomes a set of non-negative weights summing to 1==.
- This full grid, after softmax, is called the **attention pattern**.
- **Compact formula (from the original paper):**
$$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  - `Q`, `K` = full arrays of query/key vectors.
  - `QK^T` = the whole grid of dot products, compactly written.
  - Division by `√(dimension of key/query space)` = numerical-stability technical detail.
  - softmax applied column-by-column, as described.
  - `V` = the value vectors.
### Masking
- **Why needed:** during training, the model is made to simultaneously predict the next token after *every* subsequence (not just the very end) — much more training-efficient than one prediction per example.
- ==**The problem this creates:** later tokens must NOT be allowed to influence earlier tokens' attention (that would leak the "answer" for what comes next).==
- ==**The fix:** before softmax, set all "later token → earlier token" entries in the grid to **negative infinity**. After softmax, `e^(-∞) = 0`, so those entries become exactly zero== — while the columns still stay properly normalized (summing to 1), unlike if you'd just set them to zero directly (which would break the normalization).
- Masking is always applied in GPT-style models, though it's most relevant during training (less so when just running as a chatbot).
### Context size and attention's scaling cost
- Attention pattern's size = **context_size²** — this quadratic scaling is why context size is such a major bottleneck/cost driver for large language models. Recent architecture variations aim to make this more scalable (not covered in this tutorial).
## Values — actually updating the embeddings
- <mark style="background: #FF5582A6;">Computing the attention pattern only tells you which words are relevant to which — doesn't yet move any information</mark>.
- **Value matrix** `W_V`: multiply by an embedding (e.g., "fluffy") to get a **value vector** — "if this word is relevant to updating something else, here's exactly what should be added."
- For each column (e.g., "creature"'s column), take every value vector, scale each by its corresponding attention-pattern weight, and **sum them all together** → produces `Δe` (a change vector).
- <mark style="background: #FF5582A6;">Add `Δe` to the original embedding → refined embedding, now encoding richer contextual meaning (e.g., "a fluffy blue creature")</mark>.
- Applied to **every column** (every token position) in parallel → produces the full sequence of refined embeddings that the attention block outputs.
## Counting parameters — one attention head
- Three matrices so far: Key, Query, Value — each conceptually full-size, but in practice **factored**:
  - `W_Q`, `W_K`: 12,288 columns (embedding dim) × 128 rows (key/query space) ≈ **1.5 million parameters each**.
  - Naive `W_V` (full embedding-to-embedding map) would be ~150 million parameters — but in practice, factored into two smaller matrices for efficiency:
    - **"Value down"** matrix: maps embedding space → smaller space (same size as key/query space).
    - **"Value up"** matrix: maps smaller space → back up to embedding space.
    - This is a **low-rank transformation** — same total parameter count as the key/query matrices, much cheaper than a full square matrix.
- **Total per head:** ~6.3 million parameters (4 same-size matrices: Q, K, value-down, value-up).

## Cross-attention (brief note)
- Nearly identical mechanism to self-attention, but **keys and queries come from two different data sources** (e.g., two languages in translation, or speech audio + its transcription) rather than the same sequence.
- **No masking** typically applied — there's no "later token" concept when relating two separate sequences.
- <mark style="background: #FF5582A6;">VERY VERY IMPORTANT: This is exactly the mechanism HyperDAS uses — hyper-network's own tokens (concept description) provide the query; target model's hidden states (`h̄`, `ĥ`) provide the keys/values.</mark>

## Multiple heads
- ==One head captures one "type" of contextual update (e.g., adjectives→nouns). Many other types exist simultaneously (e.g., "they crashed the" affecting "car"'s implied shape; "wizard" + "Harry" → Harry Potter vs. "Queen, Sussex, William" + "Harry" → Prince Harry).==
- **Multi-headed attention** = run many heads in parallel, each with its own separate Q/K/V matrices. GPT-3: **96 heads per block**.
- ==Each head proposes its own change to add to each token's embedding; all heads' proposed changes get **summed together** and added to the original embedding — producing one slice of the multi-headed attention block's output==.
- **Parameter tally:** 96 heads × ~6.3M each ≈ **~600 million parameters per attention block**.

## The output matrix (a naming nuance)
- In practice, all heads' "value up" matrices are **stapled together into one big "output matrix"** for the whole block, rather than kept as 96 separate small matrices.
- When papers refer to "the value matrix" for a head, they usually mean only the "value down" step — the "value up" step lives inside this shared output matrix instead. (Framing/implementation detail, not a conceptual change.)

## Going deeper — full network scale
- A transformer alternates: attention block → MLP block → attention block → MLP block → ... many times.
- <mark style="background: #FF5582A6;">Each repetition gives embeddings more chances to absorb richer, more nuanced context — hopefully building up to abstract concepts (sentiment, tone, genre, underlying scientific truths), not just grammar/descriptors</mark>.
- **GPT-3 full tally:** 96 layers × ~600M per attention block ≈ **just under 58 billion parameters devoted to attention** — a large number, but only **about a third** of the full 175 billion. Most parameters actually live in the MLP blocks (next chapter's topic), not attention.
## Why attention specifically succeeded
- Not primarily about any one specific behavior it enables — ==its major practical advantage is being **extremely parallelizable** (efficient on GPUs), which matters enormously given that scale alone has driven huge model-quality improvements in recent years.==

## Connection to DAS / HyperDAS
- **This is the full, rigorous version of the toy "cat sat mat" Q/K/V example worked through earlier** — same dot-product → softmax → weighted-value-sum pipeline, just now with real terminology (query/key/value matrices, attention pattern, masking) and the multi-head/multi-layer scaling context.
- **HyperDAS's Formula 3** (`e'_p = MHA(Q(e_p), K(e_p), V(e_p))`) = exactly this self-attention mechanism, applied to the hyper-network's own concept-description tokens.
- **HyperDAS's Formulas 4 & 5** (cross-attention to `h̄` and `ĥ`) = exactly the cross-attention variant described here — queries from the hyper-network, keys/values from the target model's hidden states — confirming the earlier explanation of "peeking at the target model's hidden states" was mechanically accurate.
- **Low-rank value-matrix factorization** (value-down/value-up) is conceptually similar in spirit to HyperDAS's own low-rank orthogonal matrix `R` for subspace selection — both are ways of avoiding a full-size, expensive transformation by projecting through a smaller intermediate space.
---
# 3. MLP Blocks & Where Facts Live in LLMs
The motivating question:
- Given "Michael Jordan plays the sport of ___" → model predicts "basketball" — where inside the network is that fact stored? Google DeepMind researchers found evidence facts seem to live specifically in the **MLP blocks**, not attention — full mechanistic understanding still unsolved.

Quick refresher
- Transformer = tokens → embeddings → repeated (attention block → MLP block → normalization) → final vector used to predict next token.
- Directions in the high-dim embedding space encode semantic meaning (recall: gender direction example, `woman − man` added to `uncle` ≈ corresponding feminine noun).
- Attention handles incorporating **context**; MLP blocks hold the **majority of parameters** — hypothesis: ==they provide extra capacity to store **facts**.==

Example assumptions (simplifying, not literally true)
- Assume 3 nearly-perpendicular directions exist in the embedding space: "first name Michael," "last name Jordan," "basketball."
- A vector's **dot product** with a direction = 1 if it encodes that concept, ≤0 otherwise (simplified — real dot products aren't binary).
- A vector encoding the full name "Michael Jordan" → dot product ≈1 with **both** the Michael and Jordan directions.
- Since "Michael Jordan" spans 2 tokens, an earlier **attention** block must have already merged both names' info into the second token's vector.

Inside an MLP block — step by step
- Each vector in the sequence is processed **independently and in parallel** (no cross-talk between vectors, unlike attention) — <mark style="background: #FF5582A6;">so understanding what happens to one vector tells you what happens to all of them</mark>.

Step 1 — "Up projection": `W_up · E + B_up`
- Multiply the input embedding `E` by a big matrix `W_up` (rows = "questions" being asked about the vector).
- **Row-by-row view:** <mark style="background: #ABF7F7A6;">each row is itself a direction vector; the dot product of that row with `E` measures how well `E` aligns with that direction</mark>.
- **Worked example:** if row 1 = (Michael direction + Jordan direction), then `row1 · E = M·E + J·E` — this sum equals **2** only if `E` encodes *both* names, and less otherwise.
- **Bias term `B_up`:** added after the matrix multiply. In the example, bias = −1 for that row, so the result is positive **only** when both names are present (2 − 1 = 1 > 0), and ≤0 otherwise — engineered to act as a clean detector.
- **GPT-3 numbers:** ~50,000 rows (exactly 4× the embedding dimension — a deliberate, hardware-friendly design choice, not a mathematical necessity).

Step 2 — ReLU (the non-linearity)
- Problem: the linear step alone is too "leaky" — a "Michael + Jordan" detector would also partially fire for "Michael + Phelps" or "Alexis + Jordan," since dot products distribute additively.
- **ReLU**: clips all negative values to 0, leaves positive values unchanged. Applied to the up-projection's output, this turns the earlier "positive iff both names present" value into a clean **AND-gate** behavior: exactly 1 for "Michael Jordan," 0 otherwise.
- ==These post-ReLU values are what's meant by the **"neurons"** of a transformer== (the dots seen in a classic neural-net diagram). A neuron is **active** when positive, **inactive** when zero.
- Real models often use **GELU** (smoother version of the same shape) instead of plain ReLU, but ReLU is used in the tutorial for clarity.

Step 3 — "Down projection": `W_down · (ReLU output) + B_down`
- Maps back down from the large intermediate space to the embedding dimension.
- **Column-by-column view** (more natural here, since columns match embedding dimension): ==each column of `W_down` = a direction that gets **added to the result if the corresponding neuron is active**, scaled by that neuron's value; inactive neurons (value 0) contribute nothing.==
- **Worked example:** column 1 = the "basketball" direction. When the Michael-Jordan-detecting neuron (from step 2) is active, this column gets added — injecting "basketball" into the output.
- <mark style="background: #FF5582A6;">A column can encode multiple associated features simultaneously, not just one.</mark>
- Bias `B_down` is added regardless of neuron state — purpose unclear/hard to interpret ("maybe some bookkeeping").

Final step — residual addition
- ==The down-projection's output is **added to** the original input vector (not replacing it) → final output of the MLP block for that position==.
- Applied to **every token position in parallel** — so GPT-3's MLP block effectively has 50,000 neurons × (number of tokens in the input), not just 50,000 total.

**Overall structure:** two matrix multiplications (each with a bias) + one simple non-linearity in between — the *same basic architecture* as the simplest neural networks (e.g., handwritten-digit recognition), just embedded as one repeated piece of a much larger system.

Superposition — the big caveat on the toy example
- **Reality check:** individual neurons very rarely represent one single clean feature like "Michael Jordan" — even though the math (rows = directions, columns = what gets added) is technically accurate, actual learned features are messier.
- **Superposition hypothesis:** if you require features to occupy *exactly* perpendicular directions in an `n`-dimensional space, you can fit at most `n` independent features (the mathematical definition of dimensionality).
- **But if you relax to "nearly perpendicular"** (e.g., 89°–91° apart, tolerating a little noise), the situation changes dramatically in **high** dimensions specifically — by the **Johnson-Lindenstrauss lemma**, ==the number of near-perpendicular directions you can pack in grows **exponentially** with the number of dimensions== (seen in the video with a 100-dimensional, 10,000-vector optimization example — random vectors already cluster near 90°, and explicit optimization tightens this to a narrow 89°–91° range for all pairs).
- ==**Implication for LLMs:** a model may represent **far more independent features/concepts than it has literal dimensions or neurons**, by using near-perpendicular (not exactly perpendicular) directions — this may partly explain why bigger models scale so well (10× the dimensions → potentially far more than 10× the storable concepts).==
- **Consequence for interpretability:** if this is happening, individual features won't show up as one neuron lighting up — t==hey appear as a **specific combination of neurons** (a superposition) instead==. Relevant tool for extracting true features despite this: **sparse auto-encoders**.

Connection again to DAS/HyperDAS:
- Superposition is directly relevant to my thesis's core research problem
- My refined research question:
	- Validating causal abstractions independently via the commutativity criterion, rather than DAS's self-validating IIA.
- If real features live in superposition (combinations of neurons, not single clean directions), that's exactly the kind of complication that makes finding a *faithful* subspace alignment (via DAS, Boundless DAS, or HyperDAS) <mark style="background: #FF5582A6;">hard</mark>.
- The "rotate into a new basis" approach of DAS (orthogonal matrix `R`) is saying that ==superposed features can still be *disentangled* via a change of basis.
- **Sparse auto-encoders** were also mentioned in the HyperDAS paper itself (as a comparison baseline, Figure 11/A.5) — this video gives the conceptual "why SAEs exist" backstory for that comparison.


