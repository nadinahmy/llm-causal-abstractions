# Causal Abstractions & LLMs — Master's Thesis Research Notes

Research notebook for my Master's thesis at the **AIML Excellence Team, TU Darmstadt** (Prof. Kersting's group), supervised by **Matej Zecevic**.

This repository holds my day-to-day research notes, paper readings, derivations, coding experiments, and visual sketches.
---

## What this repository is for

This is a **working research log**, not a polished paper. Its purpose is to:

- **Maintain the literature** on causal abstraction as I read it, in my own words, with the confusing parts worked out explicitly in a way that I fully understand.
- **Build up the theoretical foundation** my thesis will be based on — structural causal models, abstraction relations, and their approximate and probabilistic generalisations.
- **Track the coding side** of the work alongside the theory.
- **Leave a chronological trail** so that any day's work can be justified, revisited, and reused, and so that discussions with my supervisor always have a concrete visual and written starting point.

The notes are purposely unpolished sometimes, writing things down as I understand them is part of the learning process for me and part of training for the research writing that the thesis itself will require.

---

## Research direction

The playground for my thesis is the intersection of **causal abstraction** and **large language models**. Two framings are on the table, and the notes so far are mapping out the ground needed for both:

1. **Can LLMs learn causal abstractions?** — treating abstraction as something a model could acquire.
2. **Can what an LLM has already learned be understood *as* a causal abstraction?** — treating a trained network's internal computation as a low-level model that a simpler, human-intelligible high-level model abstracts.

---

## Repository structure & conventions

My notes follow a day-to-day system, where each working period gets its own folder named by date, containing a main .md note plus any attachments I reviewed that day:

```
YYYYMMDD/
├── YYYYMMDD.md                     ← main note for the day
└── YYYYMMDD/ (or note-title/)      ← images & attachments for that note
    ├── IMG-....png
    └── ...
PDFs (derivations, handwritten notes, annotated definitions) sit alongside the note.
```

Also a small naming convention note: some folders span a range (e.g. `20260630-20260701`) when a single piece of work took me more than one working day, and some folders I've given a descriptive title to represent what the note is mainly about (e.g. `Causal Inference Coding Tutorial`, `Approximate Causal Abstraction`).

**Tooling & conventions:**

- I'm writing and maintaining this notebook using **Obsidian** (Markdown + LaTeX), with wiki-style links (`[[...]]`) connecting related definitions my notes.
- Any attachments I add are organised per-note via the **Attachment Management** and **Consistent Attachments and Links** community plugins which I've found very helpful.
- Callout blocks (`> [!DEFINITION]`, `> [!IMPORTANT]`, `> [!TIP]`, …) flag definitions, thereoms, solutions and conclusions, and Ive found this format to be very useful to me especially when recalling earlier notes from past working days.

---

## Notes index

I will update this index as I go, and I think it'll be of significant help when I need to recall a particular concept or topic that I covered in the past.

| Date | Folder | Focus |
|------|--------|-------|
| 2026-06-21/22 | `20260621` | **Causal Representation Learning Workshop (UAI 2022).** Talks by Sander Beckers and Fabio Zennaro. Introduces Causal Abstraction Learning (CAL), the Reduced-Form Auto-Encoder (RFAE) and its causal middle layer, encoder/decoder reconstruction loss, disentanglement, the abstraction map τ and intervention map ω, and Beckers & Halpern's four increasingly strict "rungs" of abstraction. |
| 2026-06-23/24 | `20260623` | **Close reading of Beckers & Halpern, *Abstracting Causal Models*.** Structural equations, contexts, causal dependency and the natural partial order, recursion/strong recursivity, interventions and the `[Y←y]φ` semantics, the unique-exogenous-variables (uev) assumption, why "exact transformation" is too weak, uniform transformations, and **Definition 3.12** (the τ-abstraction with its restriction sets and induced ω_τ). |
| 2026-06-26 | `20260626` | **Continuation of the Beckers paper.** Worked through Example 3.16 (the pixel-grid upper/left-count abstraction) to see exactly why it is a τ-abstraction but *not* a **strong** one, then the paper's conclusions: the motivation for **approximate abstraction** and the direct relevance to explainable AI. |
| 2026-06-27/28 | `20260627-28` | **Causal Inference Coding Tutorial.** Hands-on: identifying causal estimands, standardization / parent adjustment, the Average Treatment Effect, the back-door criterion, forks/chains/colliders and d-separation, Inverse Probability Weighting and propensity scores, and finally **neural estimators** — a CNN recovering a hidden confounder (species) from CIFAR-10 images and plugging into IPW and outcome-model adjustment. |
| 2026-06-30/07-01 | `20260630-20260701` | **Approximate Causal Abstraction (Beckers).** The `d_max` worst-case model distance, **Definitions 3.1–3.2** (α-approximation and approximate τ-abstraction via the commuting-pathway test), the likelihood-of-serious-error distance (Def. 4.2), probabilistic distance with intervention distributions (Def. 4.8), and the robustness-vs-sensitivity of τ captured by the magnification factor *k*. |
| 2026-07-02 | `20260702` | **Geiger, Ibeling et al., *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability* (JMLR).** Behavioral vs mechanistic interpretability, faithfulness and "just-so stories", polysemantic neurons and distributed representations, **interventionals**, hard vs soft interventions, solution sets, and the **intervention algebra** (Def. 15) that lets a feature spread across many neurons be treated as one clean high-level causal variable. |