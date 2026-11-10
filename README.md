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
notes/
└── YYYYMMDD/
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

The full chronological index of what each day's note covers lives in its own file: **[INDEX.md](INDEX.md)**.
