# Plan
* **Week 1 – Survey Agent Frameworks & Learning**

  * *Objective:* Select the agent stack that best fits JCIG-style workflows and learn its basics.
  * *Key tasks:*

    * Build simple project in **LangChain + LangGraph, AutoGen, and CrewAI**.
    * Note **3 strengths & 3 weaknesses** for each stack.
    * Choose the front-runner and Learn it properly.
    * build simple coding multi agent
  * *Deliverables / checkpoints:*

    * Complete comparison table.
    * Runnable demo script for each framework.
    * Decision note in the README.
    * multi agent code writer agent 

* **Week 2 – Data Collection & Core Utilities**

  * *Objective:* Create a clean pipeline from raw manuals to vector embeddings.
  * *Key tasks:*

    * Download and clean **CNSE, CNSS, IFIXIT** (plus any extra manuals).
    * Implement `document loder(Json, PDF and Other)` that outputs sentences & sections.
    * Implementing emebedding techniques and we can add cacheing tecnique (If the same sentence was embedded earlier, reuse the stored result Don’t call the model again for already-seen sentences).
    * tiny checks to make sure it's doing what we expect (correct shape, consistent output, no crashing).
  * *Deliverables / checkpoints:*

    * datasets folder.
    * Loader and embedding modules.
    * Passing test suite and notes on edge cases.

* **Week 3 – Graph Representations: Baseline JCIG + One Alternative**

  * *Objective:* Re-create the paper’s JCIG and compare it with at least one alternative representation.
  * *Key tasks:*

    * Code vanilla JCIG (concept clustering → graph building → direction deduction(HP and SGS)).
    * Implement one alternative.
    * Run both on 3 dataset; log metrics such as edge F1 and doc-matching accuracy.
    * Analyze where each representation performs best.
  * *Deliverables / checkpoints:*

    * baseline and alternative modules.
    * CSV of results plus a brief analysis note or plot.

* **Week 4 – Wrap-Up & Reporting**

  * *Objective:* polish documentation.
  * *Key tasks:*

    * Write a full **README** (install guide, architecture diagram, run examples).
    * Draft a **4–5-page report** (intro, method, experiments, future work).
    * Create a **10–12-slide deck** and rehearse the demo.
    * Leave buffer time for mentor feedback and final bug fixes.
  * *Deliverables / checkpoints:*

    * PDF report and slide deck.
    * Verification checklist
