# Where an LLM was used in this project

Covers Projects 1 and 2. Updated as later projects are done.

The implementation was drafted with an LLM (Claude, via Claude Code) over a
single working session, then read through and corrected. This file records
which parts that applies to and where the line ran, since "AI-assisted" on its
own says nothing useful.

## By component

| Part | How it was produced |
|---|---|
| Django wiring — `startapp`, `INSTALLED_APPS`, URL includes, forms, session handling | Drafted by the model. This is the same twelve steps every Django app needs and there is one correct way to do it. |
| Templates and CSS | Drafted by the model against a brief: keep the skeleton's blue-and-white identity, override the absolutely-positioned `.box` from the shared stylesheet because these pages are taller than the viewport. |
| Matplotlib figures | Drafted by the model. Choosing the four figures was a decision; getting the spines, palette and legend placement right afterwards was typing. |
| CSV loading, dropping bookkeeping columns | Rules decided first, then written up. See below. |
| Classification vs regression detection | Rule decided first, then written up. See below. |
| Fold construction, the sweep, the selection rule | Specified as an algorithm, then written up. See below. |
| Choice of models, hyperparameters and scores | Decided against the lecture material, not delegated. |
| Verification | Model-run: a script exercising every view and error path with Django's test client, plus a browser pass. |

## The parts that were not delegated

Three components had more than one defensible answer, and those are the ones the
brief is actually asking about. Each was settled as a rule first and the model
was used to write the rule down, not to choose it.

**Dropping an id column.** Matching on the column name is the obvious approach
and it is wrong — `patient_id` can be a real feature. The rule used here is
structural: integer, unique across every row, and consecutive. A column that
satisfies all three is numbering the rows.

**Classification or regression.** The common shortcut is a fixed cutoff, "ten or
fewer distinct values means classes". That misreads a twelve-class problem and
also misreads a small regression table. The cutoff here grows with the square
root of the row count, floored at ten and capped at twenty, and the result is
always shown on screen with an override available at upload.

**Picking the hyperparameter.** `GridSearchCV` is one line and would have done
it. It also picks the argmax of five noisy numbers. What is implemented instead
is a hand-rolled stratified fold split (deal each class's shuffled indices
round-robin into k folds), an explicit sweep recording the mean, the standard
error and the training-set score for every candidate, and a one-standard-error
selection rule that prefers the plainer model among values statistically
indistinguishable from the best. There is no scikit-learn equivalent for that
last part. On the iris data every tree depth from 3 to 14 ties in
cross-validation while the training score climbs to 1.0, which is the case the
rule exists for.

## Whether it was worth it

For the Django and presentation layers, clearly yes. That work is well-specified
and has a known shape; time spent typing it out is time not spent on the part
being graded.

For the three components above, the honest answer is that the model saved typing
and nothing else. The decisions had to be made against the project brief and
lecture 1 either way, and a model asked to "pick a hyperparameter" reaches for
`GridSearchCV`, which is exactly the answer the brief is trying to get past.
Using it as a drafting tool worked; using it as the source of the design would
have produced a worse project.

## Project 2 specifically

The brief for Project 2 names parts that must be our own, and those are tracked
formally rather than in prose: `home/own_work.py` holds each requirement with the
brief's exact wording, the file satisfying it, and what was decided. Each such
block in the code carries a marker comment, and `/home/own-work/` re-reads the
files and reports whether the markers survive. That page fails loudly if a marked
block is deleted or moved.

| Requirement | What was decided, and by whom |
|---|---|
| Ω for logistic regression (Task 3) | Number of original features with a non-zero coefficient under L1. Decided first, then written up. The reasoning — that Ω must measure reader effort in both model classes, and that L2 would leave Ω constant — is the answer the brief is asking for, and it is not something to delegate. |
| Noising categorical data (§2) | Flip-with-probability, redrawn from the empirical distribution. Decided first. |
| Sparse counterfactual proposals | Mine, and not asked for. Perturbing all four measurements every draw made every answer useless; restricting each draw to a random subset turned "change all four things slightly" into "grow the bill by 4.5 mm". |
| The λ frontier | Mine, and not asked for. Recognising that acc − λΩ is a line in λ, so only the upper concave hull is ever selectable, is a small piece of geometry that makes the slider explainable instead of mysterious. Model-drafted code, human-noticed idea. |
| PDP and ALE (Task 5) | Formulas taken from lecture 3 and the brief; the implementation written out. No library is called. The exact-vs-discretised distinction the brief asks about was reasoned through, not looked up. |

The honest summary is the same as for Project 1: the model was fast at Django
plumbing, matplotlib and templates, and would have reached for
`sklearn.inspection.partial_dependence` for Task 5 — which is precisely the
answer the brief exists to prevent.

## What was checked by hand

- The stratified folds were checked to be disjoint, complete, and class-balanced.
- Ridge on the diabetes dataset reproduces lecture 1's reported MSE of roughly
  2856 to within the noise of the split, which is the closest thing to a
  reference result available.
- Project 2's frontier was checked against brute force: sweeping λ over 35,000
  values and taking the arg max reproduces exactly the models the hull predicts.
- Project 2's two ALE routes were checked against each other; the closed-form
  softmax derivative and the finite-difference estimate agree to 1.5e-4.
- The PDP was checked against an independent hand computation.
- Four bugs surfaced during verification of Project 1 and were fixed: a misleading error
  message on very small files, a crash when the default `k` grid exceeded the
  rows available in a fold, a hardcoded claim in the AutoML warning text that
  happened to be true for iris and false in general, and a 500 when a text
  target was forced to be a regression problem.

The last of those is the one worth remembering. The override had been tested,
but only on a numeric target, where it happens to work. A test that exercises a
feature on the one input it was written against is not a test. The override is
now checked across every combination of the three problem-type settings and
three shapes of target column.

Two more were found in Project 2. The `saga` solver is stochastic, and without a
fixed `random_state` the fitted coefficients — and therefore Ω, and therefore the
whole frontier — changed between runs; caught by running the same computation in
three separate processes and comparing. And `text-transform: uppercase` turns a
Greek lambda into a capital lambda, which reads as an "A", so the slider label
said "REGULARISATION A" until it was noticed on screen.
