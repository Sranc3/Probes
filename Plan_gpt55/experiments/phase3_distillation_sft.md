# Phase 3: Distillation / SFT Feasibility

## Purpose

Only enter training if earlier phases show that uncertainty signals can select safer or more decisive behavior.

## Entry Conditions

At least one must hold:

- reranking improves correctness or stability on pure test;
- offline controller predicts safe actions above random baseline;
- adaptive gate produces stable non-drop correctness and hesitation reduction.

## What Distillation Should Learn

The student should not learn entropy distributions directly.

It should learn:

- better final answers;
- more stable answer style;
- lower unnecessary hesitation;
- teacher preference over candidate answers;
- behavior selected by reranking/controller.

Entropy is used as:

- sample weight;
- curriculum signal;
- uncertainty tag;
- rejection/reranking feature;
- confidence-aware loss modifier.

## Data Construction

Positive examples:

- correctness preserved or improved;
- semantic breadth within band;
- lower token hesitation;
- reasonable answer length.

Negative examples:

- correctness drop;
- semantic collapse;
- excessive hesitation;
- unstable or verbose output.

## Candidate Training Objectives

1. teacher-generated SFT;
2. pairwise preference distillation;
3. confidence-weighted SFT;
4. answer-level reranking distillation;
5. optional entropy-aware regularization.

## Initial Scale

Before any large run:

- collect 500-2000 teacher/reranked examples;
- train or tune only lightweight adapter if possible;
- evaluate on pure held-out test and separate external QA if available.

## Failure Criteria

Stop if:

- correctness drops;
- output diversity collapses;
- gains only appear on train-mixed evaluation;
- improvements disappear with new seeds.
