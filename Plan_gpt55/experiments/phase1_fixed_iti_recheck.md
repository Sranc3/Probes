# Phase 1A: Pure Held-out Fixed ITI Recheck

## Purpose

Re-evaluate fixed ITI candidates on pure held-out test questions only.

This experiment corrects the interpretation risk discovered in `direction_asset_audit.md`: the previous deployment-style `all_rows[0:200]` selection included train/val/test.

## Hypothesis

If fixed ITI is deployable, it should produce measurable answer behavior changes on pure test data without reducing correctness.

## Data

- Dataset source: `/zhutingqi/song/layer_level/output/run_20260403_050400`
- Split manifest: `/zhutingqi/song/ITI_for_entropy/Plan_A/artifacts/feat_plus_mlpdown_entropy_weighted/split_manifest.json`
- Split: `test`
- Expected questions: `31`
- Seeds: `[42, 43, 52, 53, 54]`

## Candidates

| ID | Mode | Alpha | Sites | Notes |
| --- | --- | ---: | --- | --- |
| `baseline` | none | `0` | none | No intervention |
| `cand008_primary` | `increase_semantic_high` | `0.005` | `18,24` | Existing candidate |
| `ind0067_safety_pareto` | `reduce_semantic_high` | `0.0070296743` | `18,24` | Fallback Pareto representative |
| `layer18_only` | both signs | grid | `18` | Site ablation |
| `layer24_only` | both signs | grid | `24` | Site ablation |

## Required Metrics

- answer change rate
- correctness transition table
- exact / contains / NLI semantic correct
- semantic entropy weighted/uniform
- token mean entropy
- token max entropy
- event count and event density
- seed-level stability

## Pass Criteria

- correctness non-drop rate = `1.0`
- semantic band success rate >= `0.80`
- answer change rate >= `0.10`
- token mean nonpositive rate >= `0.67`
- token max nonpositive rate >= `0.50`

## Decision

- Pass: fixed ITI remains viable as a candidate control primitive.
- Partial pass: use fixed ITI as action space for learned controller.
- Fail: stop treating fixed ITI as deployment route.
