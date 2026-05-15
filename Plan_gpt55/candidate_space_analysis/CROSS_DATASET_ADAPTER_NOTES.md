# Cross-Dataset Adapter Notes

This study should not mix TriviaQA, HotpotQA, and LongBench examples in one tuning pool at this stage.

## Supported Normalized Format

All datasets should be converted to a lightweight `batch_summary.json` with:

- `id`: stable example id.
- `question`: exact prompt question sent to the model.
- `raw_question`: original question before context injection, when available.
- `ideal_answers`: accepted short answers for strict exact/contains evaluation.
- `dataset`: source dataset name.
- `strict_evaluator`: currently `normalized_exact_or_contains`.
- `prompt_includes_context`: whether the adapter injected evidence/context.

The adapter script is:

```bash
python /zhutingqi/song/Plan_gpt55/candidate_space_analysis/scripts/prepare_qa_dataset_adapter.py \
  --dataset hotpotqa \
  --input /path/to/hotpotqa.jsonl \
  --output-dir /path/to/output_batch \
  --num-questions 500 \
  --include-context
```

## Recommended Order

1. HotpotQA first: closest to short-answer factual QA, but context/evidence handling must be fixed before generation.
2. LongBench second: long-context prompts may change token-cost scaling and answer style, so do not compare it directly with TriviaQA until prompt and evaluator are stabilized.

## No-Leak Rules

- Tune controller thresholds on TriviaQA or a train split only.
- Evaluate HotpotQA/LongBench transfer separately.
- Do not expose gold answers, strict correctness, rescue labels, damage labels, or future full-8 basin labels to generation-time features.
- Keep verifier calls, if any, separately costed from generation tokens and latency.

## Current Scale500 Decision Context

The 500-question x 2-seed TriviaQA run reached the planned 1000 seed-question pair scale. The best generation-time adaptive row improved strict correctness by `+1.70pp`, below the `+2pp` adaptive-inference continuation threshold, although it saved substantial cost versus full-8. This makes HotpotQA transfer the next useful check before committing to GRPO-style post-training.
