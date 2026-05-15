# Cross-Model Anchor Pilot Recording Checklist

## Goal

Use `gpt-oss-120b` as a strong heterologous teacher to probe whether Qwen candidate basins are factually anchored rather than merely stable. The run should be self-contained because loading and generation are expensive.

## Run Metadata

- Run id, timestamp, script path, git/workspace note.
- Teacher model path, model class, tokenizer class, model dtype, device map.
- Python executable, torch/transformers/kernels/safetensors versions.
- GPU list, visible devices, max memory seen if available.
- Dataset path, candidate feature path, selected split/offset/seed/question count.
- Generation parameters: samples per question, greedy/sample mix, max new tokens, temperature, top_p, do_sample, reasoning level, prompt template.
- No-leak statement: gold answers are not included in prompts and are used only after generation for labeling/analysis.

## Per Generation Row

- `question_id`, `question_index`, `question`, `ideal_answers` for offline labeling.
- Teacher sample index and decode mode: greedy or sampled.
- Full prompt/messages used for teacher generation.
- Raw generated text with harmony special tokens preserved.
- Parsed `analysis` text if present, parsed `final` answer, fallback parsed answer.
- Normalized answer and canonical basin key.
- Prompt token count, completion token count, total token count.
- Latency per generation and optional CUDA peak memory.
- Strict correctness, exact/contains flag, best token F1 against gold.

## Per Teacher Basin

- Teacher basin id, canonical key, representative final answer.
- Basin size, basin mass, member sample indices.
- Basin correctness: any strict correct, max F1, mean F1.
- First-seen sample index and greedy membership flag.

## Qwen Baseline/Basin Features To Join

- Qwen sample0 answer, sample0 correctness, sample0 cluster id.
- Existing Qwen candidate rows for the same question: sample index, answer text, cluster id, cluster size, cluster mass.
- Qwen entropy/logprob features: logprob avg, token mean/max entropy, token count.
- Question-level semantic entropy and number of semantic clusters.
- Qwen oracle availability and any-correct indicator.

## Cross-Model Alignment Features

- Best teacher-to-Qwen answer similarity by normalized exact/contains/token F1.
- Best teacher-to-Qwen cluster id, cluster size, cluster mass, and correctness.
- Teacher support for each Qwen basin: max similarity, count of teacher samples aligned, teacher mass aligned.
- Qwen-only stable basin indicator: high Qwen mass but low teacher support.
- Teacher-supported Qwen basin indicator: high teacher support and nontrivial Qwen mass.
- Conflict indicator: Qwen sample0 basin differs from teacher majority basin and teacher majority is correct.

## Anchor Scores

- Candidate-level `anchor_support`: teacher mass aligned to candidate/Qwen basin.
- `anchor_correct_support`: teacher aligned mass from teacher-correct basins.
- `qwen_stability`: Qwen cluster mass/size.
- `qwen_internal_confidence`: logprob/entropy score from existing verifier terms.
- `hallucination_risk`: high Qwen stability but low teacher support.
- `anchor_score`: combined feature for AUC and future VBPO pair construction.

## Output Files

- `run_metadata.json`
- `teacher_generations.jsonl`
- `teacher_basin_rows.csv`
- `qwen_candidate_anchor_rows.csv`
- `question_anchor_summary.csv`
- `anchor_metric_summary.json`
- `ANCHOR_PILOT_REPORT.md`

## First Pilot Size

Use 100 TriviaQA questions from the existing 500-question Qwen candidate feature run, with 4 teacher generations per question: one greedy plus three sampled generations. This gives enough signal for teacher support without making the first run too costly.
