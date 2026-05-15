# Basin-GRPO for TriviaQA

This folder contains an isolated heavy-method experiment for basin-aware GRPO-style post-training.

## Objective

For each question `q_i`, sample a group of completions:

```text
y_{i,1}, ..., y_{i,G} ~ pi_theta(. | q_i)
```

Each completion receives a basin-aware reward:

```text
R = 1.0 R_strict
  + 0.3 R_f1
  - lambda_len C_len
  - lambda_hall H_basin
  - lambda_damage D_damage
  + lambda_cons S_consensus
```

Group-relative advantage:

```text
A_hat_{i,g} = (R_{i,g} - mean_g R_{i,g}) / (std_g R_{i,g} + eps)
```

Clipped GRPO/PPO-style loss:

```text
L = - E[min(r A_hat, clip(r, 1-eps, 1+eps) A_hat)]
    + beta_kl KL(pi_theta || pi_ref)
```

where `r = exp(logp_theta - logp_old)`.

## Reward Terms

- `R_strict`: normalized exact/contains match against TriviaQA gold aliases.
- `R_f1`: best token F1 over gold aliases.
- `C_len`: completion length cost.
- `H_basin`: stable wrong basin penalty, proportional to answer-basin mass.
- `D_damage`: penalty when the frozen baseline sample0 is correct but the current completion is wrong.
- `S_consensus`: reward for falling into a high-mass correct basin.

## First Runs

Smoke test:

```bash
/root/miniconda3/envs/jyh_0526/bin/python \
  /zhutingqi/song/Plan_gpt55/basin_grpo/scripts/train_basin_grpo.py \
  --config /zhutingqi/song/Plan_gpt55/basin_grpo/configs/basin_grpo_smoke.json
```

Plot a run:

```bash
/root/miniconda3/envs/jyh_0526/bin/python \
  /zhutingqi/song/Plan_gpt55/basin_grpo/scripts/plot_basin_grpo.py \
  --run-dir /path/to/run_dir
```

Evaluate a LoRA checkpoint:

```bash
/root/miniconda3/envs/jyh_0526/bin/python \
  /zhutingqi/song/Plan_gpt55/basin_grpo/scripts/evaluate_basin_grpo.py \
  --config /zhutingqi/song/Plan_gpt55/basin_grpo/configs/basin_grpo_v0_triviaqa.json \
  --adapter-dir /path/to/checkpoint \
  --split test
```

## Notes

Keep all splits question-heldout. Use train rewards only for updates, dev only for hyperparameter selection, and test only once for final reporting.
