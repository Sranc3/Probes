# Verifier-Guided Basin Preference Optimization

This experiment replaces the fragile pure Basin-GRPO reward with pairwise preference optimization.

The training pairs are built only from the training split:

- `damage_guard`: if base sample0 is correct, prefer sample0 over stable wrong alternatives.
- `rescue_from_sample0`: if sample0 is wrong but a correct candidate exists, prefer the correct candidate over sample0.
- `correct_vs_stable_wrong`: prefer the best correct candidate over high-mass/high-score wrong basins.

The model is trained with a DPO-style objective:

```text
L = -log sigmoid(beta * [(log pi(chosen)-log pi(rejected)) - (log pi_ref(chosen)-log pi_ref(rejected))])
```

Evaluation must use no-leak selectors only. Gold labels are allowed for training preference construction, but never for fixed-k answer selection at evaluation time.
