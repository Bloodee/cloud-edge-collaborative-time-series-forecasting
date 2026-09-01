# Curated experiment evidence

These small CSV files transcribe values from the retained experiment reports and
paper draft. They are committed for reviewability; they were not regenerated
during the public repository cleanup.

- `benchmark.csv`: CE-BiD and the best baseline by MSE on each dataset.
- `ablation.csv`: full method, no online update, and no OFA-KD.
- `pvod_online_summary.csv`: retained event-trigger counts and aggregate metrics.

Raw predictions, checkpoints, and multi-run logs are intentionally excluded.
The hardened public code corrects scaling, checkpoint-export, trigger, and
distillation-objective issues documented in `docs/REPRODUCIBILITY.md`. These
historical values are evidence of the completed experiment, not a bit-for-bit
acceptance target for a new run.
