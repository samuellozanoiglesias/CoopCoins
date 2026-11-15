# Plan: env/optimiser changes + theft-penalty (k) sweep (written before coding)

## Files to change
| file | change |
|---|---|
| coopcoins/env.py | A1 random tie-break (Gumbel argmax, separate tie / respawn keys), wins-only counts, `tie_arrivals`, `tie_wins`; A2 memory EMA in `State`, 4 features appended (11 total); A5 whitelisted `make_env` signature, payoff validated and required |
| coopcoins/ppo.py | A3 per-subtree clipping via `optax.multi_transform`, pre-clip norms logged; A4 running return stats in `Runner` (per team, per run), de-normalise before GAE, normalised critic targets |
| coopcoins/utils.py | `make_map` carries `memory`; ratio family branch removed; payoff/k helpers |
| coopcoins/simulation.py | extra per-game stats: defence / contest opportunities and choices, mean memory features |
| coopcoins/influence.py | 11-feature synthetic probes with a memory vector; memory override for matched states; probe state-set id (pseudo-replication) |
| train.py | payoff/k/memory in config.yaml, checkpoint metadata and training_stats.csv; return stats in checkpoints; skip completed runs; `--specialists`, `--max_parallel_runs`, `--num_episodes`, `--data_root` overrides |
| simulate.py | memory in map defaults for unseen maps; payoff columns |
| crossplay.py | payoff columns, new stats |
| probe.py | memory sweep, memory swap, payoff columns, dedup flag |
| analyze_influence.py | `--source` (any experiment key), per-training-id output dir; defence/contest metrics; tie stats; best-response matrix; accounting check; memory analyses; p floor 1/B; BH q-values; non-independence note |
| analyze.py | payoff/k in outputs |
| analyze_k.py (new) | cross-k analysis |
| configs/experiments/scaling.yaml, scarcity.yaml | ratio family removed (scaling); `max_grad_norm` -> `clip`; `memory`, `return_norm` |
| configs/experiments/payoff_k{0,2,10}.yaml (new), configs/influence_k.yaml (new) | the three arms |
| launch.sh, README.md | ratio maps removed; docs |
| tests/ (new) | A7 tests |
| experiments/k_*.sh, experiments/benchmark.py, experiments/make_smoke_configs.py (new) | launch experiments |
| docs/experiment_k.md (new) | README section |
| requirements.txt | + pytest |
| coopcoins/networks.py | no change needed (input size follows N_FEATURES) |

## Conflicts / ambiguities found (and how I resolve them)
1. README and launch.sh call experiments that are not in the repo (run_scaling.sh, smoke_test.sh, speed_test.sh, run_exp1-4.sh). launch.sh cannot run as shipped. Convention followed: plain bash + nohup, logs in <data_root>/logs (no SLURM anywhere in the repo). New experiments call the python entry points directly.
2. analyze_influence.py hard-codes `experiments.exp1` and writes to influence/analysis/exp<N> regardless of training id, so three arms would overwrite each other. Added `--source <key>`; non-default sources write to influence/analysis/<training_id>/exp<N>. Default (exp1) path unchanged.
3. Map dicts carry metadata keys (density, ratio, coins_per_agent, families) that make_env swallowed with `**_`. These are whitelisted explicitly. I also make `payoff` and `memory` required arguments (a missing payoff would otherwise fall back silently, which is the hazard A5 targets).
4. Removing `max_grad_norm` breaks scaling.yaml / scarcity.yaml; both are migrated to `clip:` with the same value 0.5.
5. configs/influence.yaml still points at scaling_v1 / scarcity_v1. Those checkpoints are incompatible with the new code (11 features, new optimiser state). Left unchanged and unused, as instructed.
6. No hard-coded theft penalty exists in the analysis code (analyze.py `/ 2` averages the two teams). The only -2 lived in the env default and the docs.
7. Interpretation risk: memory lets a fixed policy condition on the partner's behaviour in-game, so that adaptation is counted as ecological E (same policy, other partner), not strategic S. The memory swap measures this channel separately.
8. All arms use PRNGKey(seed): seed s has the same init in every arm. Cross-k bootstraps resample the seed index jointly across arms and maps.
9. Return normalisation: count starts at 0, mean 0, var 1; std = sqrt(var + eps).
10. Memory means for the sweep/swap: focal team, cell (trained with B, plays B, cross pairing) of the same map, from crossplay_games.csv.
11. Defence/contest use observation features: "targeted last step" = opponents_targeting > 0; opponent_dist = nearest opponent. Computed on every decision of every cross-play game.
12. Benchmark here runs on a 1-CPU sandbox; the real estimate must come from k_smoke.sh on the GPU node.
