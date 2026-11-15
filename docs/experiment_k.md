# Theft-penalty sweep (k = 0, 2, 10)

**Question.** Does the individualist's partner-dependent behaviour become more *strategic* (cross-play
decomposition into strategic S and ecological E) when theft is more harmful?

**Manipulation.** Only the theft penalty changes: `payoff = {own: 1, other: 1, penalty: -k}`, k ∈ {0, 2, 10}. The
taker gets +1 for either colour in every arm, so the individualist is privately indifferent between coins.
k = 2 is the previous payoff; k = 0 is a placebo where theft harms nobody.

This is a fresh study: nothing trained before these code changes is loaded or compared.

## What changed in the code, and why

| change | where | why |
|---|---|---|
| Random tie-break: if several agents land on a coin in the same step, one of them (uniform, Gumbel argmax with its own PRNG key) collects it; the penalty applies only if the winner is an opponent. own/other coins count wins; missed_* include lost ties; new stats `tie_arrivals`, `tie_wins` | `coopcoins/env.py` | before, every arriving agent collected the coin and each thief triggered a full penalty (double counting), which inflates thefts on dense maps |
| Memory: 4 EMA features per agent (`mem_robbed`, `mem_opp_target_mine`, `mem_stole`, `mem_team_target_theirs`), `memory: {decay: 0.9}`, appended after the 7 existing features on every coin row (11 features), reset to 0 | `coopcoins/env.py` | lets a policy react to what the partner has been doing in this game |
| Separate gradient clipping: actor and critic subtrees each clipped by their own global norm (`clip: {max_grad_norm_actor: 0.5, max_grad_norm_critic: 0.5}`), per team; pre-clip norms logged | `coopcoins/ppo.py` | with joint clipping, large critic gradients shrank the actor update; matters more as k scales the returns |
| Return normalisation: running mean/var of returns per team and run (in `Runner`), critic predicts normalised values, de-normalised before GAE, critic trained on (ret − mean)/std; stats in checkpoints and `training_stats.csv` | `coopcoins/ppo.py`, `train.py` | return scale grows with k (in the smoke test, return std ≈ 1.2 at k = 0 vs 6–9 at k = 10) |
| `make_env` whitelists its keyword arguments; `payoff` (exactly own/other/penalty) and `memory` are required | `coopcoins/env.py` | a misspelled payoff key used to fall back silently to 1/1/−2 |
| Payoff written to config.yaml, checkpoint metadata, training_stats.csv and every simulation / cross-play / probe CSV (`payoff_own`, `payoff_other`, `payoff_penalty`, `k`) | all writers | every result file states its k |
| Ratio family removed (grid6_agents4, grid9_agents6); grid3_agents2 stays in the density family | configs, `utils.py`, `launch.sh`, README | — |

New measurements, all in the cross-play CSV: defence / contest opportunities and choices, and the mean memory per
game. Synthetic probes now take a memory vector; matched probes include a memory swap.

**Interpretation caveat (memory).** With memory, a *fixed* policy can adapt to the partner during the game. In the
cross-play decomposition that adaptation is counted as **ecological** (same policy, other partner), not strategic.
The memory sweep / memory swap analyses measure this in-game channel separately.

## Design

- Maps (density family): grid3_agents2, grid4_agents4, grid5_agents6, grid6_agents8, grid7_agents10,
  grid8_agents14, grid9_agents18, grid10_agents22; coins = agents; 150 steps.
- Generalist only (`all_maps`). Specialists: `training.specialists: true` or `train.py --specialists` (not run).
- Pairs (red vs blue): ind–ind, ind–coop, ind–comp. Seeds 0–9.
- Hyperparameters identical to `scaling.yaml`: 5000 games per map, num_envs 1, lr 1e-3, 4 epochs, 15 minibatches,
  hidden 64, ent_coef 0.01, 5 checkpoints + episode 0.

| config | training_id | k |
|---|---|---|
| configs/experiments/payoff_k0.yaml | payoff_k0_v1 | 0 |
| configs/experiments/payoff_k2.yaml | payoff_k2_v1 | 2 |
| configs/experiments/payoff_k10.yaml | payoff_k10_v1 | 10 |

Pipeline settings (`configs/influence_k.yaml`, same as the influence study): 20 games per cross-play pairing,
partner seed = focal seed + 1, co-trained control; probes from 2 games per seed, 4000 states per (map, world).

## Launch

```bash
# 0. on the GPU node, once: tests + whole pipeline on a tiny copy + benchmark of the full configs
GPU=0 ./scripts/k_smoke.sh                 # must end with "K SMOKE TEST PASSED"

# 1. everything (train -> evaluate -> analyse)
GPU=0 MAX_PARALLEL_RUNS=<RECOMMENDED value from k_smoke.sh> nohup ./scripts/k_all.sh > k_all.out 2>&1 &

# or stage by stage
GPU=0 MAX_PARALLEL_RUNS=<value> PARALLEL=0 ./scripts/k_train.sh  # PARALLEL=1 runs 3 arms at once: needs 3x the memory
GPU=0 ./scripts/k_evaluate.sh
./scripts/k_analyse.sh
```

Variables (`scripts/k_common.sh`): `GPU` (CUDA device), `MAX_PARALLEL_RUNS` (vmapped runs per batch, 0 = all 30),
`PARALLEL`, `SOURCES` (subset of `k0 k2 k10`), `INFLUENCE` (config), `FORCE=1` (redo evaluation).
`MAX_PARALLEL_RUNS` does not change results: every run's random stream depends only on its seed.

## Resume

- `./scripts/k_all.sh --from evaluate` (or `--from analyse`, `--to train`, …) restarts at a stage.
- Training: relaunching `k_train.sh` skips completed runs (final checkpoint present). Incomplete runs are wiped —
  their checkpoints and their rows in training_stats.csv — and retrained, so nothing is logged twice. Arms whose
  three pair directories all have `summary.yaml` are skipped entirely.
- Evaluation: an arm with `influence/<training_id>/.evaluate_done` is skipped unless `FORCE=1`.
- Analysis is cheap and always rerun.

## Outputs

```
<data_root>/all_maps/<red_blue>/Training_payoff_k<k>_v1/
    config.yaml (payoff, k, memory)  training_stats.csv (+ payoff cols, tie stats, actor/critic_grad_norm,
    return_mean, return_std)  training_curves.png  summary.yaml
    checkpoints/seed_<s>/episode_<n>.pkl   {'team_0', 'team_1', payoff, k, memory_decay, return_stats, ...}
    simulations/<map>/episode_5000_stochastic/games.csv ...
<data_root>/influence/payoff_k<k>_v1/
    crossplay_games.csv   probe_coefficients.csv   probe_states_mean.csv   probe_matched.csv   probe_memory_swap.csv
<data_root>/influence/analysis/payoff_k<k>_v1/
    exp1/  seed_metrics, cells, decomposition (S, E, I, T, share; p, q), coadaptation, tie_stats(_partner_team),
           best_response, accounting_check (asserted), figures
    exp2/  probe_shifts (+ pseudo_replicate flag), probe_shifts_pooled_dedup, matched_effects,
           probe_memory_sweep, memory_swap_effects, policy_vs_behaviour, figures
    exp4/  observed_vs_true, peer_regressions, figures
<data_root>/analysis/payoff_k<k>_payoff_k<k>_v1/      scaling behaviour summaries (analyze.py)
<data_root>/influence/analysis/k_sweep/
    k_decomposition.csv  k_trends.csv  k_trend_tests_S.csv  k_teacher.csv  k_training_diagnostics.csv
    k_decomposition_<contrast>.png  k_strategic_share.png  k_S_by_map.png  k_teacher.png
    k_training_diagnostics.png  k_training_curves.png
<data_root>/logs/k_{train,evaluate,analyse}_<arm>_<time>.log
```

Every analysis directory has `STATISTICS_NOTE.txt`:
- bootstrap over training seeds (2000 draws);
- p-values are never 0 (below the resolution they are reported as `<0.0005`);
- Benjamini–Hochberg q-values are computed over all p-values of the same table;
- generalist results on different maps share seeds and are not independent replications;
- the arms also share seed indices (same initialisation per seed), and the cross-k bootstrap resamples seeds
  jointly.

Defence / contest rates are undefined (NaN) for a seed and cell without any opportunity, e.g. when cooperative
opponents never target the focal team's coins. If a whole k level has no opportunities, the trend for that metric is
NaN with an empty p-value; this is reported, not imputed.

Pre-specified test: `k_trend_tests_S.csv`, the slope of S on k (plus k − 0 differences), for
individualistic→competitive and individualistic→cooperative, per metric, per map and pooled with map fixed effects.

## Compute

The new code adds little per-step work (4 input features, one Gumbel draw per step, a few statistics). The cost of
one arm should therefore be close to the `all_maps` training of `scaling_v1`, which had 10 maps rather than 8; the
two dropped maps were small. The whole study is about 3 such trainings, plus evaluation.

Measured in the development sandbox (1 CPU core, no GPU): **3.47 s per PPO update for a single run** with all 8
maps. That is 5000 updates × 30 runs × 3 arms ≈ 430 core-hours, so CPU is not an option. (An earlier figure of
43.6 s was wrong: the timed call was recompiling, a bug now fixed — see below.)

**GPU memory.** The 30 runs of an arm do not necessarily fit in one vmapped batch. On the first GPU node tested,
`MAX_PARALLEL_RUNS=0` (all 30) ran out of memory. `scripts/benchmark.py --auto` (what `k_smoke.sh` runs) tries batch
sizes 30, 15, 10, 6, 5, 3, 2, 1, each in a fresh process, and prints `RECOMMENDED MAX_PARALLEL_RUNS=…` and the
`ESTIMATE` for that size. Use that value for `k_train.sh` / `k_all.sh`. Batch size does not change results.

```bash
python scripts/benchmark.py --config configs/experiments/payoff_k2.yaml --auto        # largest batch that fits
python scripts/benchmark.py --config configs/experiments/payoff_k2.yaml --max_parallel_runs 10
```
