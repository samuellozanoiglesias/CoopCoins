# CoopCoins — coin-targeting agents that play on any map

Two teams (red = even agents, blue = odd agents) on a G×G torus, as many coins as agents (half of
each colour). Every step each agent **chooses a coin (or stay)** and the environment moves it one
cell along the shortest path, with the same rule for everybody. Taking your team's coin gives +1.
Taking the other team's coin gives +1 to you and −k to the robbed team, split among its members
(`payoff: {own: 1, other: 1, penalty: -k}` in the experiment config; k = 2 in `scaling.yaml`). If several agents land
on the same coin in the same step, one of them, drawn uniformly at random, collects it.

**Observation:** one row per coin, in raw step counts that mean the same on every map: whether
it is my team's coin, my distance, my nearest teammate's distance, the nearest opponent's
distance, how many teammates and opponents targeted it last step, and whether I have teammates; plus four memory
features (exponential moving averages, `memory: {decay: 0.9}`): how often my team was robbed, how many opponents
targeted my colour, how often my team stole, and how many of my team targeted the other colour.

**Training:** PPO, one policy per team; actor and critic gradients clipped separately; returns normalised with
running statistics per team (the critic predicts normalised values).

**Policy:** a set network scores every coin plus "stay". It is shared by all members of a team
and works with any number of coins, so the same checkpoint plays any grid size and team size.

**Attitudes (SVO):** within a team agents are always cooperative. Between teams:
`r_i = α·mean(my team's score) + β·mean(other team's score)`, `(α, β) = (cos θ, sin θ)` of the
team's attitude. With 2 agents this is exactly the original SVO.

## Experiment (`configs/experiments/scaling.yaml`)

* Same game length (150 steps) on every map, in training and simulation.
* **density** family: agents per cell ≈ 2/9; grids 3–10 with 2, 4, 6, 8, 10, 14, 18, 22 agents.
* Pairs (red vs blue): ind–ind, ind–coop, ind–comp; 10 training seeds; 20 games per trained pair.
* Training sets: `all_maps` (generalist, trained on every map at once) and one specialist set per map.

```bash
./run_scaling.sh                      # train -> simulate -> analyze, everything
RENDER=1 ./run_scaling.sh             # also GIFs of simulated games
TRAINING_ID=<id> STEPS="simulate analyze" ./run_scaling.sh
python simulate.py --dirs <data_root>/all_maps/*/Training_<id> --maps grid12_agents32 --render   # unseen map
```

## Results layout

```
<data_root>/<all_maps | map name>/<red_blue>/Training_<id>/
    config.yaml  training_stats.csv  training_curves.png  summary.yaml
    checkpoints/seed_<s>/episode_<n>.pkl                 {'team_0': params, 'team_1': params}
    simulations/<map>/episode_<n>_stochastic/            games.csv summary.yaml report.png [gifs/]
<data_root>/analysis/scaling_<id>/                       tables + figures (see analyze.py docstring)
<data_root>/logs/
```

## Before the full run

```bash
./smoke_test.sh     # whole pipeline on tiny settings in /tmp; must end with "SMOKE TEST PASSED"
./speed_test.sh     # GPU + CPU speed with the full settings, extrapolated; writes speed_test_<time>.log
```

## Theft-penalty sweep (k = 0, 2, 10)

See [`docs/experiment_k.md`](docs/experiment_k.md): `./scripts/k_smoke.sh` first, then `./scripts/k_all.sh`.

## Social-influence study (strategic vs ecological)

`configs/influence.yaml` sets the focal attitude, the partners (first = reference) and which trainings each
experiment uses. Definitions are in `coopcoins/influence.py` and `analyze_influence.py`.

| script | what | trains? | output |
|---|---|---|---|
| `run_exp1.sh` | cross-play decomposition, scaling_v1 agents | no | `influence/scaling_v1/crossplay_games.csv`, `influence/analysis/exp1/` |
| `run_exp2.sh` | synthetic + state-matched policy probes | no | `influence/scaling_v1/probe_*.csv`, `influence/analysis/exp2/` |
| `run_exp3.sh` | scarcity x group size (new trainings) | yes, ~3.5 h GPU | `scarcity_v1` trainings, `influence/scarcity_v1/`, `influence/analysis/exp3/` |
| `run_exp4.sh` | observational estimates vs truth | no | `influence/analysis/exp4/` |
