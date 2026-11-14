# CoopCoins — coin-targeting agents that play on any map

Two teams (red = even agents, blue = odd agents) on a G×G torus, as many coins as agents (half of
each colour). Every step each agent **chooses a coin (or stay)** and the environment moves it one
cell along the shortest path, with the same rule for everybody. Taking your team's coin gives +1.
Taking the other team's coin gives +1 to you and −2 to the robbed team, split among its members.

**Observation:** one row per coin, in raw step counts that mean the same on every map: whether
it is my team's coin, my distance, my nearest teammate's distance, the nearest opponent's
distance, how many teammates and opponents targeted it last step, and whether I have teammates.

**Policy:** a set network scores every coin plus "stay". It is shared by all members of a team
and works with any number of coins, so the same checkpoint plays any grid size and team size.

**Attitudes (SVO):** within a team agents are always cooperative. Between teams:
`r_i = α·mean(my team's score) + β·mean(other team's score)`, `(α, β) = (cos θ, sin θ)` of the
team's attitude. With 2 agents this is exactly the original SVO.

## Experiment (`configs/experiments/scaling.yaml`)

* Same game length (150 steps) on every map, in training and simulation.
* **density** family: agents per cell ≈ 2/9; grids 3–10 with 2, 4, 6, 8, 10, 14, 18, 22 agents.
* **ratio** family: grid size / agents = 1.5; grids 3, 6, 9 with 2, 4, 6 agents.
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
