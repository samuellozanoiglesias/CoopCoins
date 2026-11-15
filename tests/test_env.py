"""A7 tests: tie-break, memory, payoff (env.py)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coopcoins.env import MEMORY, N_BASE, N_FEATURES, State, make_env

MEM = {"decay": 0.9}


def env_k(k, G=5, N=2, M=2, **kw):
    return make_env(G, N, num_coins=M, payoff={"own": 1, "other": 1, "penalty": -k}, memory=MEM, **kw)


def state(agents, coins, memory=None, target=None):
    a, c = jnp.array(agents, jnp.int32), jnp.array(coins, jnp.int32)
    n = len(agents)
    return State(a, c, jnp.full(n, -1, jnp.int32) if target is None else jnp.array(target, jnp.int32),
                 jnp.zeros((n, len(MEMORY)), jnp.float32) if memory is None else memory)


def batch_step(env, s, actions, n_keys):
    keys = jax.random.split(jax.random.PRNGKey(123), n_keys)
    return jax.vmap(lambda k: env.step(k, s, jnp.array(actions, jnp.int32)))(keys)


# ------------------------------------------------------------------ tie-break
def test_two_agents_one_coin_exactly_one_reward_and_penalty_only_if_thief_wins():
    k = 2.0
    env = env_k(k)
    # coin 0 (red, team 0) at (2,2); red agent 0 at (2,1), blue agent 1 at (2,3); both go to coin 0
    s = state([[2, 1], [2, 3]], [[2, 2], [0, 0]])
    _, _, pure, info, taken = batch_step(env, s, [0, 0], 4000)
    pure, own, other = np.asarray(pure), np.asarray(info["own_coins"]), np.asarray(info["other_coins"])
    assert np.all(np.asarray(taken)[:, 0])
    assert np.all(own[:, 0] + other[:, 1] == 1), "exactly one collector per coin"
    red_wins = own[:, 0] == 1
    # red wins: +1 to red, nothing to blue (no theft). blue wins: +1 to blue, -k / team_size to red
    assert np.allclose(pure[red_wins], [1.0, 0.0])
    assert np.allclose(pure[~red_wins], [-k, 1.0])
    assert abs(red_wins.mean() - 0.5) < 0.04
    # tie stats and misses
    assert np.all(np.asarray(info["tie_arrivals"]) == 1)
    tw = np.asarray(info["tie_wins"])
    assert np.all(tw.sum(1) == 1) and np.all(tw[:, 0] == red_wins)
    assert np.all(np.asarray(info["missed_own"])[:, 0] == (~red_wins))       # red lost a reachable own coin
    assert np.all(np.asarray(info["missed_other"])[:, 1] == red_wins)       # blue lost a reachable enemy coin


@pytest.mark.parametrize("n_arrive", [2, 3, 4])
def test_win_rate_is_one_over_n(n_arrive):
    env = env_k(2.0, G=5, N=4, M=4)
    around = [[2, 1], [2, 3], [1, 2], [3, 2]]                              # 4 neighbours of (2,2)
    agents = around[:n_arrive] + [[0, 0], [4, 4], [0, 4]][: 4 - n_arrive]
    coins = [[2, 2], [0, 2], [4, 0], [4, 2]]
    acts = [0] * n_arrive + [4] * (4 - n_arrive)                           # 4 = stay
    _, _, _, info, _ = batch_step(env, state(agents, coins), acts, 6000)
    wins = np.asarray(info["own_coins"] + info["other_coins"])[:, :n_arrive]
    assert np.all(wins.sum(1) == 1)
    assert np.allclose(wins.mean(0), 1 / n_arrive, atol=0.03)
    assert np.all(np.asarray(info["tie_arrivals"])[:, :n_arrive] == 1)
    assert np.all(np.asarray(info["tie_arrivals"])[:, n_arrive:] == 0)


def test_no_tie_single_arrival():
    env = env_k(2.0)
    s = state([[2, 1], [0, 4]], [[2, 2], [4, 4]])
    _, _, pure, info, _ = batch_step(env, s, [0, 2], 50)                    # agent 1 stays
    assert np.all(np.asarray(info["own_coins"])[:, 0] == 1) and np.all(np.asarray(info["tie_arrivals"]) == 0)
    assert np.allclose(np.asarray(pure), [1.0, 0.0])


# ------------------------------------------------------------------ memory
def test_memory_three_step_sequence_and_reset():
    env = env_k(2.0)
    key = jax.random.PRNGKey(0)
    obs, s0 = env.reset(key)
    assert obs["coins"].shape[-1] == N_FEATURES == 11
    assert np.all(np.asarray(s0.memory) == 0) and np.all(np.asarray(obs["coins"][..., N_BASE:]) == 0)

    # step 1: blue (agent 1) steals red coin 0; red (agent 0) heads for blue coin 1 but does not reach it
    s = state([[2, 1], [2, 3]], [[2, 2], [0, 0]])
    s = s._replace(agent_pos=jnp.array([[4, 4], [2, 3]], jnp.int32))
    _, s1, _, info, _ = env.step(key, s, jnp.array([1, 0]))
    assert int(info["other_coins"][1]) == 1
    x1 = np.array([[1, 1, 0, 1],      # red:  robbed 1, opponent targeted my colour, stole 0, I targeted theirs
                   [0, 1, 1, 1]])     # blue: robbed 0, red targeted blue coin, stole 1, I targeted theirs
    m1 = 0.1 * x1
    assert np.allclose(np.asarray(s1.memory), m1, atol=1e-6)

    # step 2: both stay -> x = 0
    s2_in = state([[0, 0], [4, 4]], [[2, 2], [1, 3]], memory=s1.memory)
    _, s2, _, _, _ = env.step(key, s2_in, jnp.array([2, 2]))
    m2 = 0.9 * m1
    assert np.allclose(np.asarray(s2.memory), m2, atol=1e-6)

    # step 3: red takes its own coin 0; blue heads for its own coin 1 (does not reach)
    s3_in = state([[2, 1], [4, 4]], [[2, 2], [1, 3]], memory=s2.memory)
    obs3, s3, _, info3, _ = env.step(key, s3_in, jnp.array([0, 1]))
    assert int(info3["own_coins"][0]) == 1
    x3 = np.zeros((2, 4))             # no theft; nobody targeted the other colour
    m3 = 0.9 * m2 + 0.1 * x3
    assert np.allclose(np.asarray(s3.memory), m3, atol=1e-6)
    # the observation carries the memory on every coin row
    assert np.allclose(np.asarray(obs3["coins"][:, :, N_BASE:]), m3[:, None, :], atol=1e-6)

    obs_r, s_r = env.reset(jax.random.PRNGKey(5))
    assert np.all(np.asarray(s_r.memory) == 0)


def test_memory_fractions_with_teams():
    env = env_k(2.0, G=6, N=4, M=4)    # red = agents 0, 2; blue = 1, 3; coin c belongs to team c % 2
    s = state([[0, 0], [0, 3], [3, 0], [3, 3]], [[5, 5], [5, 2], [2, 5], [2, 2]])
    # agent 0 -> coin 1 (blue), agent 2 stays, agent 1 -> coin 0 (red), agent 3 -> coin 3 (blue)
    _, s1, _, _, _ = env.step(jax.random.PRNGKey(0), s, jnp.array([1, 0, 4, 3]))
    m = np.asarray(s1.memory) / 0.1
    assert np.allclose(m[[0, 2], 1], 0.5) and np.allclose(m[[0, 2], 3], 0.5)   # 1 of 2 opponents / 1 of 2 of us
    assert np.allclose(m[[1, 3], 1], 0.5) and np.allclose(m[[1, 3], 3], 0.5)
    assert np.allclose(m[:, [0, 2]], 0)


# ------------------------------------------------------------------ payoff
@pytest.mark.parametrize("k", [0, 2, 10])
def test_payoff_rewards(k):
    env = env_k(k, G=6, N=4, M=4)     # team_size 2
    # blue agent 1 steals red coin 0; red agent 2 takes its own coin 2
    s = state([[0, 1], [1, 0], [3, 1], [5, 5]], [[0, 0], [4, 4], [3, 0], [2, 2]])
    _, _, pure, info, _ = env.step(jax.random.PRNGKey(0), s, jnp.array([4, 0, 2, 4]))
    assert np.allclose(np.asarray(pure), [-k / 2, 1.0, 1.0 - k / 2, 0.0])
    assert int(info["other_coins"][1]) == 1 and int(info["own_coins"][2]) == 1
    assert env.payoff == (1.0, 1.0, -float(k))


def test_unknown_kwarg_raises():
    with pytest.raises(TypeError):
        make_env(5, 2, payoff={"own": 1, "other": 1, "penalty": -2}, memory=MEM, payof={"own": 1})
    with pytest.raises(TypeError):
        make_env(5, 2, memory=MEM)                                          # payoff is required
    with pytest.raises(ValueError):
        make_env(5, 2, payoff={"own": 1, "other": 1, "penality": -2}, memory=MEM)
    # map metadata keys are accepted
    make_env(5, 2, payoff={"own": 1, "other": 1, "penalty": -2}, memory=MEM, name="x", density=0.1, ratio=1.0,
             coins_per_agent=1.0, families=["density"])


def test_accounting_identity_in_simulated_games():
    """pure = own * p_own + other * p_other - k * (partner team's thefts per agent), per team and game."""
    from coopcoins.networks import init_policy
    from coopcoins.simulation import SIM_STATS, make_simulator
    for k in (0, 2, 10):
        env = env_k(k, G=4, N=4, M=4)
        params = jax.vmap(lambda kk: init_policy(kk, 16))(jax.random.split(jax.random.PRNGKey(k), 2))
        params = jax.tree.map(lambda x: x[None], params)
        res = np.asarray(make_simulator(env)(params, jnp.eye(2), jax.random.split(jax.random.PRNGKey(1), 8)))[0]
        st = {n: res[:, i] for i, n in enumerate(SIM_STATS)}                     # (S, N)
        tm = lambda x, t: x[:, env.agent_team == t].mean(1)
        for t in (0, 1):
            pred = tm(st["own_coins"], t) + tm(st["other_coins"], t) - k * tm(st["other_coins"], 1 - t)
            assert np.allclose(tm(st["pure_reward"], t), pred, atol=1e-3)
