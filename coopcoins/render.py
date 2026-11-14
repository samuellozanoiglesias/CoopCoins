"""Render a recorded game as an animated GIF (PIL only).

Agents = triangles pointing where they last moved; coins = circles; red / blue = team.
A coloured ring around a coin marks that an agent of that team is heading for it.
Header: each team's cumulative score (sum of its members' pure rewards) and coins taken.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEAM = [(214, 48, 49), (9, 92, 186)]
COIN = [(250, 160, 150), (140, 185, 240)]
BG, BOARD, LINE, TEXT = (250, 250, 247), (236, 236, 230), (200, 200, 192), (40, 40, 40)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _triangle(cx, cy, r, d):
    dx, dy = d
    px, py = -dy, dx
    return [(cx + dx * r, cy + dy * r),
            (cx - dx * r * 0.7 + px * r * 0.8, cy - dy * r * 0.7 + py * r * 0.8),
            (cx - dx * r * 0.7 - px * r * 0.8, cy - dy * r * 0.7 - py * r * 0.8)]


def render_gif(init_state, traj, env, path, title="", frame_ms=180):
    states, _, pure, info = traj
    G, N = env.grid_size, env.num_agents
    A = np.concatenate([np.asarray(init_state.agent_pos)[None], np.asarray(states.agent_pos)])  # (T+1, N, 2)
    C = np.concatenate([np.asarray(init_state.coin_pos)[None], np.asarray(states.coin_pos)])
    TG = np.concatenate([np.asarray(init_state.target)[None], np.asarray(states.target)])
    pure, own, other = np.asarray(pure), np.asarray(info["own_coins"]), np.asarray(info["other_coins"])
    team, cteam = env.agent_team, env.coin_team
    T = pure.shape[0]

    cs = max(24, min(125, 480 // G))
    pad, head = 12, 74
    W, H = G * cs + 2 * pad, G * cs + 2 * pad + head + 18
    f_big, f_small, f_tiny = _font(18), _font(13), _font(max(8, cs // 4))
    f_title = f_small
    for size in (13, 12, 11, 10, 9):
        f_title = _font(size)
        if ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(title, font=f_title) <= W - 2 * pad:
            break

    def xy(p):  # row increases upwards
        return pad + (p[1] + 0.5) * cs, head + pad + (G - 1 - p[0] + 0.5) * cs

    direction = [(0, -1)] * N
    score, coins = np.zeros(2), np.zeros((2, 2), int)
    frames = []
    for t in range(T + 1):
        if t > 0:
            dpos = (A[t] - A[t - 1] + G // 2) % G - G // 2
            for i in range(N):
                if dpos[i].any():
                    direction[i] = (int(dpos[i, 1]), -int(dpos[i, 0]))
            for k in (0, 1):
                score[k] += pure[t - 1, team == k].sum()
                coins[k] += [own[t - 1, team == k].sum(), other[t - 1, team == k].sum()]
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((pad, 6), title, fill=TEXT, font=f_title)
        for k, x in ((0, pad), (1, W // 2 + 4)):
            d.text((x, 26), f"{'Red' if k == 0 else 'Blue'}  {score[k]:+.0f}", fill=TEAM[k], font=f_big)
            d.text((x, 50), f"own {coins[k, 0]}  stolen {coins[k, 1]}", fill=TEXT, font=f_small)
        d.text((W // 2, H - 14), f"step {t}/{T}", fill=TEXT, font=f_small, anchor="mm")
        d.rectangle([pad, head + pad, pad + G * cs, head + pad + G * cs], fill=BOARD)
        for k in range(G + 1):
            d.line([pad + k * cs, head + pad, pad + k * cs, head + pad + G * cs], fill=LINE)
            d.line([pad, head + pad + k * cs, pad + G * cs, head + pad + k * cs], fill=LINE)
        for c in range(C.shape[1]):
            x, y = xy(C[t, c]); r = cs * 0.2
            for j, k in enumerate(sorted({int(team[i]) for i in range(N) if TG[t, i] == c})):
                rr = r + 4 + 4 * j
                d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=TEAM[k], width=2)
            d.ellipse([x - r, y - r, x + r, y + r], fill=COIN[cteam[c]], outline=TEAM[cteam[c]], width=2)
        cells = {}
        for i in range(N):
            cells.setdefault(tuple(A[t, i]), []).append(i)
        for pos, ids in cells.items():
            x0, y0 = xy(pos)
            for j, i in enumerate(ids):
                off = 0 if len(ids) == 1 else cs * 0.22
                ang = 2 * np.pi * j / len(ids)
                x, y = x0 + off * np.cos(ang), y0 + off * np.sin(ang)
                r = cs * (0.32 if len(ids) == 1 else 0.2)
                d.polygon(_triangle(x, y, r, direction[i]), fill=TEAM[team[i]], outline=(255, 255, 255))
                if N > 2 and cs >= 40:
                    d.text((x, y), str(i), fill=(255, 255, 255), font=f_tiny, anchor="mm")
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0, optimize=True)
