"""
Slither Booper - Python training arena
======================================

A simplified slither.io world where 3 bots run the same AI as the Slither Booper
userscript (version 3.6):

  * safety filter: thin "squeeze" hitbox, bodies as solid segments, predicted enemy
    heads, turn-aware sweep check (never turns the short way into a body)
  * game plan: GROW (eat, head to the centre, no hunting) until length 500, then
    FIGHT in the centre, food always first, limited boosting in fights
  * big food piles, eating the victim after a boop
  * hunting: target choice (bait, big snakes, boop net), SHADOW -> CUT -> HOLD with
    5 boop styles (TIGHT CUT, WIDE CUT, WRAP, DEEP CUT, ENCIRCLE)
  * every kill is rated from 1 to 100 (same rules as the userscript)
  * learning: settings-learning (45 s experiments) per bot, and 3 neural networks
    (danger / boop / style) with the same inputs and layer sizes as the userscript

The 3 bots share one "brain" (the 3 networks), so it learns 3x faster than one bot.
The world runs much faster than real time, so a lot of experience piles up quickly.

Run:
    python slither_bots.py                 # train headless, print stats
    python slither_bots.py --steps 200000  # longer run
    python slither_bots.py --view          # watch it live in a pygame window (pip install pygame)
    python slither_bots.py --snapshot a.png --steps 3000   # save one picture
    python slither_bots.py --gif bots.gif --steps 6000     # train a bit, then save an animation
    python slither_bots.py --kill-photos kills --steps 60000   # one picture per kill-rating band

The brain (networks, memories, learned settings) is saved to booper_brain.json and loaded
again next run.

ONE SHARED BRAIN with the browser bot (live sync):
While this program runs, it also runs a tiny local server (http://127.0.0.1:8765, only
reachable from your own computer). The Tampermonkey bot connects to it automatically:
  * every 30 s it sends its REAL-GAME experiences here (deaths, boops, styles), and the
    shared networks train on them together with the simulator's experiences
  * and it downloads the updated networks, so the browser bot keeps getting better
So both bots use and improve the same brain. Turn it off with --no-sync.
(Without the simulator running, the browser bot just keeps learning on its own.)
You can still move the brain by hand: in slither.io press O and pick booper_brain.json.
Needs: numpy. pygame for --view (pip install pygame); matplotlib for --snapshot / --gif.

Viewer keys:  1 / 2 / 3 follow a bot   0 follow the action automatically
              mouse wheel or + / -  zoom     up / down  simulation speed
              space  pause     F  show/hide food     R  show/hide aim lines     Esc  quit (saves)
"""

import argparse
import base64
import bisect
import json
import math
import os
import random
import threading
import time

import numpy as np

TAU = 2 * math.pi

# ----------------------------------------------------------------------------
# settings (same names and meaning as CFG in the userscript)
# ----------------------------------------------------------------------------
CFG = dict(
    # boop geometry
    leadBase=60, sideOffset=55, cutAheadMargin=15, cutLateralMax=260, cutOvershoot=160,
    holdMs=320, shadowForceMs=2200, shadowGiveUpMs=4000,
    # targeting
    huntRadius=1600, stickiness=250, bigTargetBonus=120, baitBonus=300, boopNetWeight=500,
    # safety
    scanRadius=900, squeezeHitbox=0.72, squeezeClearance=36, angleSamples=24,
    riskMax=0.55,
    # food / plan
    foodRadius=900, bigFoodRadius=1400, bigFoodMinCount=8, bigFoodMass=600, bigFoodVsNormal=3,
    fightLength=500, centerZone=5000, fightBoostBudget=1800, feastMs=5000,
    minBoostLength=20, centerPull=1.2,
    # learning
    trialMs=45000, styleExplore=0.2,
)
TUNE = dict(leadBase=(20, 140), sideOffset=(30, 120), cutAheadMargin=(0, 70), cutOvershoot=(80, 260),
            holdMs=(150, 550), shadowForceMs=(900, 3500), squeezeHitbox=(0.66, 0.95), squeezeClearance=(28, 60))

TICK_MS = 33              # one simulation step = one bot decision (~30 per second, like the userscript)
MAP_R = 21600             # map radius, same as slither.io
SPACING = 42              # distance between body points
NORMAL_SPEED = 185 * TICK_MS / 1000   # world units per tick
BOOST_SPEED = 448 * TICK_MS / 1000

STYLE_POINTS = {'TIGHT CUT': 8, 'WIDE CUT': 4, 'WRAP': 10, 'DEEP CUT': 8, 'ENCIRCLE': 6}

STYLES = [
    dict(name='TIGHT CUT', lead=0.7, over=0.6, hold=1.0),
    dict(name='WIDE CUT', lead=1.2, over=1.35, hold=1.0),
    dict(name='WRAP', lead=1.0, over=1.0, hold=1.8, wrap=True),
    dict(name='DEEP CUT', lead=0.6, over=0.8, hold=1.2, deep=70),
    dict(name='ENCIRCLE', lead=0.9, over=1.0, hold=2.8, circle=True, minRatio=1.5),
]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def ang_diff(a, b):
    d = (a - b) % TAU
    if d > math.pi:
        d -= TAU
    return d


# ----------------------------------------------------------------------------
# length <-> segments (slither.io's own formula)
# ----------------------------------------------------------------------------
MSCPS = 411
FM, FP = [], []
for _i in range(MSCPS + 2):
    FM.append(FM[-1] if _i >= MSCPS else (1 - _i / MSCPS) ** 2.25)
    FP.append(0 if _i == 0 else FP[_i - 1] + 1 / FM[_i - 1])


def score_of(sct, fam=0.0):
    i = int(clamp(sct, 0, len(FP) - 1))
    return max(0, int(15 * (FP[i] + fam / max(1e-4, FM[i]) - 1) - 5))


SCORES = [score_of(i) for i in range(MSCPS + 1)]   # length for each number of segments


# ----------------------------------------------------------------------------
# the world
# ----------------------------------------------------------------------------
class Snake:
    _ids = 0

    def __init__(self, x, y, ang, score=10, name='npc', is_bot=False):
        Snake._ids += 1
        self.id = Snake._ids
        self.name = name
        self.is_bot = is_bot
        self.x, self.y, self.ang = x, y, ang
        self.score = float(score)
        self.boosting = False
        self.alive = True
        self.want_ang = ang
        self.want_boost = False
        n = self.segments()
        self.pts = [(x - math.cos(ang) * SPACING * i, y - math.sin(ang) * SPACING * i) for i in range(n, 0, -1)]
        self.v = NORMAL_SPEED
        self.w = 0.0

    def segments(self):
        # inverse of score_of: how many body points this length has (cached)
        key = int(self.score)
        if getattr(self, '_seg_key', None) != key:
            self._seg_key = key
            self._seg = max(2, min(MSCPS, bisect.bisect_right(SCORES, key) - 1))
            self._sc = min(6.0, 1 + (self._seg - 2) / 106.0)
        return self._seg

    def sc(self):
        self.segments()
        return self._sc

    def width(self):
        return self.sc() * 29

    def turn_rate(self):
        return 0.13 / (1 + (self.sc() - 1) * 0.35)


class World:
    def __init__(self, n_npc=12, n_food=4000, seed=None):
        self.rng = random.Random(seed)
        self.snakes = []
        self.food = np.zeros((0, 3))  # x, y, size
        self.n_food = n_food
        self.tick_no = 0
        self.kills = []   # (killer, victim) this tick
        for i in range(n_npc):
            self.snakes.append(self.spawn_npc())
        self.top_up_food()

    def rand_point(self, r_max):
        r = r_max * math.sqrt(self.rng.random())
        a = self.rng.random() * TAU
        return MAP_R + r * math.cos(a), MAP_R + r * math.sin(a)

    def spawn_npc(self):
        # NPCs live mostly near the middle so bots meet them
        x, y = self.rand_point(7000)
        s = Snake(x, y, self.rng.random() * TAU, score=self.rng.choice([10, 60, 200, 600, 1500, 4000]), name='npc')
        s.mind = dict(turn=0.0, t=0)
        return s

    def top_up_food(self):
        missing = self.n_food - len(self.food)
        if missing <= 0:
            return
        new = []
        for _ in range(missing):
            x, y = self.rand_point(9000)
            new.append((x, y, self.rng.uniform(4, 8)))
        self.food = np.vstack([self.food, np.array(new)]) if len(self.food) else np.array(new)

    def drop_food(self, pts, big):
        add = []
        for i, (x, y) in enumerate(pts):
            if big or i % 3 == 0:
                add.append((x + self.rng.uniform(-10, 10), y + self.rng.uniform(-10, 10), self.rng.uniform(10, 16) if big else 4))
        if add:
            self.food = np.vstack([self.food, np.array(add)])

    # simple human-ish NPC: wanders to food, avoids walls and bodies, sometimes boosts
    def npc_think(self, s):
        m = s.mind
        m['t'] -= 1
        if m['t'] <= 0:
            m['turn'] = self.rng.uniform(-0.6, 0.6)
            m['t'] = self.rng.randint(20, 80)
        want = s.ang + m['turn'] * 0.1
        if len(self.food):
            d2 = (self.food[:, 0] - s.x) ** 2 + (self.food[:, 1] - s.y) ** 2
            i = int(np.argmin(d2))
            if d2[i] < 600 ** 2:
                want = math.atan2(self.food[i, 1] - s.y, self.food[i, 0] - s.x)
        # look a little ahead for bodies
        ahead = (s.x + math.cos(s.ang) * 90, s.y + math.sin(s.ang) * 90)
        for o in self.snakes:
            if o is s or not o.alive:
                continue
            if (o.x - s.x) ** 2 + (o.y - s.y) ** 2 > 1500 ** 2:
                continue
            for (px, py) in o.pts[::3]:
                if (px - ahead[0]) ** 2 + (py - ahead[1]) ** 2 < (o.width() * 0.8 + 30) ** 2 and self.rng.random() < 0.85:
                    want = s.ang + (1.4 if self.rng.random() < 0.5 else -1.4)
                    break
        dc = math.hypot(s.x - MAP_R, s.y - MAP_R)
        if dc > MAP_R * 0.85:
            want = math.atan2(MAP_R - s.y, MAP_R - s.x)
        s.want_ang = want
        s.want_boost = self.rng.random() < 0.02 or (s.want_boost and self.rng.random() < 0.9)

    def step(self, bots):
        self.tick_no += 1
        self.kills = []
        for s in self.snakes:
            if s.alive and not s.is_bot:
                self.npc_think(s)
        for b in bots:
            if b.snake.alive:
                b.think(self)
        # move
        for s in self.snakes:
            if not s.alive:
                continue
            d = ang_diff(s.want_ang, s.ang)
            tr = s.turn_rate()
            turn = clamp(d, -tr, tr)
            s.ang = (s.ang + turn) % TAU
            s.boosting = s.want_boost and s.score >= 20
            spd = BOOST_SPEED if s.boosting else NORMAL_SPEED
            s.x += math.cos(s.ang) * spd
            s.y += math.sin(s.ang) * spd
            if s.boosting:
                s.score = max(10, s.score - 0.6)
                if self.rng.random() < 0.3:
                    tx, ty = s.pts[0]
                    self.food = np.vstack([self.food, [[tx, ty, 6]]])
            lx, ly = s.pts[-1]
            if (s.x - lx) ** 2 + (s.y - ly) ** 2 >= SPACING ** 2:
                s.pts.append((s.x, s.y))
            n = s.segments()
            if len(s.pts) > n:
                s.pts = s.pts[len(s.pts) - n:]
        # eat
        if len(self.food):
            keep = np.ones(len(self.food), dtype=bool)
            for s in self.snakes:
                if not s.alive:
                    continue
                r = s.width() / 2 + 14
                d2 = (self.food[:, 0] - s.x) ** 2 + (self.food[:, 1] - s.y) ** 2
                hit = (d2 < r * r) & keep
                if hit.any():
                    s.score += float(self.food[hit, 2].sum()) * 0.35
                    keep &= ~hit
            self.food = self.food[keep]
        # collisions: a head touching another snake's body dies, the body's owner gets the kill
        grid = {}
        for s in self.snakes:
            if not s.alive:
                continue
            for (px, py) in s.pts:
                grid.setdefault((int(px // 100), int(py // 100)), []).append((px, py, s))
        dead = []
        for s in self.snakes:
            if not s.alive:
                continue
            if math.hypot(s.x - MAP_R, s.y - MAP_R) > MAP_R * 0.98:
                dead.append((s, None))
                continue
            cx, cy = int(s.x // 100), int(s.y // 100)
            hit = None
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for (px, py, o) in grid.get((cx + i, cy + j), ()):
                        if o is s:
                            continue
                        rr = o.width() * 0.5 + s.width() * 0.35
                        if (px - s.x) ** 2 + (py - s.y) ** 2 < rr * rr:
                            hit = o
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                dead.append((s, hit))
        for s, killer in dead:
            if not s.alive:
                continue
            s.alive = False
            self.drop_food(s.pts, big=True)
            self.kills.append((killer, s))
        # respawn NPCs, refill food
        for i, s in enumerate(self.snakes):
            if not s.alive and not s.is_bot:
                self.snakes[i] = self.spawn_npc()
        if self.tick_no % 30 == 0:
            if len(self.food) > self.n_food * 1.6:
                self.food = self.food[-int(self.n_food * 1.6):]
            self.top_up_food()


# ----------------------------------------------------------------------------
# neural networks (same layer sizes as the userscript), trained in mini-batches
# ----------------------------------------------------------------------------
class Net:
    def __init__(self, sizes, data=None, rng=None):
        self.sizes = sizes
        rng = rng or np.random.default_rng()
        ok = data and data.get('sizes') == sizes
        self.W, self.B, self.vW, self.vB = [], [], [], []
        for l in range(len(sizes) - 1):
            n_in, n_out = sizes[l], sizes[l + 1]
            w = np.array(data['W'][l], dtype=float).reshape(n_out, n_in) if ok else rng.normal(0, math.sqrt(1 / n_in), (n_out, n_in))
            b = np.array(data['B'][l], dtype=float) if ok else np.zeros(n_out)
            self.W.append(w); self.B.append(b)
            self.vW.append(np.zeros_like(w)); self.vB.append(np.zeros_like(b))

    def predict(self, X):
        a = np.atleast_2d(np.nan_to_num(np.asarray(X, dtype=float)))
        for l, (w, b) in enumerate(zip(self.W, self.B)):
            z = a @ w.T + b
            a = 1 / (1 + np.exp(-np.clip(z, -30, 30))) if l == len(self.W) - 1 else np.tanh(z)
        return a[:, 0]

    def train(self, X, y, lr=0.02):
        acts = [np.nan_to_num(np.asarray(X, dtype=float))]
        for l, (w, b) in enumerate(zip(self.W, self.B)):
            z = acts[-1] @ w.T + b
            acts.append(1 / (1 + np.exp(-np.clip(z, -30, 30))) if l == len(self.W) - 1 else np.tanh(z))
        p = acts[-1][:, 0]
        n = len(p)
        delta = (p - y)[:, None] / n
        for l in range(len(self.W) - 1, -1, -1):
            gW = delta.T @ acts[l]
            gB = delta.sum(0)
            if l > 0:
                delta = (delta @ self.W[l]) * (1 - acts[l] ** 2)
            self.vW[l] = 0.9 * self.vW[l] - lr * gW; self.W[l] += self.vW[l]
            self.vB[l] = 0.9 * self.vB[l] - lr * gB; self.B[l] += self.vB[l]
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())

    def save(self):
        return dict(sizes=self.sizes, W=[w.flatten().round(5).tolist() for w in self.W], B=[b.round(5).tolist() for b in self.B])


class Memory:
    """Experiences with labels; old ones are replaced at random when full."""

    def __init__(self, n_in, cap):
        self.X = np.zeros((cap, n_in)); self.y = np.zeros(cap); self.n = 0; self.cap = cap

    def add(self, x, y):
        i = self.n if self.n < self.cap else random.randrange(self.cap)
        self.X[i] = x; self.y[i] = y
        self.n = min(self.cap, self.n + 1)

    def sample(self, k):
        idx = np.random.randint(0, self.n, k)
        return self.X[idx], self.y[idx]


DANGER_IN, BOOP_IN = 43, 21
STYLE_IN = BOOP_IN + len(STYLES)


SYNC_PORT = 8765


def start_sync_server(brain, port=SYNC_PORT):
    """Tiny local server so the Tampermonkey bot can share this brain. Only listens on 127.0.0.1."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _headers(self, code=200, ctype='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Private-Network', 'true')
            self.end_headers()

        def do_OPTIONS(self):
            self._headers(204)

        def do_GET(self):
            if self.path.startswith('/brain'):
                body = brain.nets_json().encode()
                self._headers(200)
                self.wfile.write(body)
            else:
                self._headers(200, 'text/plain')
                self.wfile.write(b'Slither Booper simulator is running. The browser bot syncs with /brain.')

        def do_POST(self):
            if not self.path.startswith('/experiences'):
                self._headers(404); return
            try:
                n = int(self.headers.get('Content-Length', 0))
                if n > 20_000_000:
                    raise ValueError('too big')
                payload = json.loads(self.rfile.read(n) or b'{}')
                brain.inbox.append(payload)
                self._headers(200)
                self.wfile.write(b'{"ok":true}')
            except Exception as e:
                self._headers(400)
                self.wfile.write(json.dumps(dict(ok=False, error=str(e))).encode())

        def log_message(self, *a):
            pass

    try:
        srv = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    except OSError as e:
        print('sync server could not start on port %d (%s) - browser sync is off' % (port, e))
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print('sync server running on http://127.0.0.1:%d - the Tampermonkey bot will connect by itself' % port)
    return srv


def export_path(brain_path):
    folder = os.path.dirname(os.path.abspath(brain_path))
    return os.path.join(folder, 'booper_brain_for_tampermonkey.json')


class Brain:
    """One brain shared by all 3 bots."""

    def __init__(self, path):
        self.path = path
        d = None
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    d = json.load(f)
            except Exception:
                d = None
        self.danger = Net([DANGER_IN, 48, 24, 1], d and d.get('danger'))
        self.boop = Net([BOOP_IN, 32, 16, 1], d and d.get('boop'))
        self.style = Net([STYLE_IN, 32, 16, 1], d and d.get('style'))
        self.dPos, self.dNeg = Memory(DANGER_IN, 3000), Memory(DANGER_IN, 6000)
        self.bPos, self.bNeg = Memory(BOOP_IN, 2000), Memory(BOOP_IN, 4000)
        self.sPos, self.sNeg = Memory(STYLE_IN, 2000), Memory(STYLE_IN, 4000)
        self.loss = dict(danger=0.69, boop=0.69, style=0.69)
        self.lock = threading.Lock()
        self.inbox = []          # experiences from the browser bot, added by the main loop
        # remembered experiences are kept next to the brain file, so learning carries on next run
        loaded_mem = False
        if d:
            for name in ('dPos', 'dNeg', 'bPos', 'bNeg', 'sPos', 'sNeg'):
                pk = d.get(name)
                m = getattr(self, name)
                if isinstance(pk, dict) and pk.get('n') == m.X.shape[1] + 1 and pk.get('d'):
                    try:
                        rows = np.frombuffer(base64.b64decode(pk['d']), dtype=np.int8).reshape(-1, pk['n']) / 60.0
                        n = min(len(rows), m.cap)
                        m.X[:n], m.y[:n], m.n = rows[:n, :-1], rows[:n, -1], n
                        loaded_mem = True
                    except Exception:
                        pass
        mem_path = (path or '') + '.memory.npz'
        if path and not loaded_mem and os.path.exists(mem_path):
            try:
                z = np.load(mem_path)
                for name in ('dPos', 'dNeg', 'bPos', 'bNeg', 'sPos', 'sNeg'):
                    m = getattr(self, name)
                    X, y = z[name + '_X'], z[name + '_y']
                    if X.ndim == 2 and X.shape[1] == m.X.shape[1]:
                        n = min(len(X), m.cap)
                        m.X[:n], m.y[:n], m.n = X[:n], y[:n], n
            except Exception:
                pass
        self.stats = (d and d.get('stats')) or [dict(tries=0, wins=0) for _ in STYLES]
        self.settings = (d and d.get('settings')) or None

    def danger_ready(self):
        return self.dPos.n >= 30 and self.dNeg.n >= 100

    def boop_ready(self):
        return self.bPos.n >= 3 and self.bNeg.n >= 30

    def style_ready(self):
        return self.sPos.n >= 3 and self.sNeg.n >= 20

    def add_packed(self, payload, copies=2):
        """Add experiences sent by the browser bot (userscript format). Real-game data counts double."""
        added = 0
        for name in ('dPos', 'dNeg', 'bPos', 'bNeg', 'sPos', 'sNeg'):
            pk = payload.get(name)
            m = getattr(self, name)
            if not (isinstance(pk, dict) and pk.get('n') == m.X.shape[1] + 1 and pk.get('d')):
                continue
            rows = np.frombuffer(base64.b64decode(pk['d']), dtype=np.int8).reshape(-1, pk['n']) / 60.0
            for r in rows:
                for _ in range(copies):
                    m.add(r[:-1], r[-1])
                added += 1
        self.real_added = getattr(self, 'real_added', 0) + added
        return added

    def nets_json(self):
        with self.lock:
            return json.dumps(dict(source='slither_bots.py', danger=self.danger.save(), boop=self.boop.save(),
                                   style=self.style.save(), stats=self.stats, real=getattr(self, 'real_added', 0)))

    def practice(self):
        with self.lock:
            self._practice()

    def _practice(self):
        for name, net, pos, neg in (('danger', self.danger, self.dPos, self.dNeg),
                                    ('boop', self.boop, self.bPos, self.bNeg),
                                    ('style', self.style, self.sPos, self.sNeg)):
            if pos.n and neg.n:
                xp, yp = pos.sample(32)
                xn, yn = neg.sample(32)
                l = net.train(np.vstack([xp, xn]), np.concatenate([yp, yn]), 0.02)
                self.loss[name] = self.loss[name] * 0.98 + 0.02 * l

    def save(self, best_settings=None):
        """One file for everything, in the userscript's format: the simulator loads it next run,
        and the browser bot imports the very same file (press O in slither.io)."""
        if not self.path:
            return
        self.export_js(self.path, best_settings)

    def export_js(self, out, settings=None):
        """Write the brain in the userscript's own format (press O in the game to import it)."""
        with self.lock:
            self._export_js(out, settings)

    def _export_js(self, out, settings=None):
        def pack(m):
            if m.n == 0:
                return dict(n=0, d='')
            rows = np.hstack([m.X[:m.n], m.y[:m.n, None]])
            q = np.clip(np.round(rows * 60), -127, 127).astype(np.int8)
            return dict(n=rows.shape[1], d=base64.b64encode(q.tobytes()).decode('ascii'))
        d = dict(source='slither_bots.py', danger=self.danger.save(), boop=self.boop.save(), style=self.style.save(),
                 dPos=pack(self.dPos), dNeg=pack(self.dNeg), bPos=pack(self.bPos), bNeg=pack(self.bNeg),
                 sPos=pack(self.sPos), sNeg=pack(self.sNeg), stats=self.stats,
                 settings={k: float(v) for k, v in (settings or {}).items() if k in TUNE})
        with open(out, 'w') as f:
            json.dump(d, f)


# ----------------------------------------------------------------------------
# geometry helpers (vectorised with numpy)
# ----------------------------------------------------------------------------
RAY_T = np.array([12, 24] + [40 * k for k in range(1, 11)], dtype=float)
RAY_FALL = np.maximum(0, RAY_T - 60) * 0.35
NEAR_MASK = RAY_T <= 60


def seg_dist(px, py, segs):
    """distance from points (any shape) to every segment -> shape + (N,)"""
    ax, ay, bx, by = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((px[..., None] - ax) * dx + (py[..., None] - ay) * dy) / L2, 0, 1)
    qx = ax + dx * t - px[..., None]
    qy = ay + dy * t - py[..., None]
    return np.sqrt(qx * qx + qy * qy)


def clearance(hx, hy, angles, obs, my_r):
    """room along each direction: (full, near) like rayClear2 in the userscript"""
    angles = np.asarray(angles, dtype=float)
    px = hx + np.cos(angles)[:, None] * RAY_T[None, :]
    py = hy + np.sin(angles)[:, None] * RAY_T[None, :]
    d = np.full(px.shape, 250.0)
    if len(obs['segs']):
        d = np.minimum(d, (seg_dist(px, py, obs['segs']) - obs['segs'][:, 4]).min(-1))
    if len(obs['circ']):
        c = obs['circ']
        dc = np.sqrt((px[..., None] - c[:, 0]) ** 2 + (py[..., None] - c[:, 1]) ** 2) - c[:, 2]
        d = np.minimum(d, dc.min(-1))
    wall = MAP_R * 0.98 - np.sqrt((px - MAP_R) ** 2 + (py - MAP_R) ** 2)
    d = np.minimum(d, wall)
    clr = d - my_r + RAY_FALL[None, :]
    return clr.min(1), clr[:, NEAR_MASK].min(1)


# ----------------------------------------------------------------------------
# the bot (same logic as the userscript)
# ----------------------------------------------------------------------------
class Bot:
    def __init__(self, world, brain, name, seed=0):
        self.world, self.brain, self.name = world, brain, name
        self.rng = random.Random(seed)
        self.cfg = dict(CFG)
        self.best = {k: self.cfg[k] for k in TUNE}
        if brain.settings:
            self.best.update({k: v for k, v in brain.settings.items() if k in TUNE})
        self.cfg.update(self.best)
        self.best_score, self.sigma, self.gen = None, 0.15, 0
        self.trial = None
        self.boops = self.deaths = 0
        self.kill_ratings = []
        self.last_kill = None
        self.last_style_name = None
        self.last_dn = 0
        self.target_since = 0
        self.max_score = 0
        self.snake = None
        self.track = {}
        self.respawn()

    # ---------------- lifecycle ----------------
    def respawn(self):
        x, y = self.world.rand_point(15000)
        self.snake = Snake(x, y, self.rng.random() * TAU, score=10, name=self.name, is_bot=True)
        self.world.snakes.append(self.snake)
        self.target = None; self.cut_side = 0; self.hold_until = 0; self.hold_start = 0
        self.hold_angle = 0; self.hook_ang = 0; self.orbit_dir = 1; self.style = None
        self.shadow_start = 0; self.skip = {}; self.feast = None
        self.boost_on_since = -999; self.boost_log = []; self.plan = ''; self.mode = ''
        self.d_pending, self.b_pending, self.s_pending = [], [], []
        self.start_trial()

    def ms(self):
        return self.world.tick_no * TICK_MS

    def start_trial(self):
        self.trial_start = self.ms(); self.trial_boops = self.boops; self.trial_s0 = self.snake.score; self.trial_max = self.snake.score
        if self.best_score is None or self.rng.random() < 0.34:
            self.trial = dict(self.best); self.trial_kind = 're-test best'
        else:
            t = dict(self.best)
            for k in self.rng.sample(list(TUNE), 2):
                lo, hi = TUNE[k]
                t[k] = clamp(t[k] + self.rng.gauss(0, 1) * self.sigma * (hi - lo), lo, hi)
                if hi - lo > 5:
                    t[k] = round(t[k])
            self.trial = t; self.trial_kind = 'trying'
        self.cfg.update(self.trial)

    def end_trial(self, died):
        secs = (self.ms() - self.trial_start) / 1000
        if secs < 8 and not died:
            return
        gain = max(0, self.trial_max - self.trial_s0)
        rate = 45 / max(secs, 15)
        n = self.boops - self.trial_boops
        kill_points = sum(self.kill_ratings[-n:]) if n > 0 else 0
        score = (kill_points * 1.2 + gain * 0.05) * rate + (-30 if died else 10)  # a rating-50 kill = 60 points
        self.gen += 1
        if self.best_score is None:
            self.best_score = score
        elif self.trial_kind == 're-test best':
            self.best_score = self.best_score * 0.7 + score * 0.3
        elif score > self.best_score:
            self.best = dict(self.trial)
            self.best_score = self.best_score * 0.5 + score * 0.5
            self.sigma = min(0.3, self.sigma * 1.2)
        else:
            self.sigma = max(0.04, self.sigma * 0.95)
        self.cfg.update(self.best)

    def on_death(self):
        self.deaths += 1
        now = self.ms()
        for t, x in self.d_pending:
            self.brain.dPos.add(x, 1.0 if now - t < 700 else 0.65)
        for t, s, k, x in self.s_pending:
            self.brain.sNeg.add(x, 0)
        self.end_trial(True)

    def rate_kill(self, victim):
        """Rate a kill 1-100: size, bigger than us, planned or lucky, style, boosting victim,
        crowding (clutch) and speed. Same rules as rateKill() in the userscript."""
        me = self.snake
        v_len = max(10, victim.score)
        my_len = max(10, me.score)
        why = []
        r = 30 * min(1, math.log10(v_len) / 4)
        ratio = v_len / my_len
        r += 15 * clamp(math.log2(ratio) + 1, 0, 2) / 2
        if ratio >= 1.5:
            why.append('bigger than us')
        planned = victim is self.target
        r += 15 if planned else 3
        why.append('planned' if planned else 'lucky')
        if planned and self.last_style_name in STYLE_POINTS:
            r += STYLE_POINTS[self.last_style_name]
            why.append(self.last_style_name)
        if self.is_boosting(victim):
            r += 8
            why.append('they were boosting')
        r += min(10, self.last_dn / 2)
        if self.last_dn >= 8:
            why.append('clutch')
        hunt = (self.ms() - self.target_since) / 1000 if planned else 99
        r += 10 if hunt < 3 else 5 if hunt < 6 else 0
        if hunt < 3:
            why.append('fast')
        rating = int(round(clamp(r, 1, 100)))
        self.kill_ratings.append(rating)
        self.last_kill = (rating, int(v_len), ', '.join(why))
        return rating

    def on_kill(self, victim):
        self.boops += 1
        rating = self.rate_kill(victim)
        self.feast = (victim.x, victim.y, self.ms() + self.cfg['feastMs'])
        keep = []
        for t, s, x in self.b_pending:
            if s is victim:
                self.brain.bPos.add(x, 1)
            else:
                keep.append((t, s, x))
        self.b_pending = keep
        last_k, keep = -1, []
        for t, s, k, x in self.s_pending:
            if s is victim:
                self.brain.sPos.add(x, 1); last_k = k
            else:
                keep.append((t, s, k, x))
        self.s_pending = keep
        if last_k >= 0:
            self.brain.stats[last_k]['wins'] += 1
        return rating

    # ---------------- perception ----------------
    def update_tracking(self, snakes):
        for s in snakes:
            k = self.track.get(s.id)
            if k is None:
                self.track[s.id] = dict(x=s.x, y=s.y, a=s.ang, v=NORMAL_SPEED / TICK_MS, w=0.0)
                continue
            v = math.hypot(s.x - k['x'], s.y - k['y']) / TICK_MS
            if v < 2:
                k['v'] = k['v'] * 0.7 + v * 0.3
            k['w'] = k['w'] * 0.7 + ang_diff(s.ang, k['a']) / TICK_MS * 0.3
            k['x'], k['y'], k['a'] = s.x, s.y, s.ang

    def vel(self, s):
        return self.track.get(s.id, {}).get('v', NORMAL_SPEED / TICK_MS)

    def omega(self, s):
        return clamp(self.track.get(s.id, {}).get('w', 0.0), -0.008, 0.008)

    def speed_ratio(self, s):
        return clamp(self.vel(s) / (NORMAL_SPEED / TICK_MS), 0.5, 3)

    def is_boosting(self, s):
        return self.speed_ratio(s) > 1.4

    def obstacles(self, me, enemies, target):
        hb = self.cfg['squeezeHitbox']
        R2 = self.cfg['scanRadius'] ** 2
        segs, circ = [], []
        for s in enemies:
            r = s.width() / 2 * hb
            sp = 5.7 * self.speed_ratio(s)
            steps = 0 if s is target else 3
            for k in range(steps + 1):
                px = s.x + math.cos(s.ang) * sp * 12 * k
                py = s.y + math.sin(s.ang) * sp * 12 * k
                if (px - me.x) ** 2 + (py - me.y) ** 2 < R2:
                    facing = abs(ang_diff(math.atan2(me.y - s.y, me.x - s.x), s.ang)) < 0.6
                    circ.append((px, py, r + (40 if facing and s is not target else 20)))
            pts = s.pts + [(s.x, s.y)]
            for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
                if (ax - me.x) ** 2 + (ay - me.y) ** 2 < R2 or (bx - me.x) ** 2 + (by - me.y) ** 2 < R2:
                    segs.append((ax, ay, bx, by, r, s.id))
        return dict(segs=np.array(segs) if segs else np.zeros((0, 6)), circ=np.array(circ) if circ else np.zeros((0, 3)))

    def situation(self, me, enemies, obs):
        radar_f, _ = clearance(me.x, me.y, np.arange(16) / 16 * TAU, obs, me.width() / 2 * self.cfg['squeezeHitbox'])
        heads = sorted(((math.hypot(s.x - me.x, s.y - me.y), s) for s in enemies), key=lambda t: t[0])
        heads = [(d, s, math.atan2(s.y - me.y, s.x - me.x)) for d, s in heads if d < 1200][:3]
        dn = dm = 0
        if len(obs['segs']):
            dd = seg_dist(np.array([me.x]), np.array([me.y]), obs['segs'])[0]
            dn, dm = int((dd < 150).sum()), int((dd < 350).sum())
        wall = MAP_R * 0.98 - math.hypot(me.x - MAP_R, me.y - MAP_R)
        my_segs = [(ax, ay, bx, by, me.width() / 2, me.id) for (ax, ay), (bx, by) in zip(me.pts[:-1], me.pts[1:])]
        return dict(radar=radar_f, heads=heads, dn=dn, dm=dm, wall=wall, my_segs=np.array(my_segs) if my_segs else np.zeros((0, 6)),
                    my=self.speed_ratio(me), boosting=1 if me.boosting else 0, len=me.score)

    def head_feat(self, h, a, me):
        if h is None:
            return [1.5, 0, 0, 0, 0.33, 0]
        d, s, ang = h
        rel = ang - a
        facing = math.cos(ang_diff(math.atan2(me.y - s.y, me.x - s.x), s.ang))
        return [clamp(d / 600, 0, 2), math.cos(rel), math.sin(rel), facing, self.speed_ratio(s) / 3, 1 if self.is_boosting(s) else 0]

    def danger_features(self, me, a, full, near, sit):
        k = int(round(a / (TAU / 16)))
        out = [clamp(sit['radar'][(k + i) % 16] / 200, -2, 2) for i in range(16)]
        out += [clamp(near / 100, -2, 2), clamp(full / 200, -2, 2), abs(ang_diff(a, me.ang)) / math.pi]
        for i in range(3):
            out += [clamp(v, -2, 2) for v in self.head_feat(sit['heads'][i] if i < len(sit['heads']) else None, a, me)]
        out += [clamp(sit['dn'] / 20, 0, 2), clamp(sit['dm'] / 60, 0, 2), clamp(sit['my'] / 3, 0, 2), sit['boosting'],
                clamp(math.log(sit['len'] + 1) / 6, 0, 2), clamp(sit['wall'] / 2000, -2, 2)]
        return out

    def boop_features(self, me, t, obs, sit, snakes):
        fa = t.ang
        rx, ry = me.x - t.x, me.y - t.y
        along = rx * math.cos(fa) + ry * math.sin(fa)
        lat = -rx * math.sin(fa) + ry * math.cos(fa)
        da = ang_diff(fa, me.ang)
        segs = obs['segs']
        segs = segs[segs[:, 5] != t.id] if len(segs) else segs
        if len(sit['my_segs']):
            segs = np.vstack([segs, sit['my_segs']]) if len(segs) else sit['my_segs']
        dens, ring = 0, [2.0] * 8
        if len(segs):
            dd = seg_dist(np.array([t.x]), np.array([t.y]), segs)[0]
            close = segs[dd < 400]
            dens = int((dd < 200).sum())
            if len(close):
                angs = fa + np.arange(8) / 8 * TAU
                ts = np.arange(1, 7) * 50.0
                px = t.x + np.cos(angs)[:, None] * ts
                py = t.y + np.sin(angs)[:, None] * ts
                room = (seg_dist(px, py, close) - close[:, 4]).min(-1).min(-1)
                ring = [clamp(min(300, r) / 150, -2, 2) for r in room]
        other = min([math.hypot(s.x - t.x, s.y - t.y) for s in snakes if s.alive and s is not t and s is not me] or [1500])
        x = [clamp(math.hypot(rx, ry) / 1500, -2, 2), clamp(along / 500, -2, 2), clamp(lat / 300, -2, 2),
             math.cos(da), math.sin(da), self.speed_ratio(t) / 3, self.speed_ratio(me) / 3,
             clamp(t.score / max(1, me.score), 0, 4) / 2, clamp(self.omega(t) * 100, -1, 1),
             clamp(dens / 20, 0, 2), 1, 1 if self.is_boosting(t) else 0]
        return x + ring + [clamp(min(1500, other) / 600, -2, 2)]

    # ---------------- safety ----------------
    def safe_angle(self, me, desired, obs, need, sit):
        head = me.ang
        N = self.cfg['angleSamples']
        step = TAU / N
        half = N // 2 - 1
        offs = np.arange(-half, half + 1)
        angs = head + offs * step
        my_r = me.width() / 2 * self.cfg['squeezeHitbox']
        full, near = clearance(me.x, me.y, np.append(angs, desired), obs, my_r)
        d_full, d_near = full[-1], near[-1]
        full, near = full[:-1], near[:-1]
        use_net = self.brain.danger_ready()
        risk = np.zeros(len(angs))
        r_des = 0.0
        if use_net:
            feats = [self.danger_features(me, a, f, n, sit) for a, f, n in zip(angs, full, near)]
            feats.append(self.danger_features(me, desired, d_full, d_near, sit))
            p = self.brain.danger.predict(feats)
            risk, r_des = p[:-1], p[-1]
        # sweep: room on the way from our heading to each offset
        mid = half
        sweep = np.empty(len(angs))
        m = math.inf
        for i in range(mid, len(angs)):
            m = min(m, near[i]); sweep[i] = m
        m = math.inf
        for i in range(mid, -1, -1):
            m = min(m, near[i]); sweep[i] = m
        d_off = ang_diff(desired, head)
        if abs(d_off) <= half * step:
            o = int(round(d_off / step)) + mid
            if d_full > need and sweep[o] > need * 0.5 and r_des < self.cfg['riskMax']:
                return desired, True, d_full
        c = np.minimum(full, sweep)
        ok = (c > need) & (risk < self.cfg['riskMax'])
        if ok.any():
            cost = np.abs([ang_diff(a, desired) for a in angs]) + 0.3 * np.abs(offs * step) + risk * 1.5
            cost[~ok] = np.inf
            i = int(np.argmin(cost))
            behind_only = abs(d_off) > half * step and d_full > need
            return angs[i], behind_only, c[i]
        i = int(np.argmax(c - risk * 80))
        return angs[i], False, c[i]

    # ---------------- targeting / booping ----------------
    def pick_target(self, me, enemies, obs, sit, snakes):
        best, best_score = None, math.inf
        now = self.ms()
        for s in enemies:
            if self.skip.get(s.id, 0) > now:
                continue
            d2 = (s.x - me.x) ** 2 + (s.y - me.y) ** 2
            if d2 > self.cfg['huntRadius'] ** 2:
                continue
            score = math.sqrt(d2) + 220 * abs(ang_diff(s.ang, me.ang)) / math.pi
            if self.is_boosting(s):
                score += 300
            if s is self.target:
                score -= self.cfg['stickiness']
            ratio = s.score / max(1, me.score)
            if ratio > 1:
                score -= self.cfg['bigTargetBonus'] * min(3, math.log2(ratio))
            if d2 < 450 ** 2 and math.cos(ang_diff(math.atan2(me.y - s.y, me.x - s.x), s.ang)) > 0.8:
                score -= self.cfg['baitBonus']
            if self.brain.boop_ready() and d2 < 900 ** 2:
                score -= self.cfg['boopNetWeight'] * float(self.brain.boop.predict([self.boop_features(me, s, obs, sit, snakes)])[0])
            if score < best_score:
                best_score, best = score, s
        return best

    def choose_style(self, me, t, obs, sit, snakes):
        base = self.boop_features(me, t, obs, sit, snakes)
        ratio = me.score / max(1, t.score)
        allowed = [i for i, y in enumerate(STYLES) if ratio >= y.get('minRatio', 0)]
        k = self.rng.choice(allowed)
        if self.brain.style_ready() and self.rng.random() > self.cfg['styleExplore']:
            X = [base + [1 if j == i else 0 for j in range(len(STYLES))] for i in allowed]
            k = allowed[int(np.argmax(self.brain.style.predict(X)))]
        self.brain.stats[k]['tries'] += 1
        self.last_style_name = STYLES[k]['name']
        self.s_pending.append((self.ms(), t, k, base + [1 if j == k else 0 for j in range(len(STYLES))]))
        return k

    def boop_plan(self, me, t, obs, sit, snakes):
        now = self.ms()
        cfg = self.cfg
        fa = t.ang
        fx, fy = math.cos(fa), math.sin(fa)
        nx, ny = -fy, fx
        rx, ry = me.x - t.x, me.y - t.y
        along = rx * fx + ry * fy
        lat = rx * nx + ry * ny
        side = 1 if lat >= 0 else -1
        tw = t.width()
        lead = cfg['leadBase'] + 5.7 * 12 * self.speed_ratio(t) + tw

        if now < self.hold_until:
            sty = STYLES[self.style if self.style is not None else 0]
            angle = self.hold_angle
            if sty.get('circle'):
                R = tw * 2 + 60
                d = math.hypot(me.x - t.x, me.y - t.y)
                tangent = math.atan2(me.y - t.y, me.x - t.x) + self.orbit_dir * math.pi / 2
                angle = tangent + self.orbit_dir * clamp((d - R) / R, -0.5, 0.8)
                self.plan = 'ENCIRCLE'
            elif sty.get('wrap'):
                f = clamp((now - self.hold_start) / max(1, self.hold_until - self.hold_start) / 0.7, 0, 1)
                angle = self.hold_angle + ang_diff(self.hook_ang, self.hold_angle) * f
                self.plan = 'WRAP'
            else:
                self.plan = 'HOLD'
            return dict(mode='HOLD', angle=angle, boost=False)
        if self.hold_until:
            self.hold_until = 0; self.style = None

        shadow_for = now - self.shadow_start if self.shadow_start else 0
        forced = shadow_for > cfg['shadowForceMs'] and along > -tw * 0.5
        if (along > cfg['cutAheadMargin'] + tw or forced or self.cut_side != 0) and abs(lat) < cfg['cutLateralMax'] and along > -tw:
            if self.style is None:
                self.style = self.choose_style(me, t, obs, sit, snakes)
            sty = STYLES[self.style]
            deep = sty.get('deep', 0)
            if self.cut_side == 0 and deep and not forced and along < cfg['cutAheadMargin'] + tw + deep:
                ax = t.x + fx * (lead + deep + 40) + nx * side * cfg['sideOffset'] * 0.8
                ay = t.y + fy * (lead + deep + 40) + ny * side * cfg['sideOffset'] * 0.8
                self.plan = 'DEEP CUT sprint'
                return dict(mode='SHADOW', angle=math.atan2(ay - me.y, ax - me.x), boost=True)
            self.shadow_start = 0
            if self.cut_side == 0:
                self.cut_side = side
            if side != self.cut_side:
                self.hold_angle = math.atan2(-ny * self.cut_side, -nx * self.cut_side)
                self.hook_ang = math.atan2(-fy, -fx)
                self.orbit_dir = 1 if ang_diff(me.ang, math.atan2(me.y - t.y, me.x - t.x)) >= 0 else -1
                self.hold_start = now
                self.hold_until = now + cfg['holdMs'] * sty['hold']
                self.cut_side = 0
                return dict(mode='HOLD', angle=self.hold_angle, boost=False)
            ax = t.x + fx * lead * 0.6 * sty['lead'] - nx * self.cut_side * cfg['cutOvershoot'] * sty['over']
            ay = t.y + fy * lead * 0.6 * sty['lead'] - ny * self.cut_side * cfg['cutOvershoot'] * sty['over']
            self.plan = sty['name']
            return dict(mode='CUT', angle=math.atan2(ay - me.y, ax - me.x), boost=True)

        self.cut_side = 0
        self.style = None
        if not self.shadow_start:
            self.shadow_start = now
        if now - self.shadow_start > cfg['shadowGiveUpMs']:
            self.skip[t.id] = now + 3000
            self.shadow_start = 0
        behind = max(0, -along)
        ax = t.x + fx * (lead + behind * 0.3) + nx * side * cfg['sideOffset']
        ay = t.y + fy * (lead + behind * 0.3) + ny * side * cfg['sideOffset']
        aim = math.atan2(ay - me.y, ax - me.x)
        pointed = abs(ang_diff(me.ang, aim)) < 0.9
        self.plan = 'SHADOW'
        return dict(mode='SHADOW', angle=aim, boost=pointed and along < cfg['cutAheadMargin'] + tw + 40 and math.hypot(ax - me.x, ay - me.y) > 30)

    # ---------------- food ----------------
    def big_food(self, me, food):
        # the heavy search runs every 4th decision; in between keep heading for the same spot
        if self.world.tick_no % 4 != 0 and hasattr(self, '_big'):
            b = self._big
            if b is None:
                return None
            d = math.hypot(b[0] - me.x, b[1] - me.y)
            if d < 40:
                self._big = None
                return None
            return dict(mode='BIG FOOD', angle=math.atan2(b[1] - me.y, b[0] - me.x), boost=d > 120, food=True)
        self._big = None
        if not len(food):
            return None
        d2 = (food[:, 0] - me.x) ** 2 + (food[:, 1] - me.y) ** 2
        near = food[d2 < self.cfg['bigFoodRadius'] ** 2]
        if len(near) < 5:
            return None
        cx = np.floor(near[:, 0] / 60).astype(np.int64)
        cy = np.floor(near[:, 1] / 60).astype(np.int64)
        m = near[:, 2] ** 2
        keys, inv = np.unique(cx * 100003 + cy, return_inverse=True)
        sm = np.bincount(inv, weights=m)
        sx = np.bincount(inv, weights=near[:, 0] * m)
        sy = np.bincount(inv, weights=near[:, 1] * m)
        cells = {(int(k // 100003), int(k % 100003)): (sm[i], sx[i], sy[i]) for i, k in enumerate(keys)}
        cands = []
        for (x, y), _ in cells.items():
            tm = tx = ty = 0.0
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    o = cells.get((x + i, y + j))
                    if o:
                        tm += o[0]; tx += o[1]; ty += o[2]
            cands.append((tm, tx / tm, ty / tm))
        piles = sorted(c[0] for c in cands)
        need = max(self.cfg['bigFoodMass'], piles[len(piles) // 2] * self.cfg['bigFoodVsNormal'])
        best, best_s = None, -1
        for pm, px, py in cands:
            if pm < need:
                continue
            d = math.hypot(px - me.x, py - me.y)
            sc = pm / (d + 80) ** 1.4
            if sc > best_s:
                best_s, best = sc, (px, py, d)
        if not best:
            return None
        self._big = best
        return dict(mode='BIG FOOD', angle=math.atan2(best[1] - me.y, best[0] - me.x), boost=best[2] > 120, food=True)

    def feed(self, me, food, pull):
        to_c = math.atan2(MAP_R - me.y, MAP_R - me.x)
        outness = clamp(math.hypot(me.x - MAP_R, me.y - MAP_R) / (MAP_R * 0.5), 0, 1.5)
        if len(food):
            d2 = (food[:, 0] - me.x) ** 2 + (food[:, 1] - me.y) ** 2
            sel = (d2 < self.cfg['foodRadius'] ** 2) & (d2 > 1)
            # food inside our turning circles can't be reached: chasing it just spins us in place
            turn_r = max(45.0, me.width() * 1.3)  # same as the userscript
            for side in (-1, 1):
                ccx = me.x + math.cos(me.ang + side * math.pi / 2) * turn_r
                ccy = me.y + math.sin(me.ang + side * math.pi / 2) * turn_r
                sel &= (food[:, 0] - ccx) ** 2 + (food[:, 1] - ccy) ** 2 >= (turn_r * 0.9) ** 2
            if sel.any():
                f = food[sel]; d = np.sqrt(d2[sel])
                a = np.arctan2(f[:, 1] - me.y, f[:, 0] - me.x)
                turn = np.abs((a - me.ang + math.pi) % TAU - math.pi)
                toward = np.cos((a - to_c + math.pi) % TAU - math.pi)
                score = f[:, 2] * 40 / d - turn * 0.3 + toward * outness * self.cfg['centerPull'] * 0.1 * pull
                i = int(np.argmax(score))
                return dict(mode='FEED', angle=float(a[i]), boost=False)
        return dict(mode='FEED', angle=to_c, boost=False)

    # ---------------- one decision ----------------
    def think(self, world):
        me = self.snake
        now = self.ms()
        snakes = [s for s in world.snakes if s.alive]
        self.update_tracking(snakes)
        enemies = [s for s in snakes if s is not me and (s.x - me.x) ** 2 + (s.y - me.y) ** 2 < 3500 ** 2]
        if self.target is not None and not self.target.alive:
            self.target = None; self.cut_side = 0; self.hold_until = 0; self.style = None
        obs = self.obstacles(me, enemies, self.target)
        sit = self.situation(me, enemies, obs)
        target = self.pick_target(me, enemies, obs, sit, snakes)
        if target is not self.target:
            self.cut_side = 0; self.hold_until = 0; self.shadow_start = 0; self.style = None
            self.target_since = now
        self.last_dn = sit['dn']
        self.target = target

        from_c = math.hypot(me.x - MAP_R, me.y - MAP_R)
        in_center = from_c < self.cfg['centerZone']
        self.phase = 'FIGHT' if me.score >= self.cfg['fightLength'] and in_center else 'GROW'
        mid_cut = now < self.hold_until or self.cut_side != 0
        plan = None
        if self.feast and now < self.feast[2] and not mid_cut:
            fx, fy, _ = self.feast
            d = math.hypot(fx - me.x, fy - me.y)
            if 50 < d < 1500:
                plan = dict(mode='BIG FOOD', angle=math.atan2(fy - me.y, fx - me.x), boost=True, food=True)
                self.plan = 'eat victim'
        if plan is None and not mid_cut:
            plan = self.big_food(me, world.food)
            if plan:
                self.plan = 'big food'
        if plan is None and self.phase == 'GROW' and not mid_cut:
            self.target = None
            plan = self.feed(me, world.food, 1 if in_center else 4)
            self.plan = 'grow'
        if plan is None and self.target is not None:
            plan = self.boop_plan(me, self.target, obs, sit, snakes)
        if plan is None:
            plan = self.feed(me, world.food, 1 if in_center else 4)
            self.plan = 'eat'
        if plan.get('mode') != 'CUT':
            pass
        if self.target is not None:
            obs = self.obstacles(me, enemies, self.target)
        need = self.cfg['squeezeClearance'] * (0.5 if plan['mode'] == 'CUT' else 1)
        angle, safe, clr = self.safe_angle(me, plan['angle'], obs, need, sit)
        me.want_ang = angle

        # boosting: min length, food boosts when the route is clear, fight boost budget, no flicker
        self.boost_log = [b for b in self.boost_log if now - b < 5000]
        roomy = safe or (plan.get('food') and clr > need)
        budget_ok = plan.get('food') or len(self.boost_log) * TICK_MS < self.cfg['fightBoostBudget']
        want = me.score >= self.cfg['minBoostLength'] and plan.get('boost') and budget_ok and (roomy or plan['mode'] == 'CUT')
        if want != me.want_boost and now - self.boost_on_since > (300 if me.want_boost else 250):
            me.want_boost = want
            self.boost_on_since = now
        if me.want_boost and not plan.get('food'):
            self.boost_log.append(now)
        self.mode = plan['mode'] + ('' if safe else ' (evading)')

        # ---- record experiences for the networks ----
        if world.tick_no % 3 == 0:
            f, n = clearance(me.x, me.y, [angle], obs, me.width() / 2 * self.cfg['squeezeHitbox'])
            self.d_pending.append((now, self.danger_features(me, angle, f[0], n[0], sit)))
        while self.d_pending and now - self.d_pending[0][0] > 1500:
            self.brain.dNeg.add(self.d_pending.pop(0)[1], 0)
        if self.target is not None and world.tick_no % 15 == 0:
            self.b_pending.append((now, self.target, self.boop_features(me, self.target, obs, sit, snakes)))
        while self.b_pending and now - self.b_pending[0][0] > 4000:
            self.brain.bNeg.add(self.b_pending.pop(0)[2], 0)
        while self.s_pending and now - self.s_pending[0][0] > 5000:
            self.brain.sNeg.add(self.s_pending.pop(0)[3], 0)

        # ---- settings experiment bookkeeping ----
        self.trial_max = max(self.trial_max, me.score)
        self.max_score = max(self.max_score, me.score)
        if now - self.trial_start > self.cfg['trialMs']:
            self.end_trial(False)
            self.start_trial()


# ----------------------------------------------------------------------------
# running it
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# live viewer (pygame)
# ----------------------------------------------------------------------------
def run_pygame(world, bots, brain, advance, kill_events, max_steps):
    import pygame
    pygame.init()
    W_, H_ = 1200, 820
    screen = pygame.display.set_mode((W_, H_), pygame.RESIZABLE)
    pygame.display.set_caption('Slither Booper - 3 bots learning')
    font = pygame.font.SysFont('consolas,menlo,monospace', 14)
    big = pygame.font.SysFont('consolas,menlo,monospace', 22, bold=True)
    clock = pygame.time.Clock()
    BOT_COL = [(255, 77, 77), (77, 210, 255), (255, 210, 77)]
    NPC_COL = [(150, 150, 150), (120, 160, 120), (160, 130, 170), (170, 150, 110)]
    BG, GRID, FOOD = (17, 17, 22), (32, 32, 40), (110, 200, 110)
    zoom, follow, speed, paused = 0.35, 0, 1, False
    show_food, show_aim = True, True
    speeds = [1, 2, 4, 8, 16, 32]
    cam = [bots[0].snake.x, bots[0].snake.y]
    steps_done = 0
    running = True

    def to_screen(x, y):
        return int((x - cam[0]) * zoom + W_ / 2), int((y - cam[1]) * zoom + H_ / 2)

    def text(t, x, y, col=(230, 230, 230), f=None):
        surf = (f or font).render(t, True, col)
        screen.blit(surf, (x, y))
        return surf.get_width()

    def focus_bot():
        if follow:
            return bots[follow - 1]
        alive = [b for b in bots if b.snake.alive]
        return max(alive or bots, key=lambda b: (b.target is not None, b.snake.score))

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.VIDEORESIZE:
                W_, H_ = max(400, e.w), max(300, e.h)
                screen = pygame.display.set_mode((W_, H_), pygame.RESIZABLE)
            elif e.type == pygame.MOUSEWHEEL:
                zoom = clamp(zoom * (1.15 ** e.y), 0.03, 2.0)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3):
                    follow = e.key - pygame.K_0
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    zoom = clamp(zoom * 1.25, 0.03, 2.0)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom = clamp(zoom / 1.25, 0.03, 2.0)
                elif e.key == pygame.K_UP:
                    speed = min(len(speeds) - 1, speed + 1)
                elif e.key == pygame.K_DOWN:
                    speed = max(0, speed - 1)
                elif e.key == pygame.K_f:
                    show_food = not show_food
                elif e.key == pygame.K_r:
                    show_aim = not show_aim

        if not paused:
            for _ in range(speeds[speed]):
                advance()
                steps_done += 1
            if max_steps and steps_done >= max_steps:
                running = False

        fb = focus_bot()
        # smooth camera
        cam[0] += (fb.snake.x - cam[0]) * 0.2
        cam[1] += (fb.snake.y - cam[1]) * 0.2
        view_w, view_h = W_ / zoom / 2, H_ / zoom / 2

        def on_screen(x, y, pad=200):
            return abs(x - cam[0]) < view_w + pad and abs(y - cam[1]) < view_h + pad

        screen.fill(BG)
        # grid
        g = 500
        gx0 = int((cam[0] - view_w) // g) * g
        gy0 = int((cam[1] - view_h) // g) * g
        x = gx0
        while x < cam[0] + view_w + g:
            sx, _ = to_screen(x, 0)
            pygame.draw.line(screen, GRID, (sx, 0), (sx, H_))
            x += g
        y = gy0
        while y < cam[1] + view_h + g:
            _, sy = to_screen(0, y)
            pygame.draw.line(screen, GRID, (0, sy), (W_, sy))
            y += g
        # map edge
        cx, cy = to_screen(MAP_R, MAP_R)
        pygame.draw.circle(screen, (120, 40, 40), (cx, cy), int(MAP_R * 0.98 * zoom), max(1, int(3 * zoom)))
        # food
        if show_food and len(world.food):
            f = world.food
            sel = (np.abs(f[:, 0] - cam[0]) < view_w) & (np.abs(f[:, 1] - cam[1]) < view_h)
            for fx, fy, fs in f[sel]:
                pygame.draw.circle(screen, FOOD, to_screen(fx, fy), max(1, int(fs * zoom * 0.8)))
        # snakes
        for s in world.snakes:
            if not s.alive or not on_screen(s.x, s.y, 3000):
                continue
            col = NPC_COL[s.id % len(NPC_COL)]
            bot = None
            for i, b in enumerate(bots):
                if b.snake is s:
                    col, bot = BOT_COL[i], b
            r = max(1, int(s.width() / 2 * zoom))
            pts = [to_screen(px, py) for (px, py) in s.pts] + [to_screen(s.x, s.y)]
            pts = [p for p in pts if -50 < p[0] < W_ + 50 and -50 < p[1] < H_ + 50] or pts[-1:]
            dark = tuple(int(c * 0.7) for c in col)
            if len(pts) > 1:
                pygame.draw.lines(screen, dark, False, pts, max(1, r * 2))
            for p in pts[::2]:
                pygame.draw.circle(screen, col, p, r)
            hx, hy = to_screen(s.x, s.y)
            pygame.draw.circle(screen, col, (hx, hy), r + 1)
            # eyes
            for side in (-0.5, 0.5):
                ex = hx + int(math.cos(s.ang + side) * r * 0.6)
                ey = hy + int(math.sin(s.ang + side) * r * 0.6)
                pygame.draw.circle(screen, (255, 255, 255), (ex, ey), max(1, r // 3))
            if s.boosting:
                pygame.draw.circle(screen, (255, 255, 255), (hx, hy), r + 4, 1)
            if bot:
                text('%s  %d  %s' % (bot.name, s.score, bot.plan or bot.mode), hx + r + 6, hy - r - 18, col)
        # aim lines: a hunting bot -> its target
        if show_aim:
            for i, b in enumerate(bots):
                t = b.target
                if b.snake.alive and t is not None and t.alive:
                    a, c = to_screen(b.snake.x, b.snake.y), to_screen(t.x, t.y)
                    pygame.draw.line(screen, BOT_COL[i], a, c, 1)
                    pygame.draw.circle(screen, BOT_COL[i], c, max(6, int(t.width() * zoom)), 2)
        # kill popups: rating floats up where the victim died
        now_tick = world.tick_no
        for (tk, bi, rating, kx, ky) in list(kill_events):
            age = now_tick - tk
            if age > 90:
                kill_events.remove((tk, bi, rating, kx, ky))
                continue
            px, py = to_screen(kx, ky)
            text('KILL %d/100' % rating, px - 50, py - 30 - age, BOT_COL[bi], big)

        # HUD
        pygame.draw.rect(screen, (0, 0, 0), (8, 8, 780, 158))
        mins = world.tick_no * TICK_MS / 60000
        text('%.1f game-min   speed x%d%s   following %s   zoom %.2f' % (
            mins, speeds[speed], '  PAUSED' if paused else '', fb.name if follow else 'action (' + fb.name + ')', zoom), 16, 14)
        for i, b in enumerate(bots):
            kr = b.kill_ratings
            text('%s  length %5d  %-5s  boops %3d  deaths %3d  avg kill %s  best %s  %s' % (
                b.name, b.snake.score, getattr(b, 'phase', '-'), b.boops, b.deaths,
                round(sum(kr) / len(kr)) if kr else '-', max(kr) if kr else '-', b.mode), 16, 36 + i * 20, BOT_COL[i])
        text('brain  danger %s  boop %s  style %s   (deaths seen %d, boops seen %d)' % (
            'ON' if brain.danger_ready() else 'learning', 'ON' if brain.boop_ready() else 'learning',
            'ON' if brain.style_ready() else 'learning', brain.dPos.n, brain.bPos.n), 16, 100)
        text('styles  ' + '  '.join('%s %d/%d' % (y['name'], st_['wins'], st_['tries']) for y, st_ in zip(STYLES, brain.stats)), 16, 120)
        text('1/2/3 follow  0 auto  wheel zoom  up/down speed  space pause  F food  R aim  Esc quit', 16, 140, (140, 140, 140))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


WORLD_NPCS = [12]


def draw_kill(path, world, killer_bot, victim, bots):
    """Save a picture of one kill: the bot that did it and the victim (purple, with an X)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rating, v_len, why = killer_bot.last_kill
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#111'); ax.set_facecolor('#111')
    k = killer_bot.snake
    cx, cy = (k.x + victim.x) / 2, (k.y + victim.y) / 2
    V = max(700, math.hypot(k.x - victim.x, k.y - victim.y) * 0.8 + 300)
    ax.set_xlim(cx - V, cx + V); ax.set_ylim(cy - V, cy + V); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ppu = (6 * 72 * 0.8) / (2 * V)
    if len(world.food):
        f = world.food
        f = f[(np.abs(f[:, 0] - cx) < V) & (np.abs(f[:, 1] - cy) < V)]
        ax.scatter(f[:, 0], f[:, 1], s=(f[:, 2] * ppu * 2) ** 2, c='#6c6', linewidths=0, alpha=0.6)
    colors = ['#ff4d4d', '#4dd2ff', '#ffd24d']
    for s in list(world.snakes) + [victim]:
        if len(s.pts) < 2 or (not s.alive and s is not victim):
            continue
        if abs(s.x - cx) > V * 2 or abs(s.y - cy) > V * 2:
            continue
        col, alpha = '#8a8a8a', 0.9
        for i, b in enumerate(bots):
            if b.snake is s:
                col = colors[i]
        if s is victim:
            col, alpha = '#c070ff', 0.6
        xs = [p[0] for p in s.pts] + [s.x]; ys = [p[1] for p in s.pts] + [s.y]
        ax.plot(xs, ys, color=col, linewidth=max(1.5, s.width() * ppu), solid_capstyle='round', alpha=alpha)
        ax.plot([s.x], [s.y], 'o', color='white', markersize=max(2, s.width() * ppu * 0.45))
    ax.plot([victim.x], [victim.y], 'x', color='white', markersize=16, mew=3)
    ax.text(victim.x, victim.y - victim.width() - 70, 'victim (length %d)' % v_len, color='#c070ff', ha='center', fontsize=9)
    ax.text(k.x, k.y + k.width() + 50, '%s (length %d)' % (killer_bot.name, k.score), color='white', ha='center', fontsize=9)
    ax.set_title('kill rating %d/100\n%s' % (rating, why), color='white', fontsize=11)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def run(steps, view=False, snapshot=None, brain_path='booper_brain.json', seed=None, report_every=1500,
        gif=None, gif_frames=300, gif_every=2, kill_photos=None, steps_given=True, sync=True):
    world = World(n_npc=WORLD_NPCS[0], seed=seed)
    brain = Brain(brain_path)
    bots = [Bot(world, brain, 'bot%d' % (i + 1), seed=(seed or 0) + i) for i in range(3)]
    respawn_at = {}
    t0 = time.time()

    if sync and brain_path:
        start_sync_server(brain)
    kill_events = []
    photos = {}
    if kill_photos:
        os.makedirs(kill_photos, exist_ok=True)

    def advance():
        world.step(bots)
        for killer, victim in world.kills:
            for b in bots:
                if killer is b.snake:
                    rating = b.on_kill(victim)
                    kill_events.append((world.tick_no, bots.index(b), rating, victim.x, victim.y))
                    del kill_events[:-20]
                    band = min(9, (rating - 1) // 10)  # 1-10, 11-20, ... 91-100
                    if kill_photos and band not in photos:
                        name = os.path.join(kill_photos, 'kill_%03d-%03d.png' % (band * 10 + 1, band * 10 + 10))
                        draw_kill(name, world, b, victim, bots)
                        photos[band] = name
                        print('photo: rating %d -> %s' % (rating, name))
                    if not view and not gif:
                        print('  * %s killed a length-%d snake: rating %d/100 (%s)' % (b.name, b.last_kill[1], b.last_kill[0], b.last_kill[2]))
                if victim is b.snake:
                    b.on_death()
                    respawn_at[b.name] = world.tick_no + 30
        for b in bots:
            if not b.snake.alive and respawn_at.get(b.name, 0) <= world.tick_no:
                world.snakes = [s for s in world.snakes if s is not b.snake]
                b.respawn()
        if world.tick_no % 4 == 0:
            brain.practice()
        while brain.inbox:  # real-game experiences from the browser bot
            payload = brain.inbox.pop(0)
            try:
                with brain.lock:
                    n = brain.add_packed(payload)
                print('  + %d real-game experiences from the browser bot' % n)
            except Exception as e:
                print('  ! could not read experiences from the browser:', e)
        if world.tick_no % report_every == 0:
            report()
        if world.tick_no % 20000 == 0:
            brain.save(bots[0].best)

    def report():
        mins = world.tick_no * TICK_MS / 60000
        print('--- %.1f game-minutes (%.0f steps/s) ---' % (mins, world.tick_no / max(1e-6, time.time() - t0)))
        for b in bots:
            print('  %s  length %5d  phase %-5s  boops %3d  deaths %3d  best length %5d  mode %s' % (
                b.name, b.snake.score, getattr(b, 'phase', '-'), b.boops, b.deaths, b.max_score, b.mode))
            if b.kill_ratings:
                print('        kill rating: last %d/100 (length %d: %s)  average %d  best %d' % (
                    b.last_kill[0], b.last_kill[1], b.last_kill[2], sum(b.kill_ratings) / len(b.kill_ratings), max(b.kill_ratings)))
        if getattr(brain, 'real_added', 0):
            print('  real-game experiences received from the browser bot: %d' % brain.real_added)
        print('  brain: danger %s (%d death / %d safe, err %.2f)  boop %s (%d/%d)  style %s (%d/%d)' % (
            'ACTIVE' if brain.danger_ready() else 'learning', brain.dPos.n, brain.dNeg.n, brain.loss['danger'],
            'ACTIVE' if brain.boop_ready() else 'learning', brain.bPos.n, brain.bNeg.n,
            'ACTIVE' if brain.style_ready() else 'learning', brain.sPos.n, brain.sNeg.n))
        print('  styles: ' + '  '.join('%s %d/%d' % (y['name'], s['wins'], s['tries']) for y, s in zip(STYLES, brain.stats)))

    use_pygame = False
    if view and not gif and not snapshot:
        try:
            import pygame  # noqa: F401
            use_pygame = True
        except ImportError:
            print('pygame not installed (pip install pygame) - using the slower matplotlib viewer')
    if view and not use_pygame:
        print('Opening the matplotlib window instead...')
    if not view and not gif and not snapshot:
        print('Training WITHOUT a window (add --view to watch). Progress is printed below.')
    if use_pygame:
        print('Opening the pygame window... (if you can\'t see it, check behind other windows)')
        try:
            run_pygame(world, bots, brain, advance, kill_events, steps if steps_given else 0)
        finally:
            report()
            brain.save(bots[0].best)
            print('brain saved to', brain_path)
        return
    if view or snapshot or gif:
        import matplotlib
        if not view:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor('#111')
        colors = ['#ff4d4d', '#4dd2ff', '#ffd24d']
        VIEW = 1600  # half-width of the view, in world units

        def draw():
            ax.clear()
            ax.set_facecolor('#111')
            # follow the most interesting bot: one that is hunting, else the longest
            alive = [b for b in bots if b.snake.alive]
            focus_bot = max(alive or bots, key=lambda b: (b.target is not None, b.snake.score))
            focus = focus_bot.snake
            ax.set_xlim(focus.x - VIEW, focus.x + VIEW)
            ax.set_ylim(focus.y - VIEW, focus.y + VIEW)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            pts_per_unit = (fig.get_figwidth() * 72 * 0.8) / (2 * VIEW)
            if len(world.food):
                f = world.food
                sel = (np.abs(f[:, 0] - focus.x) < VIEW) & (np.abs(f[:, 1] - focus.y) < VIEW)
                f = f[sel]
                ax.scatter(f[:, 0], f[:, 1], s=(f[:, 2] * pts_per_unit * 2) ** 2, c='#6c6', linewidths=0, alpha=0.8)
            for s in world.snakes:
                if not s.alive or len(s.pts) < 2:
                    continue
                if abs(s.x - focus.x) > VIEW * 3 or abs(s.y - focus.y) > VIEW * 3:
                    continue
                col, label = '#8a8a8a', None
                for i, b in enumerate(bots):
                    if b.snake is s:
                        col, label = colors[i], '%s  %s' % (b.name, b.plan or b.mode)
                xs = [p[0] for p in s.pts] + [s.x]
                ys = [p[1] for p in s.pts] + [s.y]
                ax.plot(xs, ys, color=col, linewidth=max(1.5, s.width() * pts_per_unit), solid_capstyle='round', alpha=0.95)
                ax.plot([s.x], [s.y], 'o', color='white', markersize=max(2, s.width() * pts_per_unit * 0.45))
                if label:
                    ax.text(s.x, s.y + s.width() + 40, label, color=col, fontsize=8, ha='center')
            # dashed line from a hunting bot to its target
            for i, b in enumerate(bots):
                if b.snake.alive and b.target is not None and b.target.alive:
                    ax.plot([b.snake.x, b.target.x], [b.snake.y, b.target.y], '--', color=colors[i], linewidth=0.8, alpha=0.6)
            mins = world.tick_no * TICK_MS / 60000
            title = '%.1f game-min   ' % mins + '   '.join('%s: %d long, %d boops%s' % (
                b.name, b.snake.score, b.boops, (' (last %d/100)' % b.last_kill[0]) if b.last_kill else '') for b in bots)
            ax.set_title(title, fontsize=8, color='white')

        if gif:
            from matplotlib.animation import FuncAnimation, PillowWriter
            for _ in range(steps):
                advance()

            def frame(_):
                for _ in range(gif_every):
                    advance()
                draw()
            anim = FuncAnimation(fig, frame, frames=gif_frames, interval=1000 / 15)
            anim.save(gif, writer=PillowWriter(fps=15))
            print('saved', gif)
        elif snapshot and not view:
            for _ in range(steps):
                advance()
            draw()
            fig.savefig(snapshot, dpi=110, facecolor=fig.get_facecolor())
            print('saved', snapshot)
        else:
            plt.ion()
            for i in range(steps):
                advance()
                if i % 3 == 0:
                    draw()
                    plt.pause(0.001)
                if not plt.fignum_exists(fig.number):
                    break
    else:
        for _ in range(steps):
            advance()
    report()
    brain.save(bots[0].best)
    print('brain saved to', brain_path)
    print('to use it in the browser bot: in slither.io press O and pick', os.path.abspath(brain_path))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='3 Slither Booper bots learning together')
    ap.add_argument('--steps', type=int, default=None, help='simulation steps (30 per game-second); with --view, runs until you close the window')
    ap.add_argument('--view', action='store_true', help='watch live in a pygame window (falls back to matplotlib)')
    ap.add_argument('--snapshot', help='save a picture after --steps steps')
    ap.add_argument('--brain', default='booper_brain.json', help='where the brain is saved/loaded')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--fight-length', type=int, default=None,
                    help='start fighting at this length (default 500, same as the userscript); lower it to practise booping sooner')
    ap.add_argument('--npcs', type=int, default=12, help='number of other snakes')
    ap.add_argument('--gif', help='train for --steps, then record an animated GIF of the bots playing')
    ap.add_argument('--gif-frames', type=int, default=300, help='frames in the GIF (15 per second)')
    ap.add_argument('--no-sync', action='store_true', help="don't share the brain with the Tampermonkey bot")
    ap.add_argument('--export', action='store_true',
                    help='just write booper_brain_for_tampermonkey.json from the saved brain, then stop')
    ap.add_argument('--kill-photos', help='folder: save one picture of a kill for each rating band (1-10 ... 91-100)')
    a = ap.parse_args()
    import sys
    if len(sys.argv) == 1:
        # started with no options (double-click or an editor's Run button): open the window
        a.view = True
        print('No options given, so opening the viewer window. (Run with --steps N to train without a window.)')
    if a.fight_length is not None:
        CFG['fightLength'] = a.fight_length
    WORLD_NPCS[0] = a.npcs
    if a.export:
        b = Brain(a.brain)
        out = export_path(a.brain)
        b.export_js(out, b.settings)
        print('wrote', out, '- in slither.io press O and pick this file (booper_brain.json works too)')
        raise SystemExit
    steps_given = a.steps is not None
    if not steps_given:
        a.steps = 30000
    run(a.steps, view=a.view, snapshot=a.snapshot, brain_path=a.brain, seed=a.seed, gif=a.gif, gif_frames=a.gif_frames, kill_photos=a.kill_photos, steps_given=steps_given, sync=not a.no_sync)
