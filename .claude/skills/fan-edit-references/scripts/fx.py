"""Camera language for the edit:
- smooth per-shot camera moves (push-in / pull-out with a little roll), like the reference edits
- drop shots enter with an eased zoom-out + roll settle ("zoom-out entry")
- whip-pan transitions on selected cuts
- shakes only on the musical accents
All of it goes through one Lanczos warp per frame, with true motion blur sampled along the path.
"""
import math
import numpy as np, cv2

FPS = 30
K = 1.0   # pixel scale vs the original 834-px-high frame (set by render.py)
DROP = 189


class Hit:
    def __init__(self, f, shake, zoom=0.0, rgb=0.0, flash=0.0, seed=0):
        self.f, self.shake, self.zoom, self.rgb, self.flash = f, shake, zoom, rgb, flash
        r = np.random.default_rng(seed + f * 7919)
        self.ang = r.uniform(0, 2 * math.pi)
        self.ph = r.uniform(0, 2 * math.pi)
        self.rot_sign = r.choice([-1, 1])


def build_hits():
    """shakes only on the musical accents"""
    return [
        Hit(26, shake=0.9, zoom=0.06, rgb=0.8, flash=0.25),    # B*TCHES
        Hit(81, shake=0.9, zoom=0.06, rgb=0.8, flash=0.25),    # HEAD
        Hit(132, shake=0.9, zoom=0.06, rgb=0.8, flash=0.25),   # WORKING
        Hit(172, shake=1.0, zoom=0.07, rgb=0.9, flash=0.3),    # MOTHERF*CKING
        Hit(189, shake=1.3, zoom=0.10, rgb=1.2, flash=0.08),   # the drop: gunshot (recoil); glow is local
        Hit(383, shake=0.8, zoom=0.07, rgb=0.8, flash=0.3),    # after the white flash
        Hit(403, shake=1.2, zoom=0.09, rgb=1.1, flash=0.08),   # second drop: gunshot
        Hit(598, shake=1.1, zoom=0.08, rgb=1.0, flash=0.4),    # final shot
    ]


TIGHT = {'B1'}  # already very close shots: keep the camera move small


def name_is_tight(timeline, a):
    return any(st == a and nm in TIGHT for st, _, nm in timeline)


def ease_out(x):  # cubic
    x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3


def ease_in(x):
    x = min(max(x, 0.0), 1.0); return x ** 3


def ease_io(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)


class FX:
    WHIP_LEN = 3  # frames on each side of the cut

    def __init__(self, W, H, hits, timeline, whips=(), no_entry=(), zooms=()):
        self.zooms = set(zooms)
        self.W, self.H, self.hits = W, H, hits
        self.timeline = timeline
        self.whips = {c: (1 if i % 2 == 0 else -1) for i, c in enumerate(sorted(whips))}
        self.no_entry = set(no_entry)
        # one camera move per base shot, alternating push-in / pull-out, alternating roll direction
        self.moves = {}
        for i, (a, b, name) in enumerate(timeline):
            self.moves[a] = ('in' if i % 2 == 0 else 'out', 1 if (i // 2) % 2 == 0 else -1)

    # ---- helpers -------------------------------------------------------------------------
    def base_shot(self, n):
        act = [(a, b) for a, b, _ in self.timeline if a <= n < b]
        if not act: return None
        return max(act)  # the latest-starting shot is on top

    def whip_dx(self, t, ref):
        """horizontal whip offset at fractional time t, evaluated on the side of the cut that frame ref is on"""
        for c, d in self.whips.items():
            L = self.WHIP_LEN
            if c - L <= ref < c:          # outgoing shot accelerates out
                return -d * self.W * 0.9 * ease_in((t - (c - L)) / L)
            if c <= ref < c + L:          # incoming shot decelerates in
                return d * self.W * 0.9 * (1 - ease_out((t - c + 1) / L))
        return 0.0

    def camera(self, t, ref):
        """smooth per-shot camera: returns dx, dy, rot, scale"""
        bs = self.base_shot(ref)
        if bs is None: return 0.0, 0.0, 0.0, 1.0
        a, b = bs
        kind, sgn = self.moves[a]
        u = (t - a) / max(1, b - a)
        drop = a >= DROP
        amp = 0.06 if drop else 0.05
        if name_is_tight(self.timeline, a): amp = 0.03
        if kind == 'in':
            sc = 1.0 + amp * ease_io(u); rot = sgn * 0.6 * ease_io(u)
        else:
            sc = 1.0 + amp * (1 - ease_io(u)); rot = sgn * 0.6 * (1 - ease_io(u))
        dx = sgn * 10 * K * (ease_io(u) - 0.5); dy = 0.0
        for c in self.zooms:               # zoom-through: rush in on the outgoing shot, land from close-up on the new one
            if c - 3 <= ref < c: sc *= 1 + 0.65 * ease_in((t - (c - 3)) / 3)
            elif c <= ref < c + 3: sc *= 1 + 0.55 * (1 - ease_out((t - c + 1) / 3))
        if drop and a not in self.no_entry and a not in self.whips and a not in self.zooms:
            e = 1 - ease_out((t - a) / 8)          # zoom-out entry over 8 frames
            sc += 0.16 * e; rot += -sgn * 2.2 * e
        return dx, dy, rot, sc

    def params(self, t, ref):
        """continuous transform at fractional time t, using the shot context of integer frame ref"""
        dx, dy, rot, sc = self.camera(t, ref)
        zoom = 0.0; rgb = 0.0; flash = 0.0; sdx = sdy = 0.0
        for h in self.hits:
            dt = (t - h.f) / FPS
            if dt < -0.5 / FPS or dt > 0.6: continue
            dt = max(dt, 0.0)               # crisp onset: blur only after the hit
            env = math.exp(-dt / 0.10)
            osc = math.cos(2 * math.pi * 8.5 * dt + h.ph)
            osc2 = math.sin(2 * math.pi * 6 * dt + h.ph * 1.7)
            a = 22 * K * h.shake * env
            sdx += a * (math.cos(h.ang) * osc - 0.45 * math.sin(h.ang) * osc2)
            sdy += a * (math.sin(h.ang) * osc + 0.45 * math.cos(h.ang) * osc2)
            rot += 1.7 * h.shake * env * h.rot_sign * math.cos(2 * math.pi * 6 * dt)
            zoom += h.zoom * math.exp(-dt / 0.13)
            rgb += h.rgb * math.exp(-dt / 0.08)
            flash += h.flash * math.exp(-dt / 0.06)
        dx += float(np.clip(sdx, -42 * K, 42 * K)); dy += float(np.clip(sdy, -34 * K, 34 * K))
        sc += min(zoom, 0.2)
        rot = float(np.clip(rot, -4, 4))
        # never show the frame edge: minimum scale for the current roll + shake
        need = 1 + abs(math.radians(rot)) * (self.W / self.H + 0.3) + 2 * max(abs(dx) / self.W, abs(dy) / self.H)
        sc = max(sc, need)
        dx += self.whip_dx(t, ref)       # whips are allowed to leave the frame (mirrored + heavily blurred)
        return dx, dy, rot, sc, min(rgb, 1.5), min(flash, 1.0)

    def _warp(self, img, dx, dy, rot, sc):
        M = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), rot, sc)
        M[0, 2] += dx; M[1, 2] += dy
        return cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)

    def apply(self, img, n):
        p = self.params(n, n)
        q = self.params(n - 0.4, n)       # shutter: 0.4 frame, same shot context
        speed = (abs(p[0] - q[0]) + abs(p[1] - q[1]) + 10 * abs(p[2] - q[2]) + 500 * abs(p[3] - q[3]))
        if speed > 2.5:                   # motion blur along the camera path
            k = int(min(18, 2 + speed / 7))
            acc = np.zeros(img.shape, np.float32)
            for i in range(k):
                a = i / (k - 1)
                acc += self._warp(img, *[q[j] + (p[j] - q[j]) * a for j in range(4)])
            out = acc / k
        else:
            out = self._warp(img, *p[:4]).astype(np.float32)
        rgb, flash = p[4], p[5]
        if n in self.whips or any(c - 1 <= n <= c for c in list(self.whips) + list(self.zooms)):
            rgb = max(rgb, 0.9)
        if rgb > 0.05:                    # chromatic split
            s = int(round(7 * K * rgb))
            if s:
                r = np.roll(out[..., 0], s, axis=1); b = np.roll(out[..., 2], -s, axis=1)
                out = np.dstack([r, out[..., 1], b])
        if flash > 0.02:
            out = out * (1 + 0.35 * flash) + 30 * flash
        self.last = p
        return np.clip(out, 0, 255).astype(np.uint8)

    def caption_offset(self, n):
        p = self.params(n, n)
        c = self.camera(n, n)
        return 0.35 * (p[0] - c[0]), 0.35 * (p[1] - c[1])
