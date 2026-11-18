"""Film look: section grading, bloom/halation, fade-from-white flashes, light leaks."""
import math
import numpy as np, cv2

DROP = 189
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

# fade-from-white after the white flash frames
FADES = {}
for start, curve in [(598, [0.35]),
                     (383, [0.3]), (283, [0.5, 0.2]), (605, [0.5, 0.2]),
                     (269, [0.92, 0.2]), (333, [0.92, 0.2]), (357, [0.85, 0.15]),
                     (484, [0.92, 0.2]), (509, [0.85, 0.15]), (548, [0.92, 0.2])]:
    for i, a in enumerate(curve): FADES[start + i] = a

MUZZLE = {189: 1.0, 190: 0.4, 403: 1.0, 404: 0.4}   # local glow around the muzzle flash only

# light leaks: (cut frame, length, direction)
LEAKS = [(105, 12, 1)]


class Look:
    def __init__(self, W, H):
        self.W, self.H = W, H
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        self.X, self.Y = x, y

    def grade(self, x, n):
        Y = (x @ LUMA)[..., None]
        if n < DROP:   # intro: muted, warm sepia, like the reference edits
            x = Y + (x - Y) * 0.32
            x = np.clip(x, 0, 1) ** 0.88   # lift the muted intro a touch
            x = 0.7 * x + 0.3 * (np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1)))
            x = x * np.array([1.05, 1.0, 0.9], np.float32) + np.array([0, 0, 0.018], np.float32) * (1 - Y) ** 2
        else:          # drop: colour comes in — teal shadows, warm highlights
            x = Y + (x - Y) * 1.02
            x = 0.68 * x + 0.32 * (np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1)))
            x = x + np.array([-0.03, 0.004, 0.03], np.float32) * (1 - Y) ** 5 \
                  + np.array([0.028, 0.012, -0.022], np.float32) * (4 * Y * (1 - Y)) \
                  + np.array([0.03, 0.012, -0.025], np.float32) * Y ** 2
            x = x * 1.04                       # let highlights reach white (white point was L*93)
        return 0.022 + x * (1 - 0.022)  # slightly matte blacks

    def bloom(self, x):
        W, H = self.W, self.H
        small = cv2.resize(x, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        hl = np.clip((small - 0.70) / 0.30, 0, 1)
        hl = cv2.GaussianBlur(hl, (0, 0), 6)
        hl = cv2.resize(hl, (W, H), interpolation=cv2.INTER_LINEAR)
        return x + 0.22 * hl * np.array([1.0, 0.82, 0.66], np.float32)  # warm halation

    def leak(self, x, n):
        for f0, dur, d in LEAKS:
            k = n - (f0 - 4)
            if not (0 <= k < dur + 4): continue
            t = k / (dur + 4)
            cx = self.W * ((-0.3 + 1.6 * t) if d > 0 else (1.3 - 1.6 * t))
            cy = self.H * (0.35 + 0.1 * t)
            r = self.H * 0.55
            m = np.exp(-(((self.X - cx) ** 2) + ((self.Y - cy) * 1.3) ** 2) / (2 * r * r))[..., None]
            inten = math.sin(math.pi * t) ** 1.5 * 0.7
            col = np.array([1.0, 0.48, 0.12], np.float32) * (1 - m) + np.array([1.0, 0.86, 0.62], np.float32) * m
            a = np.clip(m * inten, 0, 1) * col
            x = 1 - (1 - x) * (1 - a)       # screen blend
        return x

    def muzzle(self, x, v):
        W, H = self.W, self.H
        small = cv2.resize(x, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        hot = np.clip(((small @ LUMA) - 0.72) / 0.2, 0, 1)
        g = cv2.GaussianBlur(hot, (0, 0), 10) * 1.6 + cv2.GaussianBlur(hot, (0, 0), 3) * 1.0   # tight glow on the real flash
        g = cv2.resize(np.clip(g, 0, 1.5), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        a = np.clip(g * v, 0, 1) * np.array([1.0, 0.9, 0.7], np.float32)
        return 1 - (1 - x) * (1 - a)

    def apply(self, img, n):
        x = img.astype(np.float32) / 255
        x = self.grade(x, n)
        x = self.bloom(x)
        f = max(1, int(round(self.W / 1000)))   # ITO RGBFringe: opposing R/B offsets
        x = np.dstack([np.roll(x[..., 0], -f, axis=1), x[..., 1], np.roll(x[..., 2], f, axis=1)])
        x = self.leak(x, n)
        if n in MUZZLE: x = self.muzzle(x, MUZZLE[n])
        fl = np.random.default_rng(n * 31).normal() * (0.018 if n < DROP else 0.009)   # film exposure flicker
        x = x * (1 + fl)
        if n in FADES:
            a = FADES[n]; x = x * (1 - a) + a
        return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
