"""Soft black-and-white 'silver' look with halation, fine grain; colour bloom for the finale."""
import numpy as np, cv2
_VIG = {}
def _vig(h, w):
    if (h, w) not in _VIG:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.hypot((xx - w / 2) / (w * 0.62), (yy - h / 2) / (h * 0.75))
        _VIG[(h, w)] = np.clip(1 - 0.22 * np.clip(d - 0.55, 0, 1) ** 1.4, 0, 1)[..., None]
    return _VIG[(h, w)]
def _curve(Y):
    Y = np.clip((Y - 0.035) / 0.95, 0, 1)
    Y = Y + 0.10 * np.sin(np.pi * Y) * (Y - 0.45)          # gentle S around the mids
    return np.where(Y < 0.78, Y, 0.78 + (Y - 0.78) * 0.62)  # roll highlights off (no clipping)
def silver(img, n, glow=0.32, grain_sd=4.0, warm=0.018):
    x = img.astype(np.float32) / 255
    Y = x[..., 0] * 0.36 + x[..., 1] * 0.54 + x[..., 2] * 0.10    # red-weighted: skin glows
    Y = _curve(Y)
    h, w = Y.shape
    hi = np.clip((Y - 0.58) / 0.3, 0, 1) * Y
    bloom = cv2.GaussianBlur(hi, (0, 0), w / 60) * 0.65 + cv2.GaussianBlur(hi, (0, 0), w / 18) * 0.35
    Y = 1 - (1 - Y) * (1 - glow * bloom)                          # screen-blend halation
    out = np.repeat(Y[..., None], 3, 2)
    out = out * np.array([1 + warm, 1.0, 1 - warm * 1.2], np.float32)   # silver-gelatin warmth
    out = out * _vig(h, w)
    if grain_sd:
        r = np.random.default_rng(n * 104729 + 7)
        g = cv2.GaussianBlur(r.normal(0, grain_sd / 255, (h, w)).astype(np.float32), (0, 0), 0.7)[..., None]
        out = out + g * (0.6 + 0.8 * out * (1 - out) * 4) * 0.5
    return np.clip(out * 255, 0, 255).astype(np.uint8)
def colour_soft(img, n, glow=0.30):
    """the finale colour: soft warm, lifted, halation - what the B&W blooms into"""
    x = img.astype(np.float32) / 255
    Y = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    x = Y[..., None] + (x - Y[..., None]) * 0.82                  # a little less saturated
    x = np.clip((x - 0.03) / 0.95, 0, 1)
    x = np.where(x < 0.8, x, 0.8 + (x - 0.8) * 0.6)
    x = x * np.array([1.03, 1.0, 0.95], np.float32)
    h, w = Y.shape
    hi = np.clip((Y - 0.55) / 0.3, 0, 1)[..., None] * x
    bloom = cv2.GaussianBlur(hi, (0, 0), w / 40)
    x = 1 - (1 - x) * (1 - glow * bloom)
    return np.clip(x * _vig(h, w) * 255, 0, 255).astype(np.uint8)
