"""Reusable edit transitions learned from the reference edits (fan-edit-references #14-16).

All functions take/return HxWx3 uint8 RGB frames of the same size.
- match_dissolve: "match cut" — the incoming shot is aligned onto the outgoing silhouette (ORB similarity),
  cross-dissolved over a few frames with a slight exposure dip, then relaxes to its own framing.
- card_stack:     incoming shots land as portrait cards on top of the current shot, one per (half-)beat,
  slightly overlapping, then the full incoming frame takes over.
- flip180:        upside-down insert (the "Tyler upside down" beat).
"""
import numpy as np, cv2


def _ease(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)


_ALIGN = {}


def _align_matrix(a, b):
    """similarity transform that maps b onto a (cached per pair)"""
    key = (a.ctypes.data, b.ctypes.data)
    if key in _ALIGN: return _ALIGN[key]
    ga = cv2.cvtColor(cv2.resize(a, (480, int(480 * a.shape[0] / a.shape[1]))), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (480, int(480 * b.shape[0] / b.shape[1]))), cv2.COLOR_RGB2GRAY)
    s = a.shape[1] / 480
    orb = cv2.ORB_create(1500); k1, d1 = orb.detectAndCompute(gb, None); k2, d2 = orb.detectAndCompute(ga, None)
    M = None
    if d1 is not None and d2 is not None:
        m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2), key=lambda x: x.distance)[:200]
        if len(m) > 20:
            P = np.float32([k1[x.queryIdx].pt for x in m]) * s; Q = np.float32([k2[x.trainIdx].pt for x in m]) * s
            M, inl = cv2.estimateAffinePartial2D(P, Q, ransacReprojThreshold=4)
            if M is not None:
                sc = np.hypot(M[0, 0], M[1, 0])
                if not (0.7 < sc < 1.4) or inl.sum() < 15: M = None          # unrelated shots: plain dissolve
    _ALIGN[key] = M
    return M


def match_dissolve(a, b, k, n=6, b_next=None):
    """frame k (0..n-1) of a match-cut dissolve from a to b; returns the blended frame"""
    H, W = a.shape[:2]
    t = _ease((k + 1) / (n + 1))
    M = _align_matrix(a, b)
    bb = b
    if M is not None:                                                     # b starts aligned to a, relaxes to identity
        I = np.float32([[1, 0, 0], [0, 1, 0]])
        Mk = (1 - t) * M + t * I
        bb = cv2.warpAffine(b, Mk.astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    x = a.astype(np.float32) * (1 - t) + bb.astype(np.float32) * t
    x *= 1 - 0.10 * np.sin(np.pi * t)                                     # slight exposure dip mid-dissolve
    return np.clip(x, 0, 255).astype(np.uint8)


def _card(img, w, h, cx=0.5):
    """crop img to the card aspect around horizontal centre cx and resize"""
    H, W = img.shape[:2]; ar = w / h
    ch = H; cw = int(round(ch * ar))
    if cw > W: cw = W; ch = int(round(cw / ar))
    x0 = int(np.clip(cx * W - cw / 2, 0, W - cw)); y0 = (H - ch) // 2
    return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)


def card_stack(base, cards, k, layout=None):
    """base: current frame; cards: list of (frame, start_k, cx); k: frame index inside the transition.
    Each card lands at its start_k with a 2-frame settle (scale 1.06 -> 1.0) and a soft shadow."""
    H, W = base.shape[:2]
    if layout is None:                                                    # reference #16: left-centre, then right
        layout = [(0.26, 0.10, 0.36, 0.74), (0.50, 0.04, 0.38, 0.86), (0.18, 0.08, 0.40, 0.84)]
    out = base.astype(np.float32)
    for i, (img, k0, cx) in enumerate(cards):
        if k < k0: continue
        lx, ly, lw, lh = layout[i % len(layout)]
        w, h = int(W * lw), int(H * lh)
        settle = max(0.0, 1 - (k - k0) / 2.0)
        sc = 1 + 0.06 * settle
        cw, chh = int(w * sc), int(h * sc)
        card = _card(img, cw, chh, cx).astype(np.float32)
        x0 = int(W * lx - (cw - w) / 2); y0 = int(H * ly - (chh - h) / 2)
        # shadow
        sh = np.zeros((H, W), np.float32)
        sh[max(0, y0 + 10):min(H, y0 + chh + 10), max(0, x0 + 10):min(W, x0 + cw + 10)] = 1
        sh = cv2.GaussianBlur(sh, (0, 0), 18)[..., None] * 0.55
        out = out * (1 - sh)
        xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + cw), min(H, y0 + chh)
        out[ya:yb, xa:xb] = card[ya - y0:yb - y0, xa - x0:xb - x0]
    return np.clip(out, 0, 255).astype(np.uint8)


def flip180(img):
    return np.ascontiguousarray(img[::-1, ::-1])
