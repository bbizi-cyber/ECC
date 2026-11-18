"""Reusable edit transitions learned from the reference edits (fan-edit-references #14-16, #31-34).

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


# ---------------- learned from references #31-#32 (male edits, measured frame by frame) ----------------
def _affine(img, scale=1.0, rot=0.0, cx=0.5, cy=0.5, dx=0.0, dy=0.0):
    """zoom by `scale` and rotate by `rot` degrees around (cx, cy) (fractions), then shift by (dx, dy) px"""
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx * W, cy * H), rot, scale)
    M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)


def _blur_dir(img, length, angle=0.0):
    """directional motion blur of `length` px along `angle` degrees"""
    n = int(round(length))
    if n < 2: return img
    k = np.zeros((n, n), np.float32); k[n // 2, :] = 1.0 / n
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle, 1.0), (n, n))
    return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))


def window_reveal(cur, nxt, k, n=7, box=(0.32, 0.05, 0.28, 0.78), nxt_shift=(0, 0)):
    """#31: half a beat before the cut, the next shot shows through a tall window over the current one
    (n frames, ~0.23 s at 30 fps); the full cut then lands on the beat. box = (x, y, w, h) fractions;
    nxt_shift moves the incoming frame (px) so its face sits inside the window."""
    H, W = cur.shape[:2]
    x, y, w, h = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
    src = nxt if nxt_shift == (0, 0) else _affine(nxt, dx=nxt_shift[0], dy=nxt_shift[1])
    out = cur.copy()
    out[y:y + h, x:x + w] = src[y:y + h, x:x + w]
    return out


def zoom_through(img, k, n=8, total=2.4, cx=0.5, cy=0.5, blur=True):
    """#32: accelerating push into the subject over the last n frames before a cut (x2.4 in 0.27 s,
    eased in-out, motion blur growing with speed); cut at the peak to a shot that has the same subject
    at the same screen position (cx, cy)."""
    t = _ease((k + 1) / n)
    s = total ** t
    out = _affine(img, s, 0.0, cx, cy)
    if blur:                                                              # radial-ish blur ~ zoom speed
        v = total ** _ease((k + 1) / n) / max(total ** _ease(k / n), 1e-6) - 1
        r = int(min(9, 40 * v))
        if r >= 2:
            acc = out.astype(np.float32)
            for j in range(1, r + 1):
                acc += _affine(out, 1 - 0.006 * j, 0.0, cx, cy)
            out = (acc / (r + 1)).astype(np.uint8)
    return out


def punch_settle(img, k, n=4, start=1.14, cx=0.5, cy=0.5):
    """#32: the incoming shot lands ~14 % zoomed in and eases out to its framing in ~4 frames"""
    t = _ease(k / n) if k < n else 1.0
    return _affine(img, start + (1 - start) * (1 - (1 - t) ** 2), 0.0, cx, cy)


def tilt_snap(img, k, snap_k, tilt=-6.5, zoom=1.05, cx=0.5, cy=0.5):
    """#32: a Dutch-tilted (~6.5 deg) slightly zoomed hold that snaps level on the beat (frame snap_k)"""
    return _affine(img, zoom, tilt, cx, cy) if k < snap_k else img


def spin_out(img, k, n=8, total_rot=-20.0, pan_px=60, zoom=1.10):
    """#32: the shot rolls ~20 deg with a sideways smear over the last n frames into the next shot"""
    t = _ease((k + 1) / n)
    out = _affine(img, 1 + (zoom - 1) * t, total_rot * t, 0.5, 0.5, -pan_px * t, 0)
    return _blur_dir(out, 2 + 22 * t, angle=total_rot * t)


def quick_blend(a, b, k, n=2):
    """2-frame cross-dissolve (#31 at 7.33 s, #32 on the drop): softer than a hard cut, still on the beat"""
    t = (k + 1) / (n + 1)
    return cv2.addWeighted(a, 1 - t, b, t, 0)


def creep_zoom(img, k, per_frame=0.009, cx=0.5, cy=0.5, max_scale=1.35):
    """#32: after the drop every shot keeps pushing in (~0.8-1 % per frame)"""
    return _affine(img, min(max_scale, (1 + per_frame) ** k), 0.0, cx, cy)


# ---------------- learned from reference #33 (Fight Club, @raaah.qw, measured frame by frame) ----------------
def _luma(img):
    return (img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) / 255.0


def luma_burn(cur, nxt, k, n=3):
    """#33 signature transition (3 frames): the brightest parts of the next shot burn through the dark
    current shot first (luma key with a falling threshold), then both overlap, then the next shot lands
    with a 1-frame vertical smear. k = 0..n-1; afterwards play nxt as is."""
    if k >= n - 1:                                                        # last frame: vertical smear
        return _blur_dir(nxt, max(9, nxt.shape[0] // 22), angle=90)
    thr = 0.62 - 0.5 * (k / max(1, n - 2))                                # 0.62 -> 0.12
    Y = _luma(nxt)
    m = np.clip((Y - thr) / 0.12, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)[..., None]
    out = cur.astype(np.float32) * (1 - m) + nxt.astype(np.float32) * m
    if k > 0:                                                             # second frame: a ghost of both
        out = 0.55 * out + 0.45 * np.maximum(cur, nxt).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def flash_in(cur, nxt, k):
    """#33 drop flash (3 frames): current shot over-exposes, a lilac-white frame, then the next shot
    arrives washed out in warm white and is normal on the following frame."""
    if k == 0:                                                            # ramp: x1.4 plus a white lift
        x = cur.astype(np.float32) * 1.4 + 40
    elif k == 1:
        return np.full_like(cur, (251, 242, 251))
    else:                                                                 # warm wash-out of the next shot
        x = (255 - (255 - nxt.astype(np.float32)) * 0.18) * np.array([1.0, 0.86, 0.79], np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


def vsmear(img, strength=1.0):
    """#33: one vertically smeared frame on a cut (a vertical whip of 1 frame)"""
    return _blur_dir(img, max(5, int(img.shape[0] / 22 * strength)), angle=90)


def grain(img, n, sd=12.0):
    """#33 look: heavy film grain (residual sd ~12 on 8-bit), deterministic per frame index"""
    r = np.random.default_rng(n * 7919)
    g = r.normal(0, sd, img.shape[:2]).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.6)[..., None]
    return np.clip(img.astype(np.float32) + g, 0, 255).astype(np.uint8)


# ---------------- learned from reference #34 (Fight Club, "the ideal"; measured frame by frame) ----------------
def _hard_bw(img, lift=1.25, gamma=0.85):
    """high-key, high-contrast black-and-white (peak frame of the #34 face morph: sat ~37, p99 = 255)"""
    Y = np.clip(_luma(img) * lift, 0, 1) ** gamma
    Y = np.clip((Y - 0.06) / 0.88, 0, 1)
    return (np.repeat(Y[..., None], 3, 2) * 255).astype(np.uint8)


def _scan_glitch(img, k, rows=3):
    """a few thin horizontal colour-noise lines (the #34 morph tail)"""
    r = np.random.default_rng(k * 31337)
    H, W = img.shape[:2]; out = img.copy()
    for _ in range(rows):
        y = int(r.integers(0, H - 4)); h = int(r.integers(1, 4)); dx = int(r.integers(-W // 12, W // 12))
        band = np.roll(out[y:y + h], dx, axis=1).astype(np.int16)
        band[..., int(r.integers(0, 3))] += 60
        out[y:y + h] = np.clip(band, 0, 255).astype(np.uint8)
    return out


def _face_matrix(box_cur, box_other, W, H):
    """similarity transform putting the other face box onto the current one (normalized x0, y0, x1, y1)"""
    (a0, b0, a1, b1), (c0, d0, c1, d1) = box_cur, box_other
    s = ((a1 - a0) * W) / max(1e-6, (c1 - c0) * W)
    cxa, cya = (a0 + a1) / 2 * W, (b0 + b1) / 2 * H; cxo, cyo = (c0 + c1) / 2 * W, (d0 + d1) / 2 * H
    return np.float32([[s, 0, cxa - s * cxo], [0, s, cya - s * cyo]])


def face_morph(cur, other, k, n=21, peak=6, faces=None):
    """#34 "he is him": over ~0.7 s the other person's face, aligned onto the current face and turned into
    hard high-key B&W, rises through the current shot (peak at frame `peak`), then sinks back with a few
    glitch lines. Use in a quiet pause before the drop. k = 0..n-1.
    faces = (box_cur, box_other) from extras.face_boxes() aligns two different people by their faces;
    without it ORB alignment is tried (works for the same person in a similar pose)."""
    H, W = cur.shape[:2]
    M = _face_matrix(faces[0], faces[1], W, H) if faces else _align_matrix(cur, other)
    o = other if M is None else cv2.warpAffine(other, M.astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    o = _hard_bw(o)
    a = _ease((k + 1) / (peak + 1)) if k <= peak else 1 - _ease((k - peak) / (n - 1 - peak))
    if faces:                                                             # soft oval around the face: no frame edges show
        x0, y0, x1, y1 = faces[0]; fw = (x1 - x0) * W
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot((xx - (x0 + x1) / 2 * W) / (fw * 0.85), (yy - (y0 + y1) / 2 * H) / (fw * 1.15))
        a = a * np.clip((1.35 - d) / 0.5, 0, 1)[..., None]
    x = np.clip(cur.astype(np.float32) * (1 - a) + o.astype(np.float32) * a, 0, 255).astype(np.uint8)
    if peak < k < n - 3: x = _scan_glitch(x, k)
    return x


def echo_to_bw(frames, k, n=6, trail=3):
    """#34: colour -> bright B&W through a ghost echo (the last `trail` frames blended) over n frames.
    frames: the shot's frames so far (list), k: index of the current frame inside the transition."""
    i = len(frames) - 1
    acc = np.zeros_like(frames[i], np.float32); wsum = 0.0
    for j in range(trail):
        w = 0.55 ** j; acc += frames[max(0, i - j * 2)].astype(np.float32) * w; wsum += w
    x = (acc / wsum).astype(np.uint8)
    t = _ease((k + 1) / n)
    return cv2.addWeighted(x, 1 - t, _hard_bw(x, lift=1.45, gamma=0.8), t, 0)


def vanish_flicker(subject, plate, k, pattern=(7, 4, 3, 8)):
    """#34: the same framing with and without the person, alternating in `pattern` frame runs
    (subject, empty, subject, empty...). plate = clean plate of the same framing (a moment when he is
    not in the shot, or an inpainted frame). Returns the frame for transition index k."""
    edges = np.cumsum(pattern)
    run = int(np.searchsorted(edges, k, side='right'))
    return subject if run % 2 == 0 else plate


def engrave_text(img, text, font_path, size_frac=0.11, cx=0.5, cy=0.5, alpha=0.55):
    """#34: a word appears 'engraved' into the wall where the person was: serif caps, wide tracking,
    slightly lighter than the wall with a soft dark lower edge"""
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    H, W = img.shape[:2]
    f = ImageFont.truetype(font_path, int(H * size_frac))
    txt = ' '.join(text.upper())                                          # wide tracking
    base = Image.fromarray(img)
    lay = Image.new('L', (W, H), 0); d = ImageDraw.Draw(lay)
    bw, bh = d.textbbox((0, 0), txt, font=f)[2:]
    x, y = int(cx * W - bw / 2), int(cy * H - bh / 2)
    d.text((x, y), txt, fill=255, font=f)
    sh = lay.filter(ImageFilter.GaussianBlur(1.6))
    a = np.asarray(lay, np.float32)[..., None] / 255 * alpha
    s = np.roll(np.asarray(sh, np.float32), 2, axis=0)[..., None] / 255 * alpha * 0.6
    x0 = np.asarray(base, np.float32)
    out = x0 * (1 - s * 0.6)                                              # shadow under the letters
    out = out * (1 - a) + np.minimum(255, x0 * 1.35 + 18) * a             # lighter letter faces
    return np.clip(out, 0, 255).astype(np.uint8)


def stencil(img, mask=None, thr=None):
    """#34 ending: two-level 'xerox' portrait (~79 % black, ~7 % white) on black; mask (0..1) keeps
    only the person. thr defaults to the 72nd luma percentile of the person."""
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply((_luma(img) * 255).astype(np.uint8))
    Y = cv2.GaussianBlur(g.astype(np.float32) / 255, (0, 0), 0.8)        # local contrast keeps hair/shirt detail
    m = mask if mask is not None else np.ones_like(Y)
    if thr is None: thr = float(np.percentile(Y[m > 0.5], 62)) if (m > 0.5).any() else 0.5
    v = ((Y > thr) & (m > 0.5)).astype(np.float32)
    v = cv2.GaussianBlur(v, (0, 0), 0.6)
    return (np.repeat(v[..., None], 3, 2) * 255).astype(np.uint8)


# ---------------- #34 core (the user's reason it is ideal): heroes in the SAME position ----------------
_YN = {}


def face_anchor(img, model_path):
    """eye midpoint (x, y) and eye distance (fraction of width) of the biggest face, or None.
    model_path: YuNet ONNX (face_detection_yunet). Eyes line up better than box centres."""
    H, W = img.shape[:2]
    if model_path not in _YN: _YN[model_path] = cv2.FaceDetectorYN.create(model_path, '', (W, H), 0.6, 0.3, 5000)
    fd = _YN[model_path]; fd.setInputSize((W, H))
    _, f = fd.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if f is None: return None
    r = max(f, key=lambda q: q[2] * q[3])
    (rx, ry), (lx, ly) = r[4:6], r[6:8]
    return ((rx + lx) / 2 / W, (ry + ly) / 2 / H), float(np.hypot(lx - rx, ly - ry) / W)


def face_lock(img, anchor, size=None, target=(0.47, 0.36), target_size=None, max_zoom=1.3):
    """Reframe a shot so the face lands on the same screen point (and size) as in the shot before it.
    #34 keeps the face within ~0.1 of the frame across 76 % of its cuts (dialogue part: x 0.43-0.50,
    y 0.33-0.41), so the eye never has to jump - even when the cut goes to another person.
    anchor/size: from face_anchor() (eye midpoint, eye distance); target/target_size: the previous
    shot's anchor and size. Zooms in only (no borders), at most max_zoom: pick shots whose face is
    already close and let the lock correct the last 10-15 %."""
    H, W = img.shape[:2]
    cx, cy = anchor
    s = 1.0 if not (size and target_size) else float(np.clip(target_size / size, 1.0, max_zoom))
    for _ in range(40):                                                   # grow the zoom until the target is reachable
        dx, dy = target[0] - s * cx, target[1] - s * cy
        if (1 - s <= dx <= 0 and 1 - s <= dy <= 0) or s >= max_zoom: break
        s = min(max_zoom, s * 1.02)
    dx = float(np.clip(target[0] - s * cx, 1 - s, 0)); dy = float(np.clip(target[1] - s * cy, 1 - s, 0))
    M = np.float32([[s, 0, dx * W], [0, s, dy * H]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def pingpong(frames_a, frames_b, run=6, start='a'):
    """#34 signature run (21.1-21.9 s): two heroes in identical framing (same face point and size,
    same dark background) alternate every `run` frames (6 = 0.2 s), both slowly pushing in.
    frames_a/b: face-locked frame lists of equal framing; returns the interleaved list."""
    out, i, cur = [], 0, start
    n = min(len(frames_a), len(frames_b))
    while i < n:
        src = frames_a if cur == 'a' else frames_b
        out += src[i:i + run]; i += run; cur = 'b' if cur == 'a' else 'a'
    return out


# ---------------- #33 7.0-8.2 s: film to film through darkness (variant of the user favourite at 0.70-0.85 s) ----------------
def dark_bridge(cur, nxt, k, n_out=6, n_in=5, floor=0.18, light=None):
    """Two films joined as one shot: the outgoing frame dims to ~18 % over n_out frames, the cut sits in
    the dark on the beat, then the incoming frame (same angle, same kind of light) rises from the dark
    in n_in frames - faster around its own light source when `light` = (x, y) fraction is given.
    k = 0..n_out+n_in-1; frames k < n_out come from cur, the rest from nxt."""
    if k < n_out:
        g = 1 - (1 - floor) * _ease((k + 1) / n_out)
        return np.clip(cur.astype(np.float32) * g, 0, 255).astype(np.uint8)
    t = _ease((k - n_out + 1) / n_in)
    g = floor + (1 - floor) * t
    x = nxt.astype(np.float32) * g
    if light is not None and t < 1:                                       # the flame/lamp wakes first
        H, W = nxt.shape[:2]; yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot(xx - light[0] * W, yy - light[1] * H) / (0.35 * W)
        x = np.maximum(x, nxt.astype(np.float32) * np.clip(1.2 - d, 0, 1)[..., None] * min(1.0, 2 * t + 0.4))
    return np.clip(x, 0, 255).astype(np.uint8)


# ---------------- #36: full-screen stretch (male edits, sometimes, only for dark/intense material) ----------------
def stretch_full(img, W=1080, H=1920):
    """the whole 16:9 frame squeezed onto the vertical screen with no bars (aspect distortion ~x3.2):
    tall, thin, uneasy. Use the active picture (letterbox removed) as input."""
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA if img.shape[0] > H else cv2.INTER_CUBIC)
