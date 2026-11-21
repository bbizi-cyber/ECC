"""Extra edit effects: person segmentation, freeze-frame with outline glow, glitch, letterbox,
optical-flow frame interpolation (for smooth slow-mo / speed ramps)."""
import math
import numpy as np, cv2
import os
WORK = os.environ.get('FAN_EDIT_WORK', os.path.expanduser('~/fan-edits'))   # work folder: sources, models, outputs

MODELS = WORK + '/models/'
_SEG = None
K = 1.0   # pixel scale vs the original 834-px-high frame (set by render.py)


# ---------------- segmentation (U^2-Net human seg, ONNX) ----------------
def person_mask(img, keep_right_of=None):
    """img RGB uint8 -> float mask 0..1 of the person(s)"""
    global _SEG
    import onnxruntime as ort
    if _SEG is None:
        _SEG = ort.InferenceSession(MODELS + 'u2net_human_seg.onnx', providers=['CPUExecutionProvider'])
    H, W = img.shape[:2]
    x = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    out = _SEG.run(None, {_SEG.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    m = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
    m = np.clip((m - 0.35) / 0.3, 0, 1)
    # solid silhouette: biggest region only, inner holes (hair gaps) filled
    k = max(9, int(H * 0.031)) | 1          # opening: drop thin props stuck to the silhouette (hooks, cables)
    binm = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        solid = np.zeros_like(m, np.uint8)
        cv2.drawContours(solid, [max(cnts, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
        m = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 1.0)
    if keep_right_of is not None:   # keep only what is right of the cut line (soft edge)
        ramp = np.clip((np.arange(W) / W - keep_right_of) / 0.05, 0, 1)[None, :]
        m = m * ramp
    return m


def freeze_frame(img, mask, k, total, outline=True):
    """B&W freeze with darkened background and a glowing white outline around the subject."""
    x = img.astype(np.float32) / 255
    Y = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    Y = np.clip((Y - 0.04) * 1.15, 0, 1)                      # punchy mono
    m = mask[..., None]
    bg = Y[..., None] * 0.42
    fg = Y[..., None]
    out = bg * (1 - m) + fg * m
    d = max(3, int(round(9 * K))) | 1
    edge = cv2.dilate((mask > 0.5).astype(np.uint8), np.ones((d, d), np.uint8)).astype(np.float32) - (mask > 0.5)
    edge = cv2.GaussianBlur(np.clip(edge, 0, 1), (0, 0), 1.2 * K)
    glow = cv2.GaussianBlur(edge, (0, 0), 9 * K) * 1.6
    grow = min(1.0, (k + 1) / 3) if outline else 0.0             # outline draws on over 3 frames
    out = out + (edge * 0.95 + glow * 0.5)[..., None] * grow
    out = np.repeat(out, 3, axis=2) if out.shape[2] == 1 else out
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# ---------------- glitch ----------------
def glitch(img, n, strength=1.0):
    r = np.random.default_rng(n * 104729)
    H, W = img.shape[:2]
    out = img.copy()
    y = 0
    while y < H:
        h = int(r.integers(6, 60) * K)
        if r.random() < 0.55:
            dx = int(r.integers(-45, 46) * strength * K)
            band = np.roll(img[y:y + h], dx, axis=1)
            c = int(r.integers(4, 14) * K)
            band = band.copy()
            band[..., 0] = np.roll(band[..., 0], c, axis=1)
            band[..., 2] = np.roll(band[..., 2], -c, axis=1)
            out[y:y + h] = band
        y += h
    out[::3] = (out[::3].astype(np.float32) * 0.82).astype(np.uint8)   # scanlines
    return out


# ---------------- letterbox ----------------
def letterbox(img, bar):
    if bar <= 0: return img
    out = img.copy()
    b = int(round(bar))
    out[:b] = 0; out[-b:] = 0
    return out


# ---------------- optical-flow interpolation ----------------
_DIS = None


def interp(f0, f1, a):
    """frame between f0 and f1 at fraction a (0..1), flow-based"""
    global _DIS
    if a < 0.02: return f0
    if a > 0.98: return f1
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    g0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY); g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    fl01 = _DIS.calc(g0, g1, None); fl10 = _DIS.calc(g1, g0, None)
    H, W = g0.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # backward warps: a pixel at time a came from f0 at x + a*flow(1->0) and from f1 at x + (1-a)*flow(0->1)
    a = np.float32(a); b = np.float32(1) - a
    w0 = cv2.remap(f0, gx + a * fl10[..., 0], gy + a * fl10[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(f1, gx + b * fl01[..., 0], gy + b * fl01[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = float(a)
    return cv2.addWeighted(w0, 1 - a, w1, a, 0)


# ---------------- face tracking for vertical reframing (YuNet) ----------------
_FD = None


def _faces(img):
    global _FD
    h, w = img.shape[:2]
    sc = 640 / w
    small = cv2.resize(img, (640, int(round(h * sc))), interpolation=cv2.INTER_AREA)
    if _FD is None:
        _FD = cv2.FaceDetectorYN.create(MODELS + 'yunet.onnx', '', (320, 320), 0.6, 0.3, 5000)
    _FD.setInputSize((small.shape[1], small.shape[0]))
    _, f = _FD.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if f is None: return []
    return [((x + bw / 2) / sc / w, bw * bh) for x, y, bw, bh in f[:, :4] if bw * bh > 150]


def face_track(frames, nframes, cx_hint):
    """per-output-frame horizontal centre (0..1) following the face nearest the hint, smoothed"""
    idx = sorted(set(list(range(0, nframes, 3)) + [nframes - 1]))
    pts = []
    cur = cx_hint
    for i in idx:
        fs = _faces(frames[min(i, len(frames) - 1)])
        if fs:
            # nearest to where we are, bigger faces preferred
            best = min(fs, key=lambda t: abs(t[0] - cur) - 0.00002 * t[1])
            if abs(best[0] - cur) < 0.35 or not pts:
                cur = best[0]; pts.append((i, cur))
    if not pts: return np.full(nframes, cx_hint)
    xs = np.interp(np.arange(nframes), [p[0] for p in pts], [p[1] for p in pts])
    k = 7                                    # smooth: no jittery camera
    pad = np.pad(xs, (k, k), mode='edge')
    xs = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode='same')[k:-k]
    return xs


# ---------------- per-shot colour / exposure matching ----------------
def color_match(frames, strength=0.5):
    samp = [frames[0], frames[len(frames) // 2], frames[-1]]
    lab = np.concatenate([cv2.cvtColor(f, cv2.COLOR_RGB2LAB).reshape(-1, 3)[::7] for f in samp]).astype(np.float32)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    da = float(np.clip((128 - a.mean()) * 0.55, -12, 12))
    db = float(np.clip((128 - b.mean()) * 0.55, -12, 12))
    p3, p97 = np.percentile(L, 3), np.percentile(L, 97)
    t3 = p3 + (12 - p3) * strength
    t97 = p97 + (235 - p97) * strength
    g = (t97 - t3) / max(20.0, p97 - p3)
    lut = np.clip((np.arange(256) - p3) * g + t3, 0, 255).astype(np.uint8)
    out = []
    for f in frames:
        x = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)
        x[..., 0] = cv2.LUT(x[..., 0], lut)
        ab = x[..., 1:].astype(np.int16) + np.array([round(da), round(db)], np.int16)
        x[..., 1:] = np.clip(ab, 0, 255).astype(np.uint8)
        out.append(cv2.cvtColor(x, cv2.COLOR_LAB2RGB))
    return out


def face_boxes(img):
    """normalized (x0, y0, x1, y1) boxes of faces in an RGB frame"""
    global _FD
    h, w = img.shape[:2]
    sc = 640 / w
    small = cv2.resize(img, (640, int(round(h * sc))), interpolation=cv2.INTER_AREA)
    if _FD is None:
        _FD = cv2.FaceDetectorYN.create(MODELS + 'yunet.onnx', '', (320, 320), 0.6, 0.3, 5000)
    _FD.setInputSize((small.shape[1], small.shape[0]))
    _, f = _FD.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if f is None: return []
    sw, sh = small.shape[1], small.shape[0]
    return [(x / sw, y / sh, (x + bw) / sw, (y + bh) / sh) for x, y, bw, bh in f[:, :4]]


def tone(frames, black=0.05, hi=0.8):
    """per-shot tone fix: crush lifted blacks, roll highlights off above 0.6"""
    x = np.arange(256, dtype=np.float32) / 255
    y = np.clip((x - black) / (1 - black), 0, 1)
    y = np.where(y < 0.6, y, 0.6 + (y - 0.6) * hi)
    lut = (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)
    return [cv2.LUT(f, lut) for f in frames]
