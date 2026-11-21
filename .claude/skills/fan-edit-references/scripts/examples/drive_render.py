"""Ryan Gosling x Kavinsky - Nightcall: Drive (2011) fan edit.

Hook: opening monologue word by word over a muffled bar -> drop on the chorus with a white flash
and the pink "Drive" title -> 4 chorus lines as 4 chapters (love / elevator / darkness / calm) -> fade, end tag.
"""
import json, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

D = WORK + '/drive_edit'
CC = WORK + '/cc/drive'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # the fan-edit library
import extras as EX                                         # flow interpolation for slow motion
WORK = os.environ.get('FAN_EDIT_WORK', os.path.expanduser('~/fan-edits'))   # work folder: sources, models, outputs

FPS_N, FPS_D = 24000, 1001
FPS = FPS_N / FPS_D
W, H = 1920, 1080
PREVIEW = '--preview' in sys.argv
VERT = '--vertical' in sys.argv
VW, VH, BAND_Y, BAND_H = 1080, 1920, 616, 608          # TikTok canvas, 16:9 band 40 px above centre
OUT = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else D + '/out/drive_master.mp4'

SONG = D + '/nightcall.wav'
T0 = 70.147                                                 # song time at output 0 (a beat)
BEATS = np.load(D + '/beats.npy')
OB = BEATS[BEATS >= T0 - 1e-3] - T0                         # output beat times
def bt(k): return float(OB[k])

def src(name):
    if name.startswith('d'): return f'{CC}/{name}.mp4'
    return f'{D}/yt/{name}.mp4'

# (src, start, beats, cx, cy, zoom0, zoom1, speed)
SHOTS = [
    # hook: 10 beats, monologue
    ('opening', 17.0, 4, 0.62, 0.5, 1.00, 1.05, 1.0),
    ('d002',     0.05, 3, 0.50, 0.5, 1.00, 1.06, 0.62),
    ('d001',     9.40, 3, 0.50, 0.5, 1.00, 1.05, 0.6),    # face in the car (keeps the toothpick close-up for the ending)
    # chorus line 1: love
    ('scenepack', 1.2, 2, 0.50, 0.5, 1.00, 1.04, 1.0),
    ('d055',     3.60, 2, 0.45, 0.5, 1.00, 1.04, 1.0),
    ('scenepack', 134.3, 2, 0.55, 0.5, 1.00, 1.04, 1.0),
    ('scenepack', 155.6, 2, 0.45, 0.5, 1.00, 1.04, 1.0),
    # chorus line 2: elevator
    ('d020',     9.70, 2, 0.62, 0.5, 1.00, 1.04, 1.0),
    ('d020',     3.00, 2, 0.40, 0.5, 1.00, 1.04, 1.0),
    ('elevator', 186.4, 4, 0.60, 0.42, 1.25, 1.33, 0.6),   # the kiss, flow slow-motion
    # chorus line 3: darkness, one beat each
    ('elevator', 269.2, 1, 0.55, 0.42, 1.25, 1.28, 1.0),   # the stomp, in silhouette under the lamp
    ('opening', 344.6, 1, 0.50, 0.5, 1.00, 1.03, 1.0),
    ('d026',     0.40, 1, 0.42, 0.5, 1.00, 1.03, 1.0),
    ('d089',     7.60, 1, 0.50, 0.5, 1.00, 1.03, 1.0),
    ('opening', 524.0, 1, 0.45, 0.5, 1.00, 1.03, 1.0),
    ('scenepack', 54.2, 1, 0.50, 0.5, 1.00, 1.03, 1.0),
    ('d008',    17.80, 1, 0.45, 0.40, 1.08, 1.10, 1.0),
    ('scenepack', 25.0, 1, 0.45, 0.5, 1.00, 1.03, 1.0),
    # chorus line 4: calm, him alone
    ('scenepack', 142.0, 2, 0.50, 0.5, 1.00, 1.04, 1.0),
    ('d097',     7.90, 2, 0.55, 0.5, 1.00, 1.04, 1.0),    # profile only (clip cuts at 9.93)
    ('d026',     6.45, 4, 0.50, 0.5, 1.00, 1.08, 0.5),
]

# beat boundaries
bounds = [0]
for s in SHOTS: bounds.append(bounds[-1] + s[2])
CUT = [round(bt(b) * FPS) for b in bounds]                 # cut frames
DROP_F = CUT[3]                                             # chorus downbeat
L3_F, L4_F, END_F = CUT[10], CUT[18], CUT[-1]
TAIL = round(2.4 * FPS)                                     # black + end tag
TOTAL = END_F + TAIL

# white flashes (frame -> mix), punch-zooms (frame -> strength)
FLASH = {DROP_F: 1.0, L3_F: 1.0, CUT[14]: 0.85}
PUNCH = {CUT[k]: (0.07 if k == 10 else 0.045) for k in range(10, 18) if k != 14}   # no punch on the soft police shot
SHAKE = {CUT[10]: 1.0, CUT[17]: 0.8}
PULSE = [round(bt(bounds[5] + 1) * FPS), round(bt(bounds[9] + 2) * FPS)]   # only twice: by the stream, on the kiss
LEAKS = [CUT[7], CUT[18]]                                   # neon light leaks on chapter changes

FONT_W = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf', 118)
FONT_T = ImageFont.truetype(D + '/fonts/MrDafoe.ttf', 300)
FONT_TAG = ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-BlackItalic.otf', 80)
PINK = np.array([255, 46, 147], np.float32)


ACTIVE_FIX = {'opening': (1920, 800, 0, 140)}


def active_area(path, t):
    for k, v in ACTIVE_FIX.items():
        if path.endswith(f'/{k}.mp4'): return v
    out = subprocess.run(['ffmpeg', '-ss', str(max(0, t - 0.5)), '-t', '2', '-i', path, '-vf', 'cropdetect=24:2:0',
                          '-f', 'null', '-'], capture_output=True, text=True).stderr
    cr = [tuple(map(int, l.split('crop=')[1].split()[0].split(':'))) for l in out.splitlines() if 'crop=' in l]
    return max(cr, key=lambda c: c[0] * c[1])


def load(shot, n):
    name, t, beats, cx, cy, z0, z1, sp = shot
    path = src(name)
    aw, ah, ax, ay = active_area(path, t)
    need = n / FPS * sp + 0.25
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', path, '-t', f'{need:.3f}',
                          '-vf', f'crop={aw}:{ah}:{ax}:{ay}', '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True, check=True).stdout
    nat = np.frombuffer(raw, np.uint8).reshape(-1, ah, aw, 3)
    if name in ('scenepack', 'elevator'):                    # uploader over-sharpened: soften halos before scaling
        sig = 0.75 if name == 'scenepack' else 0.5
        nat = np.stack([cv2.GaussianBlur(f, (0, 0), sig) for f in nat])
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate',
                        '-of', 'csv=p=0', path], capture_output=True, text=True).stdout.split()[0].strip(',')
    nf = eval(r)
    out = []
    for i in range(n):
        x = i / FPS * sp * nf
        i0 = min(int(x), len(nat) - 2)
        f = EX.interp(nat[i0], nat[i0 + 1], x - i0) if sp < 0.99 else nat[min(int(round(x)), len(nat) - 1)]
        u = i / max(1, n - 1); e = u * u * (3 - 2 * u)
        z = z0 + (z1 - z0) * e
        ch = ah / z; cw = ch * 16 / 9
        if cw > aw: cw = aw; ch = cw * 9 / 16
        x0 = float(np.clip(cx * aw - cw / 2, 0, aw - cw)); y0 = float(np.clip(cy * ah - ch / 2, 0, ah - ch))
        M = np.float32([[W / cw, 0, -x0 * W / cw], [0, H / ch, -y0 * H / ch]])
        out.append(cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT))
    out = EX.color_match(out, strength=0.3)
    print(f'  {name} @{t}: {n} frames, active {aw}x{ah} -> scale {H / (ah / z0):.2f}', file=sys.stderr, flush=True)
    return out


LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def grade(img, n):
    x = img.astype(np.float32) / 255
    Y = (x @ LUMA)[..., None]
    x = 0.75 * x + 0.25 * (np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1)))          # gentle S
    x = x + np.array([-0.025, 0.004, 0.03], np.float32) * (1 - Y) ** 4 \
          + np.array([0.03, 0.0, 0.012], np.float32) * Y ** 2                         # teal shadows, pink-warm highs
    if n < DROP_F:                                                                    # hook: a touch darker and cooler
        x = Y + (x - Y) * 0.8; x = x * 0.92
    small = cv2.resize(x, (W // 4, H // 4), interpolation=cv2.INTER_AREA)            # bloom on lights / neon
    hl = cv2.GaussianBlur(np.clip((small - 0.68) / 0.32, 0, 1), (0, 0), 7)
    x = x + 0.2 * cv2.resize(hl, (W, H)) * np.array([1.0, 0.8, 0.85], np.float32)
    return np.clip(x, 0, 1)


VIG = None


def leak(x, n):
    for f0 in LEAKS:
        k = n - (f0 - 7)
        if not (0 <= k < 18): continue
        t = k / 17
        yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
        cx = (W // 4) * (-0.2 + 1.4 * t); cy = (H // 4) * (0.45 + 0.15 * np.sin(t * 3))
        r = (W // 4) * 0.45
        m = np.exp(-(((xx - cx) ** 2) + ((yy - cy) * 1.4) ** 2) / (2 * r * r))
        m = cv2.resize(m, (W, H))[..., None] * (np.sin(np.pi * t) ** 1.6) * 0.48
        col = np.array([1.0, 0.25, 0.62], np.float32) * m + np.array([0.1, 0.55, 0.65], np.float32) * (m ** 3) * 0.6
        x = 1 - (1 - x) * (1 - np.clip(col, 0, 1))                 # screen blend
    return x


def chroma(x, n):
    for f0 in PUNCH:
        k = n - f0
        if 0 <= k < 3:
            sft = [6, 3, 1][k]
            x = np.dstack([np.roll(x[..., 0], sft, axis=1), x[..., 1], np.roll(x[..., 2], -sft, axis=1)])
    return x


def finish(x, n):
    global VIG
    if VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        VIG = (1 - 0.28 * np.clip(r - 0.55, 0, 1) ** 1.6)[..., None]
    x = x * VIG
    g = np.random.default_rng(n).normal(0, 0.015, (H // 2, W // 2)).astype(np.float32)
    x = x + cv2.resize(g, (W, H))[..., None]
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def warp_fx(img, n):
    s = 1.0; dx = dy = 0.0
    for f0, st in PUNCH.items():
        k = n - f0
        if 0 <= k < 8: s += st * np.exp(-k / 2.2)
    for f0, st in SHAKE.items():
        k = n - f0
        if 0 <= k < 8:
            a = 18 * st * np.exp(-k / 2.5)
            dx += a * np.cos(k * 2.1 + f0); dy += a * np.sin(k * 1.7 + f0)
    for f0 in PULSE:
        k = n - f0
        if 0 <= k < 7: s += 0.014 * np.exp(-k / 2.6)
    if s == 1.0 and dx == 0 and dy == 0: return img
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s); M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)


# hook captions: one word at a time (gradient white -> pink), timed to the monologue
WORDS = [(w0, w1, w.strip()) for seg in json.load(open(D + '/opening_words.json')) for (w0, w1, w) in seg['words']]
DLG_IN, DLG_AT = 0.90, 0.25                                   # dialogue source in-point, placed at output time
CAPS = []
for w0, w1, w in WORDS:
    o0 = w0 - DLG_IN + DLG_AT
    if 0 <= o0 < bt(10) - 0.15:
        CAPS.append([o0, o0 + max(0.2, w1 - w0), w.upper().strip('.,')])
for i in range(len(CAPS) - 1):                                # hold each word until the next (or 0.5 s pause cap)
    CAPS[i][1] = min(CAPS[i + 1][0], CAPS[i][0] + 0.9)
GRAD = None


def caption(img, word):
    global GRAD
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), word, font=FONT_W)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x, y = (W - tw) / 2 - bb[0], H * 0.5 - th / 2 - bb[1]
    m = Image.new('L', im.size, 0); ImageDraw.Draw(m).text((x, y), word, font=FONT_W, fill=255)
    sh = m.filter(ImageFilter.GaussianBlur(9)).point(lambda v: int(v * 0.7))
    im.paste((0, 0, 0), (3, 6), sh)
    yy0, yy1 = int(y + bb[1]), int(y + bb[3])
    tt = np.clip((np.arange(H) - yy0) / max(1, yy1 - yy0), 0, 1)[:, None, None]
    g = ((1 - tt) * np.array([255, 255, 255], np.float32) + tt * PINK)
    g = np.broadcast_to(g, (H, W, 3)).astype(np.uint8)
    im.paste(Image.fromarray(g), (0, 0), m)
    return np.asarray(im)


FLICK = [1.0, 0.18, 0.95, 0.05, 0.7, 1.0, 0.35, 1.0]        # neon tube catching: flicker, then steady


def neon(img, txt, size, cx, cy, a):
    im = Image.fromarray(img)
    f = ImageFont.truetype(D + '/fonts/MrDafoe.ttf', int(size))
    d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), txt, font=f)
    x, y = cx - (bb[2] - bb[0]) / 2 - bb[0], cy - (bb[3] - bb[1]) / 2 - bb[1]
    m = Image.new('L', im.size, 0); ImageDraw.Draw(m).text((x, y), txt, font=f, fill=255)
    s = size / 390
    im.paste((200, 20, 110), (0, 0), m.filter(ImageFilter.GaussianBlur(45 * s)).point(lambda v: min(255, int(v * 1.6 * a))))
    im.paste(tuple(int(c) for c in PINK), (0, 0), m.filter(ImageFilter.GaussianBlur(18 * s)).point(lambda v: int(v * 0.9 * a)))
    im.paste((255, 120, 190), (0, 0), m.filter(ImageFilter.GaussianBlur(5 * s)).point(lambda v: int(v * a)))
    im.paste((255, 70, 160), (0, 0), m.point(lambda v: int(v * a)))
    return np.asarray(im)


def credit(img, n):
    """film-style pink script credit over the LA night city"""
    a0, a1 = CUT[1] + 6, CUT[2] - 2
    if not (a0 <= n < a1): return img
    a = min(1, (n - a0) / 8) * min(1, (a1 - n) / 6)
    return neon(img, 'Ryan Gosling', 170, W * 0.5, H * 0.70, a * 0.95)


def title(img, k, nk):
    """pink neon 'Drive' over the first chorus shot"""
    a = min(1, (k + 1) / 3) * min(1, (nk - k) / 4)
    if k < len(FLICK): a *= FLICK[k]
    sc = 1.12 - 0.12 * min(1, k / 5)
    im = Image.fromarray(img)
    txt = 'Drive'
    f = ImageFont.truetype(D + '/fonts/MrDafoe.ttf', int(390 * sc))
    d = ImageDraw.Draw(im); bb = d.textbbox((0, 0), txt, font=f)
    x, y = (W - (bb[2] - bb[0])) / 2 - bb[0], H * 0.47 - (bb[3] - bb[1]) / 2 - bb[1]
    m = Image.new('L', im.size, 0); ImageDraw.Draw(m).text((x, y), txt, font=f, fill=255)
    wide = m.filter(ImageFilter.GaussianBlur(45)).point(lambda v: min(255, int(v * 1.6 * a)))
    im.paste((200, 20, 110), (0, 0), wide)
    glow = m.filter(ImageFilter.GaussianBlur(18)).point(lambda v: int(v * 0.9 * a))
    glow2 = m.filter(ImageFilter.GaussianBlur(5)).point(lambda v: int(v * a))
    im.paste(tuple(int(c) for c in PINK), (0, 0), glow)
    im.paste((255, 120, 190), (0, 0), glow2)
    im.paste((255, 70, 160), (0, 0), m.point(lambda v: int(v * a)))
    return np.asarray(im)


FONT_CAP = ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf', 62)
FONT_KEY = ImageFont.truetype(D + '/fonts/MrDafoe.ttf', 104)
KEYWORDS = {'city', 'route'}
PHRASES = []
for seg in json.load(open(D + '/opening_words.json')):
    ws = [(w0 - DLG_IN + DLG_AT, w1 - DLG_IN + DLG_AT, w.strip()) for (w0, w1, w) in seg['words']]
    ws = [(a, b, w.rstrip('.,')) for a, b, w in ws if 0 <= a < bt(10) - 0.2]
    if ws: PHRASES.append(ws)


def phrase_layout(ws):
    """fixed layout for a whole phrase (so words never shift as they appear): list of (word, x, y, key)"""
    d = ImageDraw.Draw(Image.new('L', (8, 8)))
    space = d.textlength(' ', font=FONT_CAP) * 1.05
    items = []
    for a, b, w in ws:
        key = w.lower() in KEYWORDS
        f = FONT_KEY if key else FONT_CAP
        items.append([a, b, w, key, d.textlength(w, font=f) + (10 if key else 0)])
    maxw = VW - 2 * 96
    lines, cur, cw = [], [], 0
    for it in items:
        add = it[4] + (space if cur else 0)
        if cur and cw + add > maxw:
            lines.append(cur); cur, cw = [], 0; add = it[4]
        cur.append(it); cw += add
    if cur: lines.append(cur)
    if len(lines) > 1 or len(items) > 5:                       # two lines, split where the longer line is shortest
        def lw(line): return sum(i[4] for i in line) + space * (len(line) - 1)
        best = min(range(1, len(items)), key=lambda k: max(lw(items[:k]), lw(items[k:])))
        lines = [items[:best], items[best:]]
    lh = 84
    y0 = BAND_Y + BAND_H + 70
    out = []
    for li, line in enumerate(lines):
        tw = sum(i[4] for i in line) + space * (len(line) - 1)
        x = (VW - tw) / 2
        for it in line:
            out.append((it[0], it[2], x, y0 + li * lh, it[3]))
            x += it[4] + space
    return out, ws[-1][1]


LAYOUTS = [phrase_layout(ws) for ws in PHRASES]


def captions_v(canvas, n):
    t = n / FPS
    im = Image.fromarray(canvas)
    for (items, t_end) in LAYOUTS:
        t_out = min(t_end + 0.45, bt(10) - 0.12)
        if t < items[0][0] or t > t_out + 0.3: continue
        fade_out = 1.0 if t <= t_out else max(0.0, 1 - (t - t_out) / 0.25)
        for (ta, w, x, y, key) in items:
            k = (t - ta) * FPS
            if k < 0: continue
            e = min(1.0, k / 6.0); e = e * e * (3 - 2 * e)
            a = e * fade_out
            if a <= 0.01: continue
            dy = (1 - e) * 14
            f = FONT_KEY if key else FONT_CAP
            m = Image.new('L', im.size, 0)
            yy = y + dy - (34 if key else 0)
            ImageDraw.Draw(m).text((x + (5 if key else 0), yy), w, font=f, fill=255)
            sh = m.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.55 * a))
            im.paste((0, 0, 0), (0, 3), sh)
            if key:
                wide = m.filter(ImageFilter.GaussianBlur(22)).point(lambda v: min(255, int(v * 1.3 * a)))
                im.paste((210, 25, 120), (0, 0), wide)
                glow = m.filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.9 * a))
                im.paste((255, 110, 185), (0, 0), glow)
                im.paste((255, 80, 165), (0, 0), m.point(lambda v: int(v * a)))
            else:
                im.paste((244, 242, 246), (0, 0), m.point(lambda v: int(v * 0.96 * a)))
    return np.asarray(im)


FONT_ROLE = ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-LightItalic.otf', 44)


def role_v(canvas, n):
    a0, a1 = CUT[4] + 4, CUT[7] - 4
    if not (a0 <= n < a1): return canvas
    a = min(1, (n - a0) / 9) * min(1, (a1 - n) / 9)
    dy = (1 - min(1, (n - a0) / 9)) * 10
    img = neon(canvas, 'The Driver', 104, VW / 2, BAND_Y + BAND_H + 118 + dy, a)
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    txt = 'Drive (2011)  ·  Nicolas Winding Refn'
    bb = d.textbbox((0, 0), txt, font=FONT_ROLE)
    v = int(225 * a)
    d.text(((VW - (bb[2] - bb[0])) / 2 - bb[0], BAND_Y + BAND_H + 200 + dy), txt, font=FONT_ROLE, fill=(v, v, int(v * 1.02) if v < 250 else v))
    return np.asarray(im)


def tag_v(n):
    k = n - END_F - round(0.5 * FPS)
    img = np.zeros((VH, VW, 3), np.uint8)
    if k < 0: return img
    a = min(1, k / 8) * min(1, (TAIL - round(0.5 * FPS) - k) / 10)
    if a <= 0: return img
    im = Image.fromarray(img); d = ImageDraw.Draw(im); txt = '@editfrmn'
    f = ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-BlackItalic.otf', 52)
    bb = d.textbbox((0, 0), txt, font=f); v = int(165 * a)
    d.text(((VW - bb[2]) / 2 - bb[0], BAND_Y + BAND_H / 2 - (bb[3] - bb[1]) / 2 - bb[1]), txt, font=f, fill=(v, v, v))
    return np.asarray(im)


def tag(n):
    k = n - END_F - round(0.5 * FPS)
    img = np.zeros((H, W, 3), np.uint8)
    if k < 0: return img
    a = min(1, k / 8) * min(1, (TAIL - round(0.5 * FPS) - k) / 10)
    if a <= 0: return img
    im = Image.fromarray(img); d = ImageDraw.Draw(im); txt = '@editfrmn'
    bb = d.textbbox((0, 0), txt, font=FONT_TAG); v = int(160 * a)
    d.text(((W - bb[2]) / 2, H * 0.5 - bb[3] / 2), txt, font=FONT_TAG, fill=(v, v, v))
    return np.asarray(im)


def audio(path):
    dur = TOTAL / FPS
    drop = bt(10)
    fc = (f'[0:a]atrim={T0}:{T0 + dur},asetpts=PTS-STARTPTS[s];'
          f'[s]asplit[s1][s2];'
          f'[s1]atrim=0:{drop},lowpass=f=650,volume=0.55,afade=t=out:st={drop - 0.04}:d=0.04[h];'
          f'[s2]atrim={drop}:{dur},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.02[c];'
          f'[h][c]concat=n=2:v=0:a=1[m];'
          f'[1:a]atrim={DLG_IN}:{DLG_IN + bt(10) - 0.2},asetpts=PTS-STARTPTS,highpass=f=90,volume=2.2,'
          f'afade=t=out:st={bt(10) - 0.6}:d=0.35,adelay={int(DLG_AT * 1000)}|{int(DLG_AT * 1000)}[v];'
          f'[m][v]amix=inputs=2:normalize=0,afade=t=out:st={END_F / FPS - 0.2}:d=2.3,'
          f'loudnorm=I=-15.3:TP=-1.5:LRA=9[a]')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', SONG, '-i', src('opening'), '-filter_complex', fc,
                    '-map', '[a]', '-ar', '48000', '-ac', '2', path], check=True)


def main():
    clips = []
    for s, a, b in zip(SHOTS, CUT[:-1], CUT[1:]):
        clips.append((a, b, load(s, b - a)))
    if not PREVIEW:
        os.makedirs(D + '/out', exist_ok=True)
        wav = D + '/out/mix.wav'; audio(wav)
        enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', (f'{VW}x{VH}' if VERT else f'{W}x{H}'),
                                '-r', f'{FPS_N}/{FPS_D}', '-i', '-', '-i', wav, '-map', '0:v', '-map', '1:a',
                                '-c:v', 'libx264', '-preset', 'slow', '-tune', 'film', '-crf', ('15' if VERT else '14'), '-pix_fmt', 'yuv420p',
                                '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                                '-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', '-shortest', OUT],
                               stdin=subprocess.PIPE)
    pick = []
    if PREVIEW:
        arg = sys.argv[sys.argv.index('--preview') + 1]
        if arg == 'auto':
            pick = sorted(set([(a + b) // 2 for a, b in zip(CUT[:-1], CUT[1:])] + [DROP_F, DROP_F + 4, L3_F, L3_F + 1]
                              + [round((c0 + 0.05) * FPS) for c0, c1, w in CAPS[::3]] + [END_F + 30]))
        else:
            pick = [int(v) for v in arg.split(',')]
    prev = {}
    for n in range(TOTAL):
        if n >= END_F:
            img = tag(n)
            fade = max(0.0, 1 - (n - END_F) / max(1, round(0.5 * FPS)))
            if fade > 0:
                last = clips[-1][2][-1]
                img = np.maximum(img, (grade(last, n) * 255 * fade).astype(np.uint8))
        else:
            ci = max(i for i, (a, b, _) in enumerate(clips) if a <= n)
            a, b, fr = clips[ci]
            img = warp_fx(fr[n - a], n)
            x = grade(img, n)
            x = chroma(leak(x, n), n)
            if n in FLASH: x = x * (1 - FLASH[n]) + FLASH[n]
            img = finish(x, n)
            img = credit(img, n)
            if DROP_F <= n < CUT[4]:                      # title only on the scorpion jacket
                img = title(img, n - DROP_F, CUT[4] - DROP_F)
            t = n / FPS
            if not VERT:
                for c0, c1, w in CAPS:
                    if c0 <= t < c1: img = caption(img, w)
        if VERT:
            canvas = np.zeros((VH, VW, 3), np.uint8)
            if n >= END_F:
                canvas = tag_v(n)
                fade = max(0.0, 1 - (n - END_F) / max(1, round(0.5 * FPS)))
                if fade > 0:
                    small = cv2.resize(img, (VW, BAND_H), interpolation=cv2.INTER_AREA)
                    band = canvas[BAND_Y:BAND_Y + BAND_H]
                    canvas[BAND_Y:BAND_Y + BAND_H] = np.maximum(band, small)
            else:
                canvas[BAND_Y:BAND_Y + BAND_H] = cv2.resize(img, (VW, BAND_H), interpolation=cv2.INTER_AREA)
                canvas = captions_v(canvas, n)
            img = canvas
        if PREVIEW:
            if n in pick: prev[n] = img
        else:
            enc.stdin.write(np.ascontiguousarray(img).tobytes())
    if PREVIEW:
        TW, TH = (270, 480) if VERT else (320, 180)
        th = [cv2.resize(prev[p], (TW, TH), interpolation=cv2.INTER_AREA) for p in pick]
        for p, t_ in zip(pick, th): cv2.putText(t_, f'{p} {p / FPS:.2f}', (4, 16), 0, 0.5, (255, 255, 0), 1)
        cols = 6; rows = (len(th) + cols - 1) // cols
        sheet = np.zeros((rows * TH, cols * TW, 3), np.uint8)
        for i, t_ in enumerate(th): sheet[(i // cols) * TH:(i // cols + 1) * TH, (i % cols) * TW:(i % cols + 1) * TW] = t_
        Image.fromarray(sheet).save(D + '/sheets/preview.jpg', quality=88)
        for p_ in pick: Image.fromarray(prev[p_]).save(D + f'/sheets/full_{p_}.png')
    else:
        enc.stdin.close(); enc.wait()
    print('cuts (s):', [round(c / FPS, 3) for c in CUT], 'total', round(TOTAL / FPS, 2), file=sys.stderr)


if __name__ == '__main__':
    main()
