"""Ryan Gosling - female edit. SNL laugh hook -> chorus; soft silver B&W, colour blooms on the final kiss;
Gosling's eyes locked to the same screen point across films (same-position cuts, ref #34) with 3-frame
micro-dissolves; a ping-pong of looks; one lowercase serif comment line; @editfrmn."""
import json, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
S = WORK + ''
D = S + '/gos_edit'; V = S + '/cc/gos/vid/'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # the fan-edit library; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extras as EX, faceid as FID, look_fem as L
FPS = 30
VW, VH, BAND_Y, BAND_H = 1080, 1920, 616, 608
BW = 1080
PREVIEW = '--preview' in sys.argv
OUT = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else D + '/out/gosling_fem.mp4'
SONG = D + '/apocalypse.wav'
BEATS = np.load(D + '/beats.npy')
T0 = float(BEATS[np.argmin(abs(BEATS - 112.20))])               # song time at output 0 (a beat, 5 beats before the chorus)
OB = BEATS[BEATS >= T0 - 1e-6] - T0
def bt(k): return float(OB[k])
REF = np.load(S + '/cc/gos/gosling_ref.npy')
EYE_T = (0.50, 0.40)                                          # where his eyes sit in the band, all edit long
FW_T = 0.26                                                   # his face width in the band (fraction)
from shots_fem import SHOTS, PINGPONG, HOOK                   # filled after shot selection
WORK = os.environ.get('FAN_EDIT_WORK', os.path.expanduser('~/fan-edits'))   # work folder: sources, models, outputs

def src_info(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate',
                        '-of', 'csv=p=0', path], capture_output=True, text=True).stdout.strip().split(',')
    return int(r[0]), int(r[1]), eval(r[2])

def active_area(path, t):
    out = subprocess.run(['ffmpeg', '-ss', str(max(0, t - 0.3)), '-t', '1.5', '-i', path, '-vf', 'cropdetect=24:2:0',
                          '-f', 'null', '-'], capture_output=True, text=True).stderr
    cr = [tuple(map(int, l.split('crop=')[1].split()[0].split(':'))) for l in out.splitlines() if 'crop=' in l]
    return max(cr, key=lambda c: c[0] * c[1])

def read(path, t, dur):
    """frames of the active picture (letterbox bars removed); low-bitrate clip.cafe encodes get a mild
    spatio-temporal denoise so their compression blocks do not survive the B&W contrast"""
    W, H, fps = src_info(path)
    aw, ah, ax, ay = active_area(path, t)
    vf = f'crop={aw}:{ah}:{ax}:{ay}' + (',hqdn3d=1.6:1.3:4:3' if '/cc/gos/vid/' in path else '')
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.3f}', '-i', path, '-t', f'{dur + 0.3:.3f}', '-an', '-vf', vf,
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    print(f'     src {W}x{H} active {aw}x{ah}', file=sys.stderr)
    return np.frombuffer(raw, np.uint8).reshape(-1, ah, aw, 3), fps

def eyes(frame, who='gosling'):
    """eye midpoint (px) and eye distance (px) of Gosling (or the biggest other face)"""
    H, W = frame.shape[:2]; sc = 960 / W
    sm = cv2.resize(frame, (960, int(H * sc)), interpolation=cv2.INTER_AREA); bgr = cv2.cvtColor(sm, cv2.COLOR_RGB2BGR)
    best = None
    for q in FID.faces(bgr, 0.5):
        sim = max(FID.cos(FID.embed(bgr, q), r) for r in REF)
        ok = sim > 0.33 if who == 'gosling' else sim < 0.25
        if ok and (best is None or q[2] > best[2]): best = q
    if best is None: return None
    (rx, ry), (lx, ly) = best[4:6], best[6:8]
    return np.array([(rx + lx) / 2 / sc, (ry + ly) / 2 / sc]), float(best[2] / sc)      # eyes (px), face width (px)

def track(frames, who, step=3):
    """smoothed eye track over the shot (px), holes interpolated; None if never found"""
    idx = list(range(0, len(frames), step)) + [len(frames) - 1]; pts = []
    for i in idx:
        e = eyes(frames[i], who)
        if e: pts.append((i, e[0][0], e[0][1], e[1]))
    if not pts: return None
    P = np.array(pts); n = len(frames); t = np.arange(n)
    xs, ys, ds = (np.interp(t, P[:, 0], P[:, k]) for k in (1, 2, 3))
    k = 9
    sm = lambda v: np.convolve(np.pad(v, (k, k), mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), mode='same')[k:-k]
    return sm(xs), sm(ys), sm(ds)

def framed(frames, fps_src, n, spec):
    """n output frames of a shot: speed (optical-flow slow-mo), eye-locked 16:9 crop with a slow push-in"""
    sp = spec.get('speed', 1.0); who = spec.get('who', 'gosling'); size = spec.get('size', FW_T)
    tr = track(frames, who) if spec.get('lock', True) else None
    H, W = frames[0].shape[:2]
    maxcw = min(W, H * 16 / 9); mincw = min(BW, maxcw)              # never upscale
    if spec.get('lock', True): print(f"     lock {'ok' if tr is not None else 'FAILED'}", file=sys.stderr)
    out = []
    for i in range(n):
        x = i / FPS * sp * fps_src; i0 = min(int(x), len(frames) - 2); a = x - i0
        f = EX.interp(frames[i0], frames[i0 + 1], a) if sp < 0.99 and a > 0.02 else frames[min(int(round(x)), len(frames) - 1)]
        u = i / max(1, n - 1); push = 1 + spec.get('push', 0.035) * (u * u * (3 - 2 * u))
        if tr is not None:
            j = min(int(round(x)), len(frames) - 1)
            ex, ey, fw = tr[0][j], tr[1][j], tr[2][j]
            cs = min(fw / size, maxcw)                                     # width giving the wanted face size
            reach = min(ex / EYE_T[0], (W - ex) / (1 - EYE_T[0]), ey / EYE_T[1] * 16 / 9, (H - ey) / (1 - EYE_T[1]) * 16 / 9)
            cw = float(np.clip(min(cs, max(reach, cs * 0.7)), mincw, maxcw)) / push   # position first, size second
        else:
            ex, ey = spec.get('cx', 0.5) * W, spec.get('cy', 0.5) * H; cw = maxcw / spec.get('zoom', 1.0) / push
            cw = max(cw, mincw)
        ch = cw * 9 / 16
        x0 = float(np.clip(ex - EYE_T[0] * cw, 0, W - cw)); y0 = float(np.clip(ey - EYE_T[1] * ch, 0, H - ch))
        s = BW / cw
        g = cv2.GaussianBlur(f, (0, 0), 0.42 / s) if s < 0.9 else f      # prefilter: clean downscale, no shimmer
        M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
        out.append(cv2.warpAffine(g, M, (BW, BAND_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
    return levels(out, spec.get('bright', 0.40))

def levels(frames, mid=0.40):
    """per-shot exposure match: black/white points and a gamma that brings the median luma to `mid`"""
    smp = np.concatenate([cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).ravel()[::5] for i in (0, len(frames) // 2, len(frames) - 1)]).astype(np.float32) / 255
    lo, md, hi = np.percentile(smp, 2), np.percentile(smp, 50), np.percentile(smp, 99.2)
    lo = min(lo, 0.06); hi = max(hi, 0.55)
    m = np.clip((md - lo) / (hi - lo), 0.05, 0.95)
    g = float(np.clip(np.log(mid) / np.log(m), 0.72, 1.25))   # do not dig into compressed shadows
    x = np.arange(256, dtype=np.float32) / 255
    y = np.clip((x - lo) / (hi - lo), 0, 1) ** g * 0.94 + 0.012
    lut = (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)
    print(f'     levels lo {lo:.2f} med {md:.2f} hi {hi:.2f} gamma {g:.2f}', file=sys.stderr)
    return [cv2.LUT(f, lut) for f in frames]


# ---------------- timeline ----------------
DROP_K = 5                                                    # song beat where the chorus enters (output 3.18 s)
def build():
    """returns list of (start_frame, end_frame, frames, kind) and the cut list"""
    segs = []; f0 = 0
    # hook: SNL laugh, live sound
    fr, fps = read(HOOK['src'], HOOK['t'], bt(DROP_K) + 0.2)
    n = round(bt(DROP_K) * FPS)
    segs.append((0, n, framed(fr, fps, n, HOOK), 'hook')); f0 = n
    k = DROP_K
    for spec in SHOTS:
        if spec.get('pingpong'):
            nb = spec['beats']; n = round(bt(k + nb) * FPS) - f0
            A, B = PINGPONG
            fa, fpa = read(A['src'], A['t'], n / FPS * A.get('speed', 1) + 0.2); fb, fpb = read(B['src'], B['t'], n / FPS * B.get('speed', 1) + 0.2)
            ga, gb = framed(fa, fpa, n, A), framed(fb, fpb, n, B)
            run = spec.get('run', 8); out = []
            for i in range(n): out.append((ga if (i // run) % 2 == 0 else gb)[i])
            segs.append((f0, f0 + n, out, 'pingpong')); f0 += n; k += nb; continue
        nb = spec['beats']; n = round(bt(k + nb) * FPS) - f0
        fr, fps = read(spec['src'], spec['t'], n / FPS * spec.get('speed', 1) + 0.2)
        segs.append((f0, f0 + n, framed(fr, fps, n, spec), spec.get('kind', 'shot'))); f0 += n; k += nb
        print(f"  {os.path.basename(spec['src'])[:48]} @{spec['t']} {nb}b -> {n} fr", file=sys.stderr, flush=True)
    return segs, f0

# ---------------- text ----------------
FONT_LINE = ImageFont.truetype(D + '/fonts/Cormorant_Garamond_ital_wght_1_500.ttf', 64)
FONT_TAG = ImageFont.truetype(D + '/fonts/Cormorant_Garamond_ital_wght_1_400.ttf', 50)
LINE = 'the way he looks at her'
TAG = '@editfrgrl'                                            # female edits sign as @editfrgrl, male as @editfrmn
_TXT = {}
def glow_text(txt, font, a, colour=(255, 250, 244), glow=(255, 236, 214), gr=10, ga=0.55):
    key = (txt, id(font))
    if key not in _TXT:
        d = ImageDraw.Draw(Image.new('L', (8, 8))); bb = d.textbbox((0, 0), txt, font=font)
        w, h = bb[2] - bb[0] + 4 * gr, bb[3] - bb[1] + 4 * gr
        m = Image.new('L', (w, h), 0); ImageDraw.Draw(m).text((2 * gr - bb[0], 2 * gr - bb[1]), txt, font=font, fill=255)
        core = np.asarray(m, np.float32) / 255; halo = np.asarray(m.filter(ImageFilter.GaussianBlur(gr)), np.float32) / 255
        _TXT[key] = (core, halo)
    core, halo = _TXT[key]
    rgb = np.zeros(core.shape + (3,), np.float32); al = np.clip(core + halo * ga, 0, 1)
    rgb[:] = np.array(glow, np.float32); rgb = rgb * (1 - core[..., None]) + np.array(colour, np.float32) * core[..., None]
    return rgb, al * a

def paste(canvas, rgb, al, cx, cy):
    h, w = al.shape; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    reg = canvas[y0:y0 + h, x0:x0 + w].astype(np.float32)
    canvas[y0:y0 + h, x0:x0 + w] = np.clip(reg * (1 - al[..., None]) + rgb * al[..., None], 0, 255).astype(np.uint8)

def line_alpha(n, end_f):
    d0 = round(bt(DROP_K) * FPS) + 6                             # appears just after the drop
    a = np.clip((n - d0) / 14, 0, 1) * np.clip((end_f - 10 - n) / 14, 0, 1)
    return float(a * a * (3 - 2 * a))


# ---------------- compositing ----------------
TAIL = round(2.3 * FPS)
def band_frame(segs, n, end_f):
    """graded band image for output frame n: 3-frame micro-dissolve across every cut after the drop"""
    ci = max(i for i, s in enumerate(segs) if s[0] <= n)
    a, b, fr, kind = segs[ci]
    x = fr[min(n - a, len(fr) - 1)]
    if ci + 1 < len(segs) and segs[ci + 1][3] != 'hook' and n >= b - 1 and kind != 'hook':
        nx = segs[ci + 1][2][0]; x = cv2.addWeighted(x, 0.62, nx, 0.38, 0)     # frame before the cut leans in
    if ci > 0 and segs[ci - 1][3] != 'hook' and n - a <= 1 and kind != 'hook':
        pv = segs[ci - 1][2][-1]; w = 0.45 if n == a else 0.18
        x = cv2.addWeighted(x, 1 - w, pv, w, 0)                                  # first two frames keep a ghost
    bw = L.silver(x, n)
    if kind == 'finale':                                                         # colour blooms into the kiss
        u = (n - a) / max(1, b - a); t = float(np.clip((u - 0.22) / 0.45, 0, 1)); t = t * t * (3 - 2 * t)
        if t > 0: bw = cv2.addWeighted(bw, 1 - t, L.colour_soft(x, n), t, 0)
    if kind == 'hook' and n < 6:                                                 # open from black
        bw = (bw.astype(np.float32) * (n + 1) / 7).astype(np.uint8)
    if n >= end_f - 12:                                                          # into black before the tag
        k = (end_f - n) / 12; bw = (bw.astype(np.float32) * max(0, k)).astype(np.uint8)
    return bw

def compose(segs, n, end_f):
    canvas = np.zeros((VH, VW, 3), np.uint8)
    if n < end_f:
        canvas[BAND_Y:BAND_Y + BAND_H] = band_frame(segs, n, end_f)
        a = line_alpha(n, end_f)
        if a > 0:
            rgb, al = glow_text(LINE, FONT_LINE, a * 0.92); paste(canvas, rgb, al, VW / 2, BAND_Y + BAND_H + 104)
    else:
        k = n - end_f - 8
        a = np.clip(k / 12, 0, 1) * np.clip((TAIL - 8 - k) / 12, 0, 1)
        if a > 0:
            rgb, al = glow_text(TAG, FONT_TAG, float(a) * 0.62, gr=12, ga=0.7); paste(canvas, rgb, al, VW / 2, BAND_Y + BAND_H / 2)
    return canvas

# ---------------- audio ----------------
def audio(path, end_f):
    dur = (end_f + TAIL) / FPS; drop = bt(DROP_K); end = end_f / FPS
    fc = (f'[0:a]atrim={T0}:{T0 + dur},asetpts=PTS-STARTPTS[s];[s]asplit[s1][s2];'
          f'[s1]atrim=0:{drop},lowpass=f=900,volume=0.42,afade=t=out:st={drop - 0.05}:d=0.05[h];'
          f'[s2]atrim={drop}:{dur},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.03[c];[h][c]concat=n=2:v=0:a=1[m];'
          f'[1:a]atrim={HOOK["t"]}:{HOOK["t"] + drop + 0.25},asetpts=PTS-STARTPTS,highpass=f=80,volume=1.25,'
          f'afade=t=in:st=0:d=0.12,afade=t=out:st={drop - 0.08}:d=0.3[v];'
          f'[m][v]amix=inputs=2:normalize=0,afade=t=out:st={end - 1.6}:d={1.6 + TAIL / FPS - 0.3},'
          f'loudnorm=I=-14.7:TP=-1.5:LRA=9[a]')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', SONG, '-i', HOOK['src'], '-filter_complex', fc,
                    '-map', '[a]', '-ar', '48000', '-ac', '2', path], check=True)

def main():
    segs, end_f = build()
    cuts = [s[0] for s in segs]
    print('cuts (s):', [round(c / FPS, 2) for c in cuts], 'end', round(end_f / FPS, 2), file=sys.stderr)
    total = end_f + TAIL
    if PREVIEW:
        arg = sys.argv[sys.argv.index('--preview') + 1]
        pick = sorted(set([c + 4 for c in cuts] + [c - 1 for c in cuts[1:]] + [end_f - 20, end_f - 2, end_f + 40])) if arg == 'auto' else [int(v) for v in arg.split(',')]
        th = []
        for p in pick:
            img = compose(segs, p, end_f)[BAND_Y - 40:BAND_Y + BAND_H + 200]
            t_ = cv2.resize(img, (360, int(360 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.putText(t_, f'{p} {p / FPS:.2f}', (4, 16), 0, 0.5, (255, 255, 0), 1); th.append(t_)
        cols = 5; rows = (len(th) + cols - 1) // cols; TH, TW = th[0].shape[:2]
        sheet = np.zeros((rows * TH, cols * TW, 3), np.uint8)
        for i, t_ in enumerate(th): sheet[(i // cols) * TH:(i // cols + 1) * TH, (i % cols) * TW:(i % cols + 1) * TW] = t_
        Image.fromarray(sheet).save(D + '/sheets/preview.jpg', quality=88); return
    os.makedirs(D + '/out', exist_ok=True); wav = D + '/out/mix.wav'; audio(wav, end_f)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{VW}x{VH}', '-r', str(FPS),
                            '-i', '-', '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'slow',
                            '-crf', '13', '-tune', 'grain', '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                            '-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', '-shortest', OUT], stdin=subprocess.PIPE)
    for n in range(total): enc.stdin.write(np.ascontiguousarray(compose(segs, n, end_f)).tobytes())
    enc.stdin.close(); enc.wait()

if __name__ == '__main__':
    main()
