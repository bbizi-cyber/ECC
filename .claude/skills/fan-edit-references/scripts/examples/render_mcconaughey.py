"""Matthew McConaughey edit — rebuilds the timing/structure of the reference edit."""
import subprocess, sys, json, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # the fan-edit library
import fx as FXM
import look as LOOKM
import extras as EX
WORK = os.environ.get('FAN_EDIT_WORK', os.path.expanduser('~/fan-edits'))   # work folder: sources, models, outputs

S = WORK + ''
DL = S + '/dl'
VERT = '--vertical' in sys.argv
W, H, FPS = (1080, 1920, 30) if VERT else (1920, 1080, 30)   # horizontal is now 16:9
K = H / 834 if not VERT else 1.0
FXM.K = K; EX.K = K
TOTAL = 660
PREVIEW = '--preview' in sys.argv
OUTFILE = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else WORK + '/out/final.mp4'

# active picture areas (letterbox removed): w,h,x,y
ACTIVE = {
    'x7qz1vb': (1920, 800, 0, 140),   # The Gentlemen (1080p)
    'x2bt5yr': (1280, 536, 0, 4),     # The Wolf of Wall Street
    'x86ljwh': (1920, 800, 0, 140),   # Dallas Buyers Club (1080p)
    'x81klvj': (848, 352, 0, 4),      # The Lincoln Lawyer
    'x7xm3jc': (848, 352, 0, 4),      # Gold
    'x7u8hdi': (848, 470, 0, 5),      # True Detective
    'x55pf9r': (512, 288, 0, 0),      # 'This is Lincoln' ad (buzz60 mark top-right: keep cx <= 0.42)
    'x2dozke': (848, 480, 0, 0),      # Lincoln MKZ ad
    'x1ee12z': (848, 480, 0, 0),      # Oscars 2014
    'x7xj1gh': (512, 216, 0, 36),     # Serenity
    'x18ejsg': (1280, 544, 0, 0),     # Interstellar
    # true-HD clips (clip.cafe) - no upscaling beyond 1.35x anywhere
    'wolf_name': (2592, 1080, 0, 0), 'wolf_money': (1920, 800, 0, 140),
    'dt26': (1920, 800, 0, 0), 'dt60': (1920, 800, 0, 140),
    'ser2': (2570, 1076, 0, 2), 'ser119': (2570, 1076, 0, 2), 'ser94': (2570, 1076, 0, 2), 'ser101': (2570, 1076, 0, 2),
    'sah1': (1920, 1080, 0, 0), 'sah17': (1920, 1080, 0, 0),
    'int88': (3840, 1600, 0, 280), 'int97': (3840, 2160, 0, 0), 'int11': (3840, 1600, 0, 280),
    'int43': (3840, 2160, 0, 0), 'int19': (3840, 1600, 0, 280),
    'wolf_rule': (1920, 800, 0, 0),
}

# name: (src, start, speed, cx, zoom)   speed<1 = slow motion; cx = horizontal centre (0..1)
SHOTS = json.load(open(S + '/tools/shots.json'))

# timeline: (start_frame, end_frame, shot)
A = [(0, 14, 'A1'), (14, 29, 'A2'), (29, 56, 'A3'), (56, 81, 'A4'), (81, 105, 'A5'),
     (105, 138, 'A6'), (138, 164, 'A7'), (164, 188, 'A8')]
B1 = [(189, 213, 'B1'), (210, 230, 'B2'), (230, 253, 'B3'), (253, 269, 'B4'), (269, 294, 'B5'),
      (294, 317, 'B6'), (317, 333, 'B7'), (333, 357, 'B8'), (357, 382, 'B9'), (376, 403, 'B10')]
B2 = [(403, 427, 'B1'), (424, 445, 'C2'), (445, 468, 'C3'), (468, 484, 'C4'), (484, 509, 'C5'),
      (509, 532, 'C6'), (532, 548, 'C7'), (548, 570, 'C8'), (570, 597, 'B9'), (591, 618, 'B10')]   # second loop: new looks
TIMELINE = A + B1 + B2

# special frames
WHITE = {382, 597}
BLACK = {188, 402}                            # one black frame right before each gunshot drop                       # full white flash frames
BOX = {209: 'B2', 375: 'B10', 423: 'B2', 590: 'B10'}     # small white box frame before an inset transition
FREEZE = {283: (294, None), 605: (618, None)}
OUTLINE = {605}   # both freezes start on a beat (9.43 s, 20.18 s)      # start: (end, keep people right of x-fraction)
GLITCH = {172, 173}


def bar_height(n):
    full = 0.12 * H if not VERT else 0.06 * H
    if n < 186: return full
    if n < 192: return full * (1 - FXM.ease_out((n - 186) / 6))
    if n >= 606: return full * FXM.ease_io((n - 606) / 6)
    return 0


INSET = set(range(210, 213)) | set(range(376, 382)) | set(range(424, 427)) | set(range(591, 597))

# captions: (from, to, text, style)  style: n = normal, big-colour = accent pop
CAPS = [(0, 1, 'WHY', 'bigw'), (2, 5, 'WHY', 'n'), (6, 10, 'WHY DO', 'n'), (11, 20, 'WHY DO ALL', 'n'),
        (21, 25, 'THESE', 'n'), (26, 28, 'B*TCHES', 'red'), (29, 38, 'THESE B*TCHES', 'n'),
        (39, 43, 'KEEP', 'n'), (44, 51, 'KEEP ON', 'n'), (52, 65, "KEEP ON STRESSIN'", 'n'),
        (66, 72, 'OUT', 'n'), (73, 80, 'OUT MY', 'n'), (81, 83, 'HEAD', 'yellow'), (84, 104, 'OUT MY HEAD', 'n'),
        (105, 113, 'SHE', 'n'), (114, 119, 'SHE CAN', 'n'), (120, 126, 'SHE CAN SEE', 'n'),
        (127, 131, "I'M", 'n'), (132, 135, 'WORKING', 'orange'), (136, 146, "I'M WORKING", 'n'),
        (147, 151, 'FOR', 'n'), (152, 159, 'FOR THAT', 'n'), (160, 171, 'MOTHER-', 'n'),
        (172, 174, 'MOTHERF*CKING', 'red'), (175, 187, 'MOTHERF*CKING', 'n')]
COLORS = {'n': (255, 255, 255), 'bigw': (255, 255, 255), 'red': (235, 20, 20),
          'yellow': (255, 226, 20), 'orange': (255, 140, 0)}
F_N = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Bold.otf', 29)
F_BIG = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf', 62)
F_SIG = ImageFont.truetype('/usr/share/fonts/opentype/inter/InterDisplay-BlackItalic.otf', 46)


SR_SRC = {'x81klvj', 'x7xm3jc', 'x7u8hdi', 'x2bt5yr', 'x55pf9r', 'x2dozke', 'x1ee12z', 'x7xj1gh', 'x18ejsg'}   # low-res trailers -> AI upscale
USE_SR = '--nosr' not in sys.argv
PREP = {}


def prep(src, f, x_lo, x_hi, fscale):
    key = (src, hash(f.tobytes()), x_lo, x_hi, round(fscale, 4))
    if key in PREP: return PREP[key]
    crop = np.ascontiguousarray(f[:, x_lo:x_hi])
    k = 1
    if USE_SR and src in SR_SRC:
        import sr
        crop = sr.upscale(crop); k = 4
    tw, th = int(round((x_hi - x_lo) * fscale)), int(round(f.shape[0] * fscale))
    interp = cv2.INTER_AREA if tw < crop.shape[1] else cv2.INTER_LANCZOS4
    out = cv2.resize(crop, (tw, th), interpolation=interp)
    PREP[key] = out
    return out


# time remapping: speed ramp (slow -> fast, total source seconds) or smooth slow-mo, both flow-interpolated
SPEED = {'B4': ('ramp', 0.50), 'B8': ('slow', 0.50), 'B1': ('slow', 0.2, 'blend0'), 'B6': ('slow', 0.5)}
VCX = {'B1': [(0, 0.27), (4, 0.27), (10, 0.66)], 'B2': 0.22, 'C2': 0.76, 'C5': 0.66, 'B8': 0.43, 'B6': 0.47, 'A6': 0.37}
SHOT_TONE = {'B2': (0.07, 0.78), 'C5': (0.03, 0.8)}   # (black crush, highlight gain above 0.6)
WMZOOM = {'x55pf9r': 1.3}   # 16:9 crop of a 16:9 source must zoom to clear the buzz60 mark
WATERMARK = {'x55pf9r': (0.79, 0.13)}   # src: keep the crop's right edge left of x-frac whenever it includes rows above y-frac


def native_fps(src):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate',
                        '-of', 'csv=p=0', f'{DL}/{src}.mp4'], capture_output=True, text=True).stdout.strip()
    a, b = r.split()[0].strip(',').split('/'); return float(a) / float(b)


def remapped_frames(src, start, aw, ah, ax, ay, nframes, spec):
    kind, val = spec[0], spec[1]
    blend0 = len(spec) > 2 and spec[2] == 'blend0'   # crossfade (not flow) out of the muzzle-flash frame
    if kind == 'ramp':
        u = np.arange(nframes) / max(1, nframes - 1)
        sp = 0.28 + 1.9 * np.clip((u - 0.4) / 0.45, 0, 1) ** 2 * (3 - 2 * np.clip((u - 0.4) / 0.45, 0, 1))
        st = np.concatenate([[0], np.cumsum(sp[:-1])]); st = st / st[-1] * val
    else:
        st = np.arange(nframes) / FPS * val
    nf = native_fps(src)
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(start), '-i', f'{DL}/{src}.mp4', '-t', f'{st[-1] + 0.3:.3f}',
           '-vf', f'crop={aw}:{ah}:{ax}:{ay}', '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    nat = np.frombuffer(raw, np.uint8).reshape(-1, ah, aw, 3)
    out = []
    for t in st:
        x = t * nf; i0 = min(int(x), len(nat) - 2)
        if blend0 and i0 == 0:
            out.append(cv2.addWeighted(nat[0], 1 - (x - i0), nat[1], x - i0, 0))
        else:
            out.append(EX.interp(nat[i0], nat[i0 + 1], x - i0))
    return out


def load_shot(name, nframes):
    PREP.clear()
    src, start, speed, cx, zoom = SHOTS[name]
    aw, ah, ax, ay = ACTIVE[src]
    if name in SPEED:
        fr = remapped_frames(src, start, aw, ah, ax, ay, nframes, SPEED[name])
    else:
        dur = nframes / FPS * speed + 0.2
        cmd = ['ffmpeg', '-v', 'error', '-ss', str(start), '-i', f'{DL}/{src}.mp4', '-t', f'{dur:.3f}',
               '-vf', f'crop={aw}:{ah}:{ax}:{ay},setpts=PTS/{speed},fps={FPS}', '-an',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(-1, ah, aw, 3)

    if VERT and name in VCX:            # hand-set vertical framing (constant or keyframes, eased)
        v = VCX[name]
        if isinstance(v, (int, float)): cxs = np.full(nframes, float(v))
        else:
            ks = [k for k, _ in v]; vs = [c for _, c in v]
            u = np.arange(nframes, dtype=float)
            cxs = np.interp(u, ks, vs)
            for (k0, c0), (k1, c1) in zip(v, v[1:]):       # ease each segment
                m = (u >= k0) & (u <= k1)
                if k1 > k0: t = (u[m] - k0) / (k1 - k0); cxs[m] = c0 + (c1 - c0) * (t * t * (3 - 2 * t))
    elif VERT:
        cxs = EX.face_track(fr, nframes, cx)
    else:
        cxs = np.full(nframes, cx)

    def window(i):
        z = max(zoom, WMZOOM.get(src, 1.0) if not VERT else 1.0) * (1.0 + 0.035 * i / max(1, nframes - 1))      # gentle push-in
        ch = ah / z; cw = ch * W / H
        if cw > aw: cw = aw; ch = cw * H / W
        y0 = (ah - ch) / 2
        hi = aw - cw
        if src in WATERMARK and y0 < WATERMARK[src][1] * ah:
            hi = min(hi, WATERMARK[src][0] * aw - cw)
        x0 = float(np.clip(cxs[i] * aw - cw / 2, 0, max(0, hi)))
        return x0, y0, cw, ch

    wins = [window(i) for i in range(nframes)]
    ch0 = wins[0][3]
    x_lo = max(0, int(min(w[0] for w in wins)) - 6)
    x_hi = min(aw, int(np.ceil(max(w[0] + w[2] for w in wins))) + 6)
    fscale = H / ch0
    out = []
    for i in range(nframes):
        f = fr[min(i, len(fr) - 1)]
        x0, y0, cw, ch = wins[i]
        p = prep(src, f, x_lo, x_hi, fscale)
        sx, sy = (W / cw) / fscale, (H / ch) / fscale
        M = np.float32([[sx, 0, (x_lo - x0) * W / cw], [0, sy, -y0 * H / ch]])
        out.append(cv2.warpAffine(p, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT))
    out = EX.color_match(out)
    if name in SHOT_TONE: out = EX.tone(out, *SHOT_TONE[name])
    print(f'  shot {name}: {nframes} frames', file=sys.stderr, flush=True)
    return out


_FONTS = {}


def font_at(style, scale):
    base = (29 if style == 'n' else 62) * K
    size = max(8, int(round(base * scale)))
    key = (style == 'n', size)
    if key not in _FONTS:
        path = '/usr/share/fonts/opentype/inter/' + ('Inter-Bold.otf' if style == 'n' else 'Inter-ExtraBold.otf')
        _FONTS[key] = ImageFont.truetype(path, size)
    return _FONTS[key]


def caption(img, text, style, off=(0, 0), scale=1.0, yfrac=0.49):
    im = Image.fromarray(img)
    font = font_at(style, scale)
    d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x, y = (W - tw) / 2 - bb[0] + off[0], H * yfrac - th / 2 - bb[1] + off[1]
    sh = Image.new('L', im.size, 0)
    ImageDraw.Draw(sh).text((x + K, y + 2 * K), text, font=font, fill=235, stroke_width=int(round(3 * K)), stroke_fill=235)
    halo = sh.filter(ImageFilter.GaussianBlur(9 * K)).point(lambda v: int(v * 0.55))
    sh = sh.filter(ImageFilter.GaussianBlur(2.5 * K))
    im.paste((0, 0, 0), (0, 0), halo)
    im.paste((0, 0, 0), (0, 0), sh)
    sw = max(1, int(round((1.6 if style == 'n' else 2.2) * K)))
    ImageDraw.Draw(im).text((x, y), text, font=font, fill=COLORS[style], stroke_width=sw, stroke_fill=(20, 20, 20))
    return np.asarray(im)


def white_box(img, scale=0.53):
    out = img.copy()
    bw, bh = int(W * scale), int(H * scale)
    box = np.zeros((H, W), np.float32)
    box[(H - bh) // 2:(H + bh) // 2, (W - bw) // 2:(W + bw) // 2] = 1
    box = cv2.GaussianBlur(box, (0, 0), 6)[..., None]
    return (out * (1 - box) + 255 * box).astype(np.uint8)


def inset(bg, fg, scale=0.55):
    out = bg.copy()
    bw, bh = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    small = cv2.resize(fg, (bw, bh), interpolation=cv2.INTER_AREA)
    y, x = (H - bh) // 2, (W - bw) // 2
    out[y:y + bh, x:x + bw] = small
    return out


def main():
    clips = [load_shot(name, b - a) for a, b, name in TIMELINE]

    def active(n):
        return [(i, a) for i, (a, b, _) in enumerate(TIMELINE) if a <= n < b]

    pipe = None
    if not PREVIEW:
        music = '/root/.claude/uploads/f42b2dab-627e-5bb9-8d84-a82f0d26b256/4a106f91-v26044gc0000dbqdf0nog65le881b26g.mp4'
        tag = ("drawtext=fontfile=/usr/share/fonts/opentype/inter/InterDisplay-BlackItalic.otf:text='@editfrmn':"
               f"fontsize={34 if VERT else int(round(30 * K))}:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:"
               "alpha='if(lt(t,20.8),0,if(lt(t,21.1),(t-20.8)/0.3*0.6,if(lt(t,21.6),0.6,if(lt(t,21.95),(21.95-t)/0.35*0.6,0))))'")
        vf = 'unsharp=5:5:0.25:5:5:0,vignette=PI/5.5,noise=c0s=3:c0f=t,' + tag + ',format=yuv420p'
        sfxwav = S + '/out/sfx.wav'
        subprocess.run([sys.executable, '-I', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sfx.py'), sfxwav], check=True)
        pipe = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                 '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-i', music, '-i', sfxwav,
                                 '-filter_complex', f'[0:v]{vf}[v];[1:a]atrim=0:22,asetpts=PTS-STARTPTS,aresample=48000[m];[2:a]volume=0.9[fxa];'
                                 f'[m][fxa]amix=inputs=2:normalize=0,alimiter=limit=0.74:level=disabled,afade=t=out:st=21.0:d=1.0[a]',
                                 '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-preset', 'veryslow', '-tune', 'grain',
                                 '-crf', '15', '-profile:v', 'high', '-level', '4.2', '-colorspace', 'bt709',
                                 '-color_primaries', 'bt709', '-color_trc', 'bt709',
                                 '-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', '-t', '22',
                                 OUTFILE], stdin=subprocess.PIPE)
    prev = {}
    starts = sorted({a for a, b, _ in TIMELINE})
    hits = FXM.build_hits()
    fx = FXM.FX(W, H, hits, TIMELINE, whips=[230, 294, 468, 532], no_entry=[189, 403, 210, 376, 424, 591], zooms=[253, 317, 445, 570])
    lk = LOOKM.Look(W, H)
    frozen = {}
    cap_y = {}
    for n in range(TOTAL):
        fz = next(((f0, e, kr) for f0, (e, kr) in FREEZE.items() if f0 <= n < e), None)
        if n in WHITE:
            img = np.full((H, W, 3), 255, np.uint8)
        elif n in BLACK:
            img = np.zeros((H, W, 3), np.uint8)
        elif n >= 618:
            img = np.zeros((H, W, 3), np.uint8)
            alpha = 0  # end title removed
            if alpha > 0:
                im = Image.fromarray(img); d = ImageDraw.Draw(im)
                txt = 'McCONAUGHEY'
                bb = d.textbbox((0, 0), txt, font=F_SIG)
                v = int(235 * alpha)
                d.text(((W - bb[2]) / 2, H * 0.5 - bb[3] / 2), txt, font=F_SIG, fill=(v, v, v))
                img = np.asarray(im)
        else:
            act = active(n)
            if n in INSET:
                (i0, a0), (i1, a1) = sorted(act, key=lambda t: t[1])
                img = inset(clips[i0][n - a0], clips[i1][n - a1])
            else:
                i, a = max(act, key=lambda t: t[1]) if n not in BOX else min(act, key=lambda t: t[1])
                img = clips[i][n - a]
            if n in BOX:
                img = white_box(img)
            if fz:
                f0, e, kr = fz
                if f0 not in frozen:
                    frozen[f0] = (img.copy(), EX.person_mask(img, kr))
                src_img, mask = frozen[f0]
                img = EX.freeze_frame(src_img, mask, n - f0, e - f0, outline=f0 in OUTLINE)
        if n not in WHITE and n not in BLACK:
            img = fx.apply(img, n)
            if n < 618: img = lk.apply(img, n)
            if n in GLITCH: img = EX.glitch(img, n)
        img = EX.letterbox(img, bar_height(n))
        off = fx.caption_offset(n)
        if n < 189 and any(a <= n <= b for a, b, _, _ in CAPS):
            si = max((i for i, (a, b, _) in enumerate(TIMELINE) if a <= n < b), key=lambda i: TIMELINE[i][0])
            if si not in cap_y:            # one caption height per shot: centre, unless it would sit on a face
                boxes = EX.face_boxes(img)
                cy0, cy1 = 0.49 - 0.06, 0.49 + 0.06
                hit = any(not (y1 < cy0 or y0 > cy1) and x0 < 0.65 and x1 > 0.35 for x0, y0, x1, y1 in boxes)
                if hit:
                    low = max(y1 for x0, y0, x1, y1 in boxes) + 0.07
                    cap_y[si] = min(0.80, max(0.68, low))
                else:
                    cap_y[si] = 0.49
            yfrac = cap_y[si]
        else:
            yfrac = 0.49
        for a, b, text, style in CAPS:
            if a <= n <= b:
                k = n - a
                sc = ([1.35, 1.1, 1.0][min(k, 2)]) if style != 'n' else (1.12 if k == 0 else 1.0)
                img = caption(img, text, style, off, sc, yfrac)
        if PREVIEW:
            prev[n] = img
        else:
            pipe.stdin.write(np.ascontiguousarray(img).tobytes())
    if PREVIEW:
        pick = [int(x) for x in sys.argv[sys.argv.index('--preview') + 1].split(',')]
        TW, TH = 270, int(270 * H / W)
        th = [cv2.resize(prev[p], (TW, TH), interpolation=cv2.INTER_AREA) for p in pick]
        for p, t in zip(pick, th): cv2.putText(t, str(p), (4, 18), 0, 0.6, (255, 255, 0), 2)
        cols = 8; rows = (len(th) + cols - 1) // cols
        sheet = np.zeros((rows * TH, cols * TW, 3), np.uint8)
        for i, t in enumerate(th): sheet[(i // cols) * TH:(i // cols + 1) * TH, (i % cols) * TW:(i % cols + 1) * TW] = t
        Image.fromarray(sheet).save(S + '/frames/preview.jpg', quality=88)
    else:
        pipe.stdin.close(); pipe.wait()


if __name__ == '__main__':
    os.makedirs(S + '/out', exist_ok=True)
    main()
