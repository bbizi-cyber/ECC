"""Scan clips: shot boundaries + per-shot Gosling face stats (sampled), saved as JSON + mid-frame thumbs."""
import sys, os, glob, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); import faceid as F
ref = np.load(sys.argv[1]); files = sorted(glob.glob(sys.argv[2])); outdir = sys.argv[3]; os.makedirs(outdir, exist_ok=True)
res = []
for p in files:
    cap = cv2.VideoCapture(p); fps = cap.get(5) or 24; n = 0; prev = None; cuts = [0]; samples = []
    while True:
        ok, f = cap.read()
        if not ok: break
        s = cv2.resize(f, (64, 36)); h = cv2.calcHist([cv2.cvtColor(s, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]); cv2.normalize(h, h)
        if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA) > 0.33 and n - cuts[-1] > 4: cuts.append(n)
        prev = h
        if n % 4 == 0:
            sm = cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1])))
            fs = F.faces(sm); gs = []; others = 0
            for q in fs:
                sim = max(F.cos(F.embed(sm, q), r) for r in ref)
                if sim > 0.36: gs.append((sim, float(q[0] + q[2] / 2) / sm.shape[1], float(q[1] + q[3] / 2) / sm.shape[0], float(q[2]) / sm.shape[1], float(q[4] + q[6]) / 2 / sm.shape[1], float(q[5] + q[7]) / 2 / sm.shape[0]))
                else: others += 1
            g = max(gs, key=lambda t: t[3]) if gs else None
            samples.append((n, g, others, float(cv2.Laplacian(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()), float(sm.mean())))
        n += 1
    cuts.append(n); base = os.path.basename(p)[:-4]
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        if (b - a) / fps < 0.45: continue
        ss = [s for s in samples if a <= s[0] < b]
        gfr = [s for s in ss if s[1]]
        rec = dict(clip=base, i=i, a=round(a / fps, 3), b=round(b / fps, 3), fps=fps, g_ratio=round(len(gfr) / max(1, len(ss)), 2),
                   others=round(float(np.mean([s[2] for s in ss])) if ss else 0, 2), sharp=round(float(np.median([s[3] for s in ss])) if ss else 0),
                   luma=round(float(np.mean([s[4] for s in ss])) if ss else 0))
        if gfr:
            rec['g_w'] = round(float(np.median([s[1][3] for s in gfr])), 3); rec['g_eye'] = [round(float(np.median([s[1][4] for s in gfr])), 3), round(float(np.median([s[1][5] for s in gfr])), 3)]
        cap.set(cv2.CAP_PROP_POS_FRAMES, (a + b) // 2); ok, f = cap.read()
        if ok: cv2.imwrite(f'{outdir}/{base}__{i}.jpg', cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 80])
        res.append(rec)
    print(base[:50], len(cuts) - 1, 'shots', flush=True)
json.dump(res, open(outdir + '/shots.json', 'w'))
