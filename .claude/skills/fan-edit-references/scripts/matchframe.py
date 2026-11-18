"""Find where a frame from one encode of a film appears in another encode (different crop/size/fps).
usage: matchframe.py ref.mp4 t_ref cand.mp4 [cand ...] -> best (file, time, score)"""
import sys, subprocess, numpy as np, cv2
def active(path):
    out = subprocess.run(['ffmpeg', '-ss', '5', '-t', '6', '-i', path, '-vf', 'cropdetect=24:2:0', '-f', 'null', '-'], capture_output=True, text=True).stderr
    cr = [tuple(map(int, l.split('crop=')[1].split()[0].split(':'))) for l in out.splitlines() if 'crop=' in l]
    return max(cr, key=lambda c: c[0] * c[1]) if cr else None
def sig(img, ar=1.6):
    """central region of width ar*height, gray, 80x50, zero-mean unit-var"""
    H, W = img.shape[:2]; cw = min(W, int(H * ar)); x0 = (W - cw) // 2
    g = cv2.cvtColor(cv2.resize(img[:, x0:x0 + cw], (80, 50), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0); return (g - g.mean()) / (g.std() + 1e-6)
def frames(path, fps=6, crop=None):
    vf = (f'crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]},' if crop else '') + f'fps={fps},scale=320:-2'
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True).stdout
    h = int(subprocess.run(['ffprobe', '-v', 'error', '-f', 'lavfi', '-i', f'movie={path},{vf}', '-show_entries', 'stream=height', '-of', 'csv=p=0'], capture_output=True, text=True).stdout.split()[0]) if False else None
    return p
def scan(cand, refsig, fps=6):
    c = active(cand); cap = cv2.VideoCapture(cand); nfps = cap.get(5); n = int(cap.get(7)); step = max(1, int(round(nfps / fps)))
    best = (-1, 0)
    for k in range(0, n, step):
        cap.set(1, k) if False else None
        ok = cap.grab()
        if not ok: break
    cap.release()
    cap = cv2.VideoCapture(cand); k = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if k % step == 0:
            ok, f = cap.retrieve()
            if c: f = f[c[3]:c[3] + c[1], c[2]:c[2] + c[0]]
            s = float((sig(f) * refsig).mean())
            if s > best[0]: best = (s, k / nfps)
        k += 1
    return best
if __name__ == '__main__':
    ref, t = sys.argv[1], float(sys.argv[2])
    cap = cv2.VideoCapture(ref); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
    c = active(ref)
    if c: f = f[c[3]:c[3] + c[1], c[2]:c[2] + c[0]]
    rs = sig(f)
    for cand in sys.argv[3:]:
        s, tt = scan(cand, rs)
        print(f'{cand.split("/")[-1]} t={tt:.2f} score={s:.3f}')
