"""Find the 'ideal transition' pairs: the same actor, the same angle, different films.

Input: shot thumbnails named <film>__<clip>__<i>.jpg (shotscan.py writes them). For every shot it takes the
person silhouette (U^2-Net) and, when visible, the actor's face (YuNet + SFace), then scores every pair of shots
from DIFFERENT films: silhouette overlap (same place, same size, same pose) plus face position/size/yaw match.
usage: pairfinder.py ref_embeddings.npy 'shots/*.jpg' [top_n] [--backs]  -> best pairs, highest score first.
By default both shots must show the actor's face; --backs also pairs shots where he is seen from behind (no face)."""
import sys, os, glob, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extras as EX, faceid as F

def describe(path, ref):
    img = cv2.imread(path); rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
    m = cv2.resize(EX.person_mask(rgb), (64, 36), interpolation=cv2.INTER_AREA) > 0.5
    face = None
    for q in F.faces(img, 0.5):
        if max(F.cos(F.embed(img, q), r) for r in ref) > 0.36:
            (rx, ry), (lx, ly), (nx, ny) = q[4:6], q[6:8], q[8:10]
            yaw = ((nx - rx) - (lx - nx)) / max(1.0, lx - rx)                 # <0 / >0: head turned left / right
            face = np.array([(rx + lx) / 2 / W, (ry + ly) / 2 / H, q[2] / W, yaw])
    return m, face

def score(a, b):
    (ma, fa), (mb, fb) = a, b
    inter, union = (ma & mb).sum(), (ma | mb).sum()
    if union < 30: return 0.0
    s = inter / union                                                          # silhouette IoU: place, size, pose
    if fa is not None and fb is not None:
        d = np.hypot(*(fa[:2] - fb[:2])) + abs(np.log(fa[2] / fb[2])) * 0.3 + abs(fa[3] - fb[3]) * 0.25
        s = 0.5 * s + 0.5 * max(0.0, 1 - d / 0.25)
    elif (fa is None) != (fb is None):
        s *= 0.8                                                               # a face in one shot, a back in the other
    return float(s)

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]; backs = '--backs' in sys.argv
    ref = np.load(args[0]); files = sorted(glob.glob(args[1])); top = int(args[2]) if len(args) > 2 else 15
    D = {p: describe(p, ref) for p in files}
    if not backs: files = [p for p in files if D[p][1] is not None]          # the same actor must be in both shots
    film = lambda p: os.path.basename(p).split('__')[0]
    pairs = [(score(D[a], D[b]), a, b) for i, a in enumerate(files) for b in files[i + 1:] if film(a) != film(b) and D[a][0].sum() > 40 and D[b][0].sum() > 40]
    for s, a, b in sorted(pairs, reverse=True)[:top]:
        print(f'{s:.3f}  {os.path.basename(a)}  <->  {os.path.basename(b)}')
