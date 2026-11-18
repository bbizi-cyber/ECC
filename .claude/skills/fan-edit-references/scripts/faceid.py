"""Find a given actor in images: YuNet detection + SFace recognition (OpenCV zoo models)."""
import cv2, numpy as np, glob, json, sys, os
WORK = os.environ.get('FAN_EDIT_WORK', os.path.expanduser('~/fan-edits'))   # work folder: sources, models, outputs
M = WORK + '/models/'
_det = {}; _rec = cv2.FaceRecognizerSF.create(M + 'sface.onnx', '')
def faces(bgr, thr=0.6):
    H, W = bgr.shape[:2]
    if (W, H) not in _det: _det[(W, H)] = cv2.FaceDetectorYN.create(M + 'yunet.onnx', '', (W, H), thr, 0.3, 5000)
    _, f = _det[(W, H)].detect(bgr)
    return [] if f is None else list(f)
def embed(bgr, f):
    return _rec.feature(_rec.alignCrop(bgr, f)).ravel()
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
if __name__ == '__main__':
    ref = np.load(sys.argv[1]); out = []
    for p in sorted(glob.glob(sys.argv[2])):
        img = cv2.imread(p)
        if img is None: continue
        H, W = img.shape[:2]; fs = faces(img)
        rec = dict(p=os.path.basename(p), n=len(fs), luma=round(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()), 1), faces=[])
        for f in fs:
            e = embed(img, f); s = max(cos(e, r) for r in ref)
            rec['faces'].append(dict(sim=round(s, 3), x=round(float(f[0] + f[2] / 2) / W, 3), y=round(float(f[1] + f[3] / 2) / H, 3), w=round(float(f[2]) / W, 3), conf=round(float(f[14]), 2)))
        out.append(rec)
    json.dump(out, open(sys.argv[3], 'w'))
    print(len(out), 'images')
