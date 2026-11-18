"""per-frame transition probe: cut score, global motion (scale/rot/tx/ty), sharpness, luma, saturation"""
import sys, numpy as np, cv2, json
path, end = sys.argv[1], float(sys.argv[2])
crop = sys.argv[3] if len(sys.argv) > 3 else None   # y0:y1 to restrict to the picture band
cap = cv2.VideoCapture(path); fps = cap.get(5)
prev = None; rows = []; n = 0
orb = cv2.ORB_create(800); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
while True:
    ok, f = cap.read()
    if not ok or n / fps > end: break
    if crop: y0, y1 = map(int, crop.split(':')); f = f[y0:y1]
    g = cv2.cvtColor(cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV)
    sharp = cv2.Laplacian(g, cv2.CV_32F).var()
    r = dict(t=round(n / fps, 3), n=n, luma=round(float(g.mean()), 1), sat=round(float(hsv[..., 1].mean()), 1), sharp=round(float(sharp), 1))
    if prev is not None:
        h1 = cv2.calcHist([g], [0], None, [32], [0, 256]); h0 = cv2.calcHist([prev], [0], None, [32], [0, 256])
        r['hist'] = round(float(cv2.compareHist(h0, h1, cv2.HISTCMP_BHATTACHARYYA)), 3)
        r['mad'] = round(float(np.abs(g.astype(np.float32) - prev).mean()), 1)
        k0, d0 = orb.detectAndCompute(prev, None); k1, d1 = orb.detectAndCompute(g, None)
        if d0 is not None and d1 is not None and len(k0) > 10 and len(k1) > 10:
            m = sorted(bf.match(d0, d1), key=lambda x: x.distance)[:300]
            if len(m) > 12:
                P = np.float32([k0[x.queryIdx].pt for x in m]); Q = np.float32([k1[x.trainIdx].pt for x in m])
                M, inl = cv2.estimateAffinePartial2D(P, Q, ransacReprojThreshold=3)
                if M is not None and inl.sum() > 10:
                    r['sc'] = round(float(np.hypot(M[0, 0], M[1, 0])), 4); r['rot'] = round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2)
                    r['tx'] = round(float(M[0, 2]), 1); r['ty'] = round(float(M[1, 2]), 1); r['inl'] = int(inl.sum())
    rows.append(r); prev = g.astype(np.float32) if False else g; n += 1
json.dump(rows, open(path + '.probe.json', 'w'))
print('fps', fps, 'frames', n)
