"""Synthesised sound design layered under the music: impacts, whooshes, soft accent thumps, a short riser."""
import sys, wave
import numpy as np
from scipy.signal import butter, sosfilt

SR = 48000
DUR = 22.0
rng = np.random.default_rng(7)


def env_ad(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-np.maximum(0, t - attack) / decay)


def impact(len_s=1.4):
    n = int(len_s * SR); t = np.arange(n) / SR
    f = 32 + 58 * np.exp(-t / 0.09)                      # pitch-dropping sub
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_ad(n, 0.002, 0.45)
    click = sosfilt(butter(2, [1500, 7000], 'bandpass', fs=SR, output='sos'), rng.standard_normal(n)) * env_ad(n, 0.0005, 0.018)
    body = sosfilt(butter(2, 300, 'lowpass', fs=SR, output='sos'), rng.standard_normal(n)) * env_ad(n, 0.002, 0.12)
    tail = sosfilt(butter(2, [200, 2500], 'bandpass', fs=SR, output='sos'), rng.standard_normal(n)) * env_ad(n, 0.01, 0.5) * 0.12
    x = 1.0 * sub + 0.5 * click + 0.6 * body + tail
    return np.tanh(1.6 * x) / np.tanh(1.6)


def whoosh(len_s=0.45, peak=0.6, f0=300, f1=4500):
    n = int(len_s * SR); t = np.arange(n) / SR; u = t / len_s
    noise = rng.standard_normal(n)
    out = np.zeros(n); blk = 512
    for i in range(0, n, blk):                            # swept band-pass, block-wise
        fc = f0 * (f1 / f0) ** min(1, u[i] / peak) if u[i] < peak else f1 * (f0 / f1) ** ((u[i] - peak) / (1 - peak))
        sos = butter(2, [fc * 0.6, min(fc * 1.6, SR / 2 - 100)], 'bandpass', fs=SR, output='sos')
        out[i:i + blk] = sosfilt(sos, noise[i:i + blk])
    e = np.where(u < peak, (u / peak) ** 2, ((1 - u) / (1 - peak)) ** 1.5)
    return out * e / (np.abs(out * e).max() + 1e-9)


def thump(len_s=0.35):
    n = int(len_s * SR); t = np.arange(n) / SR
    f = 45 + 70 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_ad(n, 0.002, 0.09)


def riser(len_s):
    n = int(len_s * SR); u = np.arange(n) / n
    noise = rng.standard_normal(n); out = np.zeros(n); blk = 1024
    for i in range(0, n, blk):
        fc = 400 * (6000 / 400) ** u[i]
        out[i:i + blk] = sosfilt(butter(2, [fc * 0.7, min(fc * 1.5, SR / 2 - 100)], 'bandpass', fs=SR, output='sos'), noise[i:i + blk])
    return out * u ** 2.5 / (np.abs(out).max() + 1e-9)


def zap(len_s):
    n = int(len_s * SR); t = np.arange(n) / SR
    sq = np.sign(np.sin(2 * np.pi * (900 + 2500 * np.sin(2 * np.pi * 37 * t)) * t))
    crush = np.round(sq * rng.uniform(0.4, 1.0, n) * 4) / 4
    gate = (np.sin(2 * np.pi * 60 * t) > -0.2).astype(float)
    return crush * gate * env_ad(n, 0.001, 0.05)


def click():
    n = int(0.08 * SR)
    c1 = sosfilt(butter(2, [2000, 9000], 'bandpass', fs=SR, output='sos'), rng.standard_normal(n)) * env_ad(n, 0.0003, 0.006)
    c2 = np.roll(c1, int(0.035 * SR)) * 0.7
    return c1 + c2


def gunshot(len_s=1.3):
    n = int(len_s * SR); t = np.arange(n) / SR
    crack = sosfilt(butter(2, 1200, 'highpass', fs=SR, output='sos'), rng.standard_normal(n)) * np.exp(-t / 0.004)
    blast = sosfilt(butter(2, 3200, 'lowpass', fs=SR, output='sos'), rng.standard_normal(n)) * np.exp(-t / 0.035)
    f = 45 + 110 * np.exp(-t / 0.03)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18)
    tail = sosfilt(butter(2, [180, 2800], 'bandpass', fs=SR, output='sos'), rng.standard_normal(n)) \
        * np.exp(-np.maximum(0, t - 0.01) / 0.45) * np.minimum(1, t / 0.01) * 0.22
    x = 0.9 * crack + 0.8 * blast + 1.0 * boom + tail
    for d, g in ((0.09, 0.32), (0.17, 0.18), (0.29, 0.09)):   # slap-back off walls
        k = int(d * SR); x[k:] += g * sosfilt(butter(2, 2500, 'lowpass', fs=SR, output='sos'), x[:-k])
    return np.tanh(2.2 * x) / np.tanh(2.2)


def place(buf, sig, t, gain, pan=0.0):
    i = int(t * SR); j = min(len(buf), i + len(sig))
    if i >= len(buf): return
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    buf[i:j, 0] += sig[:j - i] * gain * l * 1.414
    buf[i:j, 1] += sig[:j - i] * gain * r * 1.414


def main(out):
    buf = np.zeros((int(DUR * SR), 2))
    fps = 30
    for f in (189, 403):                         # one gunshot per drop (muzzle-flash frame)
        place(buf, gunshot(), f / fps - 0.004, 0.85)
    for f in (189, 403):                         # weight under the first shot
        place(buf, impact(), f / fps - 0.005, 0.28)
    place(buf, impact(), 598 / fps - 0.005, 0.55)    # final
    place(buf, impact(0.9), 383 / fps, 0.25)
    for f in (26, 81, 132, 172):                 # coloured accent words
        place(buf, thump(), f / fps, 0.30)
    for f in (172, 333, 509):                    # glitches
        place(buf, zap(0.12), f / fps, 0.16)
    for f in (283, 605):                         # freeze frames: shutter + low swell
        place(buf, click(), f / fps, 0.35)
        place(buf, thump(0.5), f / fps, 0.22)
    peak = np.abs(buf).max()
    if peak > 0.98: buf *= 0.98 / peak
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(out, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


if __name__ == '__main__':
    main(sys.argv[1])
