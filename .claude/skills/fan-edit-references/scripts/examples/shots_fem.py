"""Shot list for the female Gosling edit (beats counted from the chorus drop at song 115.38 s)."""
import os
WORK = os.environ.get('FAN_EDIT_WORK', os.path.expanduser('~/fan-edits'))   # work folder: sources, models, outputs
V = WORK + '/cc/gos/vid/'
SRC = WORK + '/gos_edit/src/'
YLL, YNB_BIRD, YNB_DANCE, YNB_DRAWN, YNB_DATE = (SRC + p + '.mp4' for p in ('yt_lll_love2', 'yt_nb_saybird', 'yt_nb_dance', 'yt_nb_drawn', 'yt_nb_firstdate'))
NB, LLL, BV, FG, GS, CSL = (V + p + '__' for p in ('the-notebook-2004', 'la-la-land-2016', 'blue-valentine-2010', 'the-fall-guy-2024', 'gangster-squad-2013', 'crazy-stupid-love-2011'))
HOOK = dict(src=SRC + 'snl.mp4', t=144.95, push=0.04, bright=0.42)
SZ = 0.26
SHOTS = [
    # the looks - his eyes on the same point in four films
    dict(src=YNB_DRAWN, t=54.0, beats=2, size=SZ, speed=0.85),
    dict(src=YLL, t=87.42, beats=2, size=SZ, speed=0.85),
    dict(src=BV + 'yeah-like-youve-seen-her-and-just-know-her.mp4', t=3.0, beats=2, size=SZ, speed=0.85),
    dict(src=FG + 'i-obviously-failed-at-getting-us-our-happy-ending.mp4', t=1.0, beats=2, size=SZ, speed=0.85),
    # ping-pong: Sebastian <-> Mia, same point, 8 frames each
    dict(pingpong=True, beats=4, run=8),
    # joy
    dict(src=YNB_DANCE, t=11.0, beats=2, lock=False, cx=0.5, cy=0.55, zoom=1.0, speed=0.9),
    dict(src=BV + 'what-are-thinking-about-s8.mp4', t=5.0, beats=2, lock=False, cx=0.78, cy=0.33, zoom=1.3, speed=0.85),
    dict(src=YNB_DATE, t=232.0, beats=2, lock=False, cx=0.5, cy=0.45, zoom=1.0, speed=0.85),
    # one-beat burst of smiles into the second chorus line
    dict(src=YNB_BIRD, t=31.0, beats=1, size=SZ),
    dict(src=YLL, t=43.38, beats=1, size=SZ),
    dict(src=YNB_BIRD, t=70.40, beats=1, size=SZ),
    dict(src=YLL, t=59.23, beats=1, size=SZ),
    dict(src=NB + 'great-we-got-to-go.mp4', t=1.0, beats=1, size=SZ),
    # the beach embrace, then the golden jump-hug where the colour blooms
    dict(src=YNB_BIRD, t=60.56, beats=2, lock=False, cx=0.45, cy=0.5, zoom=1.0, speed=0.6),
    dict(src=NB + 'after-that-night-allie-and-noah-spent-every-waking-hour-together.mp4', t=2.6, beats=4, lock=False, cx=0.5, cy=0.42, zoom=1.0, speed=0.55, kind='finale', push=0.06),
]
PINGPONG = (dict(src=YLL, t=101.27, size=SZ, speed=0.85),
            dict(src=YLL, t=92.43, size=SZ, speed=0.85, who='other'))
