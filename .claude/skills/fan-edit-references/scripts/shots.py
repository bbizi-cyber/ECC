import cv2, sys, json, os, numpy as np
from PIL import Image, ImageDraw
src, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS); n=0; prev=None; cuts=[0]; frames={}
while True:
    ok, f = cap.read()
    if not ok: break
    small = cv2.resize(f,(64,36)); h = cv2.calcHist([cv2.cvtColor(small,cv2.COLOR_BGR2HSV)],[0,1],None,[16,8],[0,180,0,256]); cv2.normalize(h,h)
    if prev is not None and cv2.compareHist(prev,h,cv2.HISTCMP_BHATTACHARYYA)>0.35 and n-cuts[-1]>4: cuts.append(n)
    prev=h; frames[n]=None; n+=1
cuts.append(n)
shots=[(a,b) for a,b in zip(cuts[:-1],cuts[1:]) if (b-a)/fps>=0.35]
cap=cv2.VideoCapture(src); thumbs=[]
for a,b in shots:
    cap.set(cv2.CAP_PROP_POS_FRAMES,(a+b)//2); ok,f=cap.read()
    if not ok: continue
    if f.mean()<12: continue
    im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)); im.thumbnail((240,240))
    d=ImageDraw.Draw(im); d.rectangle([0,0,130,14],fill=(0,0,0)); d.text((2,1),f"{a/fps:.1f}-{b/fps:.1f}",fill=(255,255,0))
    thumbs.append(im)
cols=6; W=240; H=max(t.size[1] for t in thumbs); rows=(len(thumbs)+cols-1)//cols
sheet=Image.new('RGB',(cols*W,rows*H)); 
for i,t in enumerate(thumbs): sheet.paste(t,((i%cols)*W,(i//cols)*H))
sheet.save(out); print(os.path.basename(src),len(thumbs),'shots', sheet.size)
