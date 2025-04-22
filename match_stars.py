#!/usr/bin/env python3
"""
match_stars.py

Star Matching Between Two Images with Clickable Arrows
------------------------------------------------------

This script:
1) Detects stars in two images (small.jpg, large.png).
2) Matches stars by exhaustive geometric alignment.
3) Outputs annotated images and CSVs into 'output/'.
4) After ✔ All done., opens an interactive window with three “screens”:
     ◀  [Before Small]   [Before Large]  ▶
     ◀  [Detected Small] [Detected Large] ▶
     ◀  [Matched Small]  [Matched Large]  ▶
   Click the left/right arrow regions to navigate (wrapping around).
"""

import os, math, csv
import numpy as np
import cv2
import sep
from itertools import combinations

# -----------------------------------------------------------------------------
# 1) Setup I/O
# -----------------------------------------------------------------------------
INPUT_SMALL   = "small.jpg"
INPUT_LARGE   = "large.png"
OUTPUT_DIR    = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMALL_BEFORE  = os.path.join(OUTPUT_DIR, "small_before.jpg")
LARGE_BEFORE  = os.path.join(OUTPUT_DIR, "large_before.png")
cv2.imwrite(SMALL_BEFORE, cv2.imread(INPUT_SMALL))
cv2.imwrite(LARGE_BEFORE, cv2.imread(INPUT_LARGE))

SMALL_DETECTED = os.path.join(OUTPUT_DIR, "small_detected.jpg")
LARGE_DETECTED = os.path.join(OUTPUT_DIR, "large_detected.jpg")
SMALL_MATCHED  = os.path.join(OUTPUT_DIR, "small_matched.jpg")
LARGE_MATCHED  = os.path.join(OUTPUT_DIR, "large_matched.jpg")
SMALL_CSV      = os.path.join(OUTPUT_DIR, "small_coords.csv")
LARGE_CSV      = os.path.join(OUTPUT_DIR, "large_coords.csv")
MATCHES_CSV    = os.path.join(OUTPUT_DIR, "matches.csv")

# -----------------------------------------------------------------------------
# 2) Star Detection
# -----------------------------------------------------------------------------
def detect_sep(path, thresh_sigma=5.0, min_area=5):
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bkg  = sep.Background(gray)
    data = gray - bkg.back()
    rms  = np.median(bkg.rms())
    sep.set_extract_pixstack(data.size * 2)
    objs = sep.extract(data, thresh=thresh_sigma*rms, err=bkg.rms(), minarea=min_area)
    stars = []
    for i,o in enumerate(objs, start=1):
        x,y,flux = o['x'], o['y'], o['flux']
        r = math.sqrt(flux/ math.pi)
        stars.append({'id':i,'x':x,'y':y,'r':r,'b':flux})
    return stars

def detect_cc(path, bg_blur=51, thresh_std=0.8, open_disk=3, min_area=2, max_area=200):
    img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    bg   = cv2.GaussianBlur(img, (bg_blur,bg_blur), 0)
    data = img - bg
    med,std = np.median(data), np.std(data)
    th = med + thresh_std*std
    bw = (data>th).astype(np.uint8)
    ker= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_disk,open_disk))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker)
    nlab, labels, stats, cents = cv2.connectedComponentsWithStats(bw,8)
    stars, idx = [],1
    for lab in range(1,nlab):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area<min_area or area>max_area: continue
        cx,cy = cents[lab]
        flux = float(np.sum(data[labels==lab]))
        r = math.sqrt(area/math.pi)
        stars.append({'id':idx,'x':cx,'y':cy,'r':r,'b':flux})
        idx +=1
    return stars

def annotate_all(src, stars, dst):
    img = cv2.imread(src)
    for s in stars:
        x,y,r = int(round(s['x'])),int(round(s['y'])),int(round(s['r']))
        cv2.circle(img,(x,y),r,(0,255,255),1)
        cv2.putText(img,str(s['id']), (x+r+2,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    cv2.imwrite(dst, img)

# -----------------------------------------------------------------------------
# 3) Exhaustive Matching
# -----------------------------------------------------------------------------
def match_stars(small, large,
                dist_tol_factor=1.2,
                abs_tol=5.0,
                match_threshold=0.6):
    best, best_score = {}, 0
    N_small = len(small)
    min_good = math.ceil(match_threshold * N_small)
    large_pts = np.array([[l['x'],l['y']] for l in large], dtype=np.float64)

    for p1,p2 in combinations(small,2):
        xs1,ys1 = p1['x'],p1['y']
        xs2,ys2 = p2['x'],p2['y']
        ds = math.hypot(xs2-xs1, ys2-ys1)
        if ds==0: continue
        ang_s = math.atan2(ys2-ys1, xs2-xs1)
        for l1,l2 in combinations(large,2):
            xl1,yl1 = l1['x'],l1['y']
            xl2,yl2 = l2['x'],l2['y']
            dl = math.hypot(xl2-xl1, yl2-yl1)
            if dl==0: continue
            scale = dl/ds
            ang_l = math.atan2(yl2-yl1, xl2-xl1)
            theta = ang_l - ang_s
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            curr = {}
            for s in small:
                dx,dy = s['x']-xs1, s['y']-ys1
                xr = cos_t*dx - sin_t*dy
                yr = sin_t*dx + cos_t*dy
                xp,yp = xr*scale + xl1, yr*scale + yl1

                tol = s['r']*scale*dist_tol_factor + abs_tol
                dists = np.hypot(large_pts[:,0]-xp,
                                 large_pts[:,1]-yp)
                j = int(np.argmin(dists))
                if dists[j] <= tol:
                    curr[s['id']] = large[j]['id']

            score = len(curr)
            if score>best_score:
                best_score, best = score, curr.copy()
                if score>=min_good:
                    return best
    return best

# -----------------------------------------------------------------------------
# 4) Annotation & CSV
# -----------------------------------------------------------------------------
def annotate_matched_small(src, small, matches, dst):
    img = cv2.imread(src)
    for s in small:
        sid = s['id']
        if sid not in matches: continue
        x,y,r = int(round(s['x'])),int(round(s['y'])),int(round(s['r']))
        cv2.circle(img,(x,y),r,(0,255,0),2)
        cv2.putText(img,str(sid),(x+r+2,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imwrite(dst, img)

def annotate_matched_large(src, large, matches, dst):
    inv = {v:k for k,v in matches.items()}
    img = cv2.imread(src)
    for l in large:
        lid = l['id']
        if lid not in inv: continue
        sid = inv[lid]
        x,y,r = int(round(l['x'])),int(round(l['y'])),int(round(l['r']))
        cv2.circle(img,(x,y),r,(0,0,255),2)
        cv2.putText(img,str(sid),(x+r+2,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.imwrite(dst, img)

def write_coords(path, stars):
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(['id','x','y','r','b'])
        for s in stars:
            w.writerow([s['id'],s['x'],s['y'],s['r'],s['b']])

def write_matches(path, matches, small, large):
    sm={s['id']:s for s in small}
    lg={l['id']:l for l in large}
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["small_id","small_x","small_y",
                    "large_id","large_x","large_y"])
        for sid,lid in matches.items():
            S,L=sm[sid],lg[lid]
            w.writerow([sid,S['x'],S['y'],lid,L['x'],L['y']])

# -----------------------------------------------------------------------------
# 5) Main
# -----------------------------------------------------------------------------
if __name__=="__main__":
    # Detect
    small=detect_sep(INPUT_SMALL)
    print(f"small: {len(small)} stars")
    annotate_all(INPUT_SMALL, small, SMALL_DETECTED)
    write_coords(SMALL_CSV, small)

    large=detect_cc(INPUT_LARGE)
    print(f"large: {len(large)} stars")
    annotate_all(INPUT_LARGE, large, LARGE_DETECTED)
    write_coords(LARGE_CSV, large)

    matches=match_stars(small, large)
    print(f"matched {len(matches)}/{len(small)} stars")
    write_matches(MATCHES_CSV, matches, small, large)

    annotate_matched_small(INPUT_SMALL, small, matches, SMALL_MATCHED)
    annotate_matched_large(INPUT_LARGE, large, matches, LARGE_MATCHED)

    print("✔ All done.")

    # -----------------------------------------------------------------------------
    # 6) Interactive 3‑Screen Display with Clickable Arrows
    # -----------------------------------------------------------------------------
    # Prepare stages
    stages = [
      ("Before Matching", SMALL_BEFORE, LARGE_BEFORE),
      ("After Detection", SMALL_DETECTED, LARGE_DETECTED),
      ("After Matching", SMALL_MATCHED, LARGE_MATCHED),
    ]

    def make_stage(title, path_s, path_l, height=600, arrow_w=100):
        # load and resize both images to same height
        im_s, im_l = cv2.imread(path_s), cv2.imread(path_l)
        h_scale = height / max(im_s.shape[0], im_l.shape[0])
        rs = cv2.resize(im_s, (int(im_s.shape[1]*h_scale), height))
        rl = cv2.resize(im_l, (int(im_l.shape[1]*h_scale), height))
        comp = cv2.hconcat([rs, rl])
        # draw title bar
        cv2.rectangle(comp, (0,0), (comp.shape[1], 40), (0,0,0), -1)
        cv2.putText(comp, title, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        # draw left arrow
        pts = np.array([[20, height//2],
                        [arrow_w-20, height//2-40],
                        [arrow_w-20, height//2+40]], np.int32)
        cv2.fillConvexPoly(comp, pts, (200,200,200))
        # draw right arrow
        w = comp.shape[1]
        pts = np.array([[w-20, height//2],
                        [w-arrow_w+20, height//2-40],
                        [w-arrow_w+20, height//2+40]], np.int32)
        cv2.fillConvexPoly(comp, pts, (200,200,200))
        return comp

    imgs = [make_stage(t, s, l) for (t,s,l) in stages]
    idx = 0
    win = "Star Matching Steps"
    arrow_w = 100

    # mouse callback
    def on_mouse(evt, x, y, flags, param):
        global idx
        if evt == cv2.EVENT_LBUTTONDOWN:
            w = imgs[idx].shape[1]
            if x < arrow_w:
                idx = (idx-1) % len(imgs)
            elif x > w - arrow_w:
                idx = (idx+1) % len(imgs)
            cv2.imshow(win, imgs[idx])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, imgs[idx])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
