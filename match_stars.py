#!/usr/bin/env python3
"""
This script identifies matching stars between a “small” image (with few stars)
and a “large” image (with many stars), by finding the best geometric alignment
(scale + rotation + translation) that maps as many small-image stars as possible
onto the large-image stars.

Algorithm overview (in simple language):
1. **Detect stars**:  
   - Convert image to grayscale and blur to reduce noise.  
   - Use a “top-hat” filter to remove uneven background light.  
   - Threshold and clean small artifacts.  
   - Find contours of bright spots, filter by size, compute each star’s center, radius, and brightness.  
   - Deduplicate overlapping detections by keeping the brightest in each cluster.  

2. **Match stars**:  
   - Randomly shuffle all pairs of small-image stars.  
   - For each small-image pair (p1, p2):  
     a. Compute their distance and angle → this defines a “model” in small-image space.  
     b. Loop over every pair in the large image (l1, l2):  
        • Compute scale = dist(l1,l2) / dist(p1,p2).  
        • Compute rotation = angle(l2–l1) – angle(p2–p1).  
        • Compute translation so that p1 maps onto l1.  
        • Apply this transform to *all* small-image stars and see which land close to a large-image star.  
        • Use two tolerances: proportional to star size and a fixed pixel margin.  
        • Count how many small-image stars “matched.”  
     c. Remember the best mapping; stop early if ≥60% of small-image stars match.  

3. **Annotate results**:  
   - Draw rectangles and IDs around matching large-image stars.  
   - Save the aligned image and a CSV of matched star coordinates.

This combination of random sampling of small-image pairs and exhaustive checking
of large-image pairs finds a high-quality alignment without testing every possible
small→large pairing (which would be too slow).
"""

import cv2
import numpy as np
import csv
import math
import random
from itertools import combinations

# -----------------------------------------------------------------------------
# Function: detect_stars
# -----------------------------------------------------------------------------
def detect_stars(image_path, annotated_path, csv_path,
                 tophat_kernel=(31,31), thresh_val=5,
                 min_area=3, max_area=8000,
                 padding=5, dup_thresh_factor=1.5):
    """
    Identify stars in an image, annotate them, and save their properties.

    Steps:
    1. Read image, convert to grayscale, blur to reduce noise.
    2. Estimate and subtract background using a morphological 'top-hat' filter.
    3. Threshold the result to isolate bright spots (potential stars).
    4. Morphologically clean small artifacts.
    5. Find contours of bright regions; filter by area to remove too-small or too-large blobs.
    6. For each remaining contour:
       - Compute bounding box with padding.
       - Compute center (cx, cy), radius r from area, and average brightness b.
    7. Sort detections by brightness descending; deduplicate overlapping blobs
       by keeping the brightest in each cluster (controlled by dup_thresh_factor).
    8. Annotate the original image with rectangles and numeric IDs.
    9. Write a CSV with columns: id, x, y, r, b.

    Returns:
        List of dicts, each with keys:
        'id'   : unique integer
        'x', 'y': center coordinates (floats)
        'r'    : estimated radius
        'b'    : mean brightness
        'bbox' : bounding box tuple (x1, y1, x2, y2)
    """
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # 1. Background removal (top-hat)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tophat_kernel)
    background  = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_open)
    tophat      = cv2.subtract(gray, background)

    # 2. Threshold to binary + clean small noise
    _, thresh = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
    kn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kn, iterations=1)

    # 3. Find contours of potential stars
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Bounding box with padding
        x,y,w,h = cv2.boundingRect(cnt)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, img.shape[1]-1)
        y2 = min(y + h + padding, img.shape[0]-1)

        # Center, radius, and brightness
        cx = x1 + (x2 - x1)/2.0
        cy = y1 + (y2 - y1)/2.0
        r  = math.sqrt(area / math.pi)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        b = cv2.mean(gray, mask=mask)[0]

        raw.append({'cx':cx, 'cy':cy, 'r':r, 'b':b, 'bbox':(x1,y1,x2,y2)})

    # 4. Deduplicate overlapping detections
    raw.sort(key=lambda s: s['b'], reverse=True)
    stars = []
    for s in raw:
        if any(math.hypot(s['cx']-u['cx'], s['cy']-u['cy']) <
               dup_thresh_factor * max(s['r'], u['r']) for u in stars):
            continue
        stars.append(s)

    # 5. Annotate image and write CSV
    rect_color, rect_thick = (0,255,255), 1
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','x','y','r','b'])
        for idx, s in enumerate(stars, start=1):
            x1,y1,x2,y2 = map(int, s['bbox'])
            cv2.rectangle(img, (x1,y1),(x2,y2), rect_color, rect_thick)
            cv2.putText(img, str(idx), (x1, y1-10),
                        font, fs, rect_color, ft, cv2.LINE_AA)
            writer.writerow([idx, s['cx'], s['cy'], s['r'], s['b']])
    cv2.imwrite(annotated_path, img)

    return [
        {'id': idx, 'x': s['cx'], 'y': s['cy'],
         'r': s['r'], 'b': s['b'], 'bbox': s['bbox']}
        for idx, s in enumerate(stars, start=1)
    ]

# -----------------------------------------------------------------------------
# Function: match_stars
# -----------------------------------------------------------------------------
def match_stars(small, large,
                dist_tol_factor=1.2,
                abs_tol=5.0,
                match_threshold=0.6):
    """
    Find the best alignment (scale + rotation + translation) that maps
    as many small-image stars onto large-image stars as possible.

    Steps:
    1. Build a random ordering of all small-image star pairs.
    2. For each small-image pair (p1, p2):
       a. Compute their distance ds and angle ang_s.
    3. For each large-image pair (l1, l2):
       a. Compute distance dl and angle ang_l.
       b. Derive scale = dl / ds, rotation = ang_l - ang_s.
       c. Compute translation so p1 → l1.
       d. Apply this transform to *every* small-image star:
          - new_x = rotated_and_scaled_x + translation_x
          - new_y = rotated_and_scaled_y + translation_y
       e. For each transformed small star, find the nearest large star.
          Use tolerance = (r_small * scale * dist_tol_factor) + abs_tol.
       f. Count how many small stars fit within tolerance → score.
    4. Keep the mapping with the highest score.
    5. Stop early if score ≥ match_threshold * total_small.

    Returns:
        Dict mapping small_id → large_id for matched stars.
    """
    # Precompute large-star coordinates array
    large_pts = np.array([[l['x'], l['y']] for l in large], dtype=np.float32)

    best_matches, best_score = {}, 0
    min_good = math.ceil(match_threshold * len(small))

    # 1. Shuffle all small-image pairs
    small_pairs = list(combinations(small, 2))
    random.shuffle(small_pairs)

    # 2. Loop over each random small-image pair
    for p1, p2 in small_pairs:
        xs1, ys1 = p1['x'], p1['y']
        xs2, ys2 = p2['x'], p2['y']
        ds = math.hypot(xs2 - xs1, ys2 - ys1)
        if ds == 0:
            continue
        ang_s = math.atan2(ys2 - ys1, xs2 - xs1)

        # 3. Try aligning to every large-image pair
        for l1, l2 in combinations(large, 2):
            xl1, yl1 = l1['x'], l1['y']
            xl2, yl2 = l2['x'], l2['y']
            dl = math.hypot(xl2 - xl1, yl2 - yl1)
            if dl == 0:
                continue

            # 3b. Compute scale and rotation
            scale = dl / ds
            ang_l = math.atan2(yl2 - yl1, xl2 - xl1)
            theta = ang_l - ang_s
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            # 3c–e. Apply transform, test all small stars
            current = {}
            for s in small:
                dx, dy = s['x'] - xs1, s['y'] - ys1
                xr = cos_t * dx - sin_t * dy
                yr = sin_t * dx + cos_t * dy
                xp = xr * scale + xl1
                yp = yr * scale + yl1

                tol = s['r'] * scale * dist_tol_factor + abs_tol
                dists = np.linalg.norm(large_pts - [xp, yp], axis=1)
                j = int(np.argmin(dists))
                if dists[j] <= tol:
                    current[s['id']] = large[j]['id']

            # 4. Update best mapping if improved
            score = len(current)
            if score > best_score:
                best_score, best_matches = score, current
                if score >= min_good:
                    return best_matches

    return best_matches

# -----------------------------------------------------------------------------
# Function: annotate_matches
# -----------------------------------------------------------------------------
def annotate_matches(image_path, matches, large, output_path):
    """
    Draw rectangles and small-image IDs around matched stars
    on the large image and save the result.
    """
    img = cv2.imread(image_path)
    color, thick = (0,255,0), 2
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    lookup = {l['id']: l for l in large}

    for sid, lid in matches.items():
        x1, y1, x2, y2 = map(int, lookup[lid]['bbox'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img, str(sid), (x1, y1-10), font, fs, color, ft, cv2.LINE_AA)

    cv2.imwrite(output_path, img)

# -----------------------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Detect stars in the small image (few stars)
    small = detect_stars(
        "small.jpg", "small_detected.jpg", "small_coords.csv",
        tophat_kernel=(15,15), thresh_val=25,
        min_area=10, max_area=200,
        padding=5, dup_thresh_factor=3.0
    )
    print(f"✔ small.jpg: detected {len(small)} stars")

    # 2. Detect stars in the large image (many stars)
    large = detect_stars(
        "large.png", "large_detected.jpg", "large_coords.csv",
        tophat_kernel=(31,31), thresh_val=5,
        min_area=3, max_area=8000,
        padding=5, dup_thresh_factor=1.5
    )
    print(f"✔ large.png: detected {len(large)} stars")

    # 3. Match stars with geometric alignment
    matches = match_stars(
        small, large,
        dist_tol_factor=1.2,   # proportional tolerance
        abs_tol=5.0,           # fixed pixel tolerance
        match_threshold=0.6    # stop when ≥60% matched
    )
    pct = len(matches) / len(small) * 100 if small else 0
    print(f"✔ matched {len(matches)}/{len(small)} stars ({pct:.1f}%)")

    # 4. Annotate and save the final matched image
    annotate_matches("large.png", matches, large, "large_matched.jpg")
    print("✔ large_matched.jpg saved")

    # 5. Write CSV of matched coordinate pairs
    with open("matches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["small_id","small_x","small_y","large_id","large_x","large_y"])
        ls = {s['id']: s for s in small}
        ll = {l['id']: l for l in large}
        for sid, lid in matches.items():
            S, L = ls[sid], ll[lid]
            w.writerow([sid, S['x'], S['y'], lid, L['x'], L['y']])
    print("✔ matches.csv saved")
    print("✔ Done.")
    