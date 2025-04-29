#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Star Matcher UI
Authors: Roni Michaeli & Neta Cohen  
Ariel University, Introduction to Space Engineering, Assignment 1  
License: MIT

This script provides a graphical interface for detecting and matching stars
between a small “template” image and a larger image using PySide6. It performs:
1. Welcome screen  
2. Image selection for small and large images  
3. Threshold classification for detection parameters  
4. Loading screen with progress bar  
5. Star detection and matching in the background  
6. Display of results and CSV download buttons  

Installation:
    # Create and activate virtual environment (optional)
    python3 -m venv venv
    source venv/bin/activate       # macOS/Linux
    venv\Scripts\activate          # Windows

    # Install dependencies
    pip install --upgrade pip
    pip install opencv-python numpy sep PySide6

Usage:
    python star_matcher_ui.py
"""

import sys
import os
import math
import csv
import numpy as np
import cv2
import sep
from itertools import combinations

# -----------------------------------------------------------------------------
# Section 1: Input/Output Setup
# -----------------------------------------------------------------------------
# Define default input image paths and output directory.
# Ensure the output directory exists and save initial copies of the images.
INPUT_SMALL = "small.jpg"
INPUT_LARGE = "large.png"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMALL_BEFORE = os.path.join(OUTPUT_DIR, "small_before.jpg")
LARGE_BEFORE = os.path.join(OUTPUT_DIR, "large_before.png")
SMALL_DETECTED = os.path.join(OUTPUT_DIR, "small_detected.jpg")
LARGE_DETECTED = os.path.join(OUTPUT_DIR, "large_detected.jpg")
SMALL_MATCHED = os.path.join(OUTPUT_DIR, "small_matched.jpg")
LARGE_MATCHED = os.path.join(OUTPUT_DIR, "large_matched.jpg")
SMALL_CSV = os.path.join(OUTPUT_DIR, "small_coords.csv")
LARGE_CSV = os.path.join(OUTPUT_DIR, "large_coords.csv")
MATCHES_CSV = os.path.join(OUTPUT_DIR, "matches.csv")

# Save copies of the original images for the "Before" screen.
cv2.imwrite(SMALL_BEFORE, cv2.imread(INPUT_SMALL))
cv2.imwrite(LARGE_BEFORE, cv2.imread(INPUT_LARGE))

# -----------------------------------------------------------------------------
# Section 2: Star Detection Functions
# -----------------------------------------------------------------------------
# These functions implement two methods for detecting stars:
#  - detect_sep: Uses the SEP library for background subtraction and thresholding.
#  - detect_cc: Uses OpenCV for grayscale subtraction and connected component analysis.

def detect_sep(path, thresh_sigma=5.0, min_area=5):
    """
    Detect stars in an image using the SEP library.

    Args:
        path (str): Path to the input image file.
        thresh_sigma (float): Detection threshold in units of background RMS.
        min_area (int): Minimum area (in pixels) to consider as a star.

    Returns:
        List[dict]: Each entry contains 'id', 'x', 'y', 'r' (radius), and 'b' (brightness).
    """
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bkg = sep.Background(gray)
    data = gray - bkg.back()
    rms = np.median(bkg.rms())
    sep.set_extract_pixstack(data.size * 2)
    objs = sep.extract(data, thresh=thresh_sigma*rms, err=bkg.rms(), minarea=min_area)
    stars = []
    for i, o in enumerate(objs, start=1):
        x, y, flux = o['x'], o['y'], o['flux']
        r = math.sqrt(flux / math.pi)
        stars.append({'id': i, 'x': x, 'y': y, 'r': r, 'b': flux})
    return stars

def detect_cc(path, bg_blur=51, thresh_std=0.8, open_disk=3, min_area=2, max_area=200):
    """
    Detect stars in an image using OpenCV connected components.

    Args:
        path (str): Path to the input image file.
        bg_blur (int): Kernel size for Gaussian blur background estimation.
        thresh_std (float): Number of standard deviations above median for threshold.
        open_disk (int): Size of the structuring element for noise removal.
        min_area (int): Minimum pixel area for valid star.
        max_area (int): Maximum pixel area for valid star.

    Returns:
        List[dict]: Each entry contains 'id', 'x', 'y', 'r' (radius), and 'b' (brightness).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    bg = cv2.GaussianBlur(img, (bg_blur, bg_blur), 0)
    data = img - bg
    med, std = np.median(data), np.std(data)
    th = med + thresh_std * std
    bw = (data > th).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_disk, open_disk))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker)
    nlab, labels, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
    stars, idx = [], 1
    for lab in range(1, nlab):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        cx, cy = cents[lab]
        flux = float(np.sum(data[labels == lab]))
        r = math.sqrt(area / math.pi)
        stars.append({'id': idx, 'x': cx, 'y': cy, 'r': r, 'b': flux})
        idx += 1
    return stars

# -----------------------------------------------------------------------------
# Section 3: Annotation and I/O Helpers
# -----------------------------------------------------------------------------
# Functions to annotate detected and matched stars on images,
# and to write detection coordinates and match results to CSV files.

def annotate_all(src, stars, dst):
    """
    Draw circles and IDs on all detected stars and save annotated image.

    Args:
        src (str): Path to the source image.
        stars (List[dict]): Detected stars list.
        dst (str): Path to save the annotated image.
    """
    img = cv2.imread(src)
    for s in stars:
        x, y, r = int(round(s['x'])), int(round(s['y'])), int(round(s['r']))
        cv2.circle(img, (x, y), r, (0, 255, 255), 1)
        cv2.putText(img, str(s['id']), (x + r + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(dst, img)

def match_stars(small, large, dist_tol_factor=1.2, abs_tol=5.0, match_threshold=0.6):
    """
    Find the best alignment (scale, rotation, translation) between small and large star sets.

    Args:
        small (List[dict]): Stars from the small image.
        large (List[dict]): Stars from the large image.
        dist_tol_factor (float): Tolerance factor relative to star size.
        abs_tol (float): Absolute distance tolerance (pixels).
        match_threshold (float): Fraction of small stars required for early stop.

    Returns:
        dict: Mapping from small star IDs to large star IDs.
    """
    best, best_score = {}, 0
    N_small = len(small)
    min_good = math.ceil(match_threshold * N_small)
    large_pts = np.array([[l['x'], l['y']] for l in large], dtype=np.float64)

    for p1, p2 in combinations(small, 2):
        xs1, ys1 = p1['x'], p1['y']
        xs2, ys2 = p2['x'], p2['y']
        ds = math.hypot(xs2 - xs1, ys2 - ys1)
        if ds == 0:
            continue
        ang_s = math.atan2(ys2 - ys1, xs2 - xs1)

        for l1, l2 in combinations(large, 2):
            xl1, yl1, xl2, yl2 = l1['x'], l1['y'], l2['x'], l2['y']
            dl = math.hypot(xl2 - xl1, yl2 - yl1)
            if dl == 0:
                continue
            scale = dl / ds
            ang_l = math.atan2(yl2 - yl1, xl2 - xl1)
            theta = ang_l - ang_s
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            curr = {}
            for s in small:
                dx, dy = s['x'] - xs1, s['y'] - ys1
                xr = cos_t * dx - sin_t * dy
                yr = sin_t * dx + cos_t * dy
                xp, yp = xr * scale + xl1, yr * scale + yl1
                tol = s['r'] * scale * dist_tol_factor + abs_tol
                dists = np.hypot(large_pts[:, 0] - xp, large_pts[:, 1] - yp)
                j = int(np.argmin(dists))
                if dists[j] <= tol:
                    curr[s['id']] = large[j]['id']

            score = len(curr)
            if score > best_score:
                best_score, best = score, curr.copy()
                if score >= min_good:
                    return best

    return best

def annotate_matched_small(src, small, matches, dst):
    """
    Draw circles and IDs on matched stars in the small image and save.

    Args:
        src (str): Path to the small source image.
        small (List[dict]): Detected small stars list.
        matches (dict): Mapping from small IDs to large IDs.
        dst (str): Path to save the annotated small matched image.
    """
    img = cv2.imread(src)
    for s in small:
        sid = s['id']
        if sid not in matches:
            continue
        x, y, r = int(round(s['x'])), int(round(s['y'])), int(round(s['r']))
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.putText(img, str(sid), (x + r + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(dst, img)

def annotate_matched_large(src, large, matches, dst):
    """
    Draw circles and IDs on matched stars in the large image and save.

    Args:
        src (str): Path to the large source image.
        large (List[dict]): Detected large stars list.
        matches (dict): Mapping from small IDs to large IDs.
        dst (str): Path to save the annotated large matched image.
    """
    inv = {v: k for k, v in matches.items()}
    img = cv2.imread(src)
    for l in large:
        lid = l['id']
        if lid not in inv:
            continue
        sid = inv[lid]
        x, y, r = int(round(l['x'])), int(round(l['y'])), int(round(l['r']))
        cv2.circle(img, (x, y), r, (0, 0, 255), 2)
        cv2.putText(img, str(sid), (x + r + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(dst, img)

def write_coords(path, stars):
    """
    Write detected star coordinates to a CSV file.

    Args:
        path (str): CSV file path.
        stars (List[dict]): Detected stars list.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(['id', 'x', 'y', 'r', 'b'])
        for s in stars:
            w.writerow([s['id'], s['x'], s['y'], s['r'], s['b']])

def write_matches(path, matches, small, large):
    """
    Write matching results to a CSV file.

    Args:
        path (str): CSV file path.
        matches (dict): Mapping from small IDs to large IDs.
        small (List[dict]): Detected small stars list.
        large (List[dict]): Detected large stars list.
    """
    sm = {s['id']: s for s in small}
    lg = {l['id']: l for l in large}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["small_id", "small_x", "small_y", "large_id", "large_x", "large_y"])
        for sid, lid in matches.items():
            S, L = sm[sid], lg[lid]
            w.writerow([sid, S['x'], S['y'], lid, L['x'], L['y']])

# -----------------------------------------------------------------------------
# Section 4: Graphical User Interface with PySide6
# -----------------------------------------------------------------------------
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QStackedWidget, QProgressBar
)
from PySide6.QtGui import QPixmap, QFont, QDesktopServices
from PySide6.QtCore import Qt, QTimer, QUrl

class StarMatcherUI(QWidget):
    """
    Main application window for the Star Matcher UI.
    Handles navigation through welcome, image selection, classification,
    loading, and results screens.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("גילוי והתאמת כוכבים")
        self.resize(900, 650)

        # Central stacked layout for multiple screens
        self.stack = QStackedWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)

        # Paths and detection results placeholders
        self.small_path = INPUT_SMALL
        self.large_path = INPUT_LARGE
        self.small_detection = None
        self.large_detection = None
        self.matches = None

        # Initialize screens in order
        self.welcome_screen()
        self.image_select_screen()
        self.classify_screen()
        self.match_screen()
        self.loading_screen()

        # Start on welcome screen
        self.stack.setCurrentIndex(0)

    # Screen 0: Welcome message
    def welcome_screen(self):
        w = QWidget()
        v = QVBoxLayout(w)
        lbl = QLabel("ברוכים הבאים למערכת גילוי והתאמת כוכבים")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("Arial", 18))

        txt = QLabel(
            "במערכת זו נזהה כוכבים בתמונות, נבחר את הזיהוי הטוב ביותר,\n"
            "ולאחר מכן נבצע התאמה בין מערכי הכוכבים.\n"
            "בסיום תוכלו להוריד את הנתונים לצורך ניתוח נוסף."
        )
        txt.setAlignment(Qt.AlignCenter)
        txt.setFont(QFont("Arial", 12))

        btn = QPushButton("התחל")
        btn.setFixedWidth(200)
        btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))

        v.addStretch()
        v.addWidget(lbl)
        v.addWidget(txt)
        v.addWidget(btn, alignment=Qt.AlignCenter)
        v.addStretch()
        self.stack.addWidget(w)

    # Screen 1: Select small and large images
    def image_select_screen(self):
        w = QWidget()
        v = QVBoxLayout(w)
        h = QHBoxLayout()

        # Large image preview
        col_large = QVBoxLayout()
        t1 = QLabel("תמונה גדולה")
        t1.setAlignment(Qt.AlignCenter)
        self.lbl_large = QLabel()
        self.lbl_large.setPixmap(
            QPixmap(self.large_path).scaled(300, 300, Qt.KeepAspectRatio)
        )
        col_large.addWidget(t1)
        col_large.addWidget(self.lbl_large)

        # Small image preview
        col_small = QVBoxLayout()
        t2 = QLabel("תמונה קטנה")
        t2.setAlignment(Qt.AlignCenter)
        self.lbl_small = QLabel()
        self.lbl_small.setPixmap(
            QPixmap(self.small_path).scaled(300, 300, Qt.KeepAspectRatio)
        )
        col_small.addWidget(t2)
        col_small.addWidget(self.lbl_small)

        h.addLayout(col_large)
        h.addLayout(col_small)

        # Buttons to change images and proceed
        btnL = QPushButton("שנה תמונה גדולה")
        btnL.clicked.connect(self.change_large)
        btnS = QPushButton("שנה תמונה קטנה")
        btnS.clicked.connect(self.change_small)
        btnN = QPushButton("המשך לגילוי כוכבים")
        btnN.clicked.connect(self.go_to_classify)

        v.addLayout(h)
        v.addWidget(btnL)
        v.addWidget(btnS)
        v.addWidget(btnN)
        self.stack.addWidget(w)

    def change_large(self):
        """
        Open file dialog to select a new large image,
        then update the preview.
        """
        fn, _ = QFileDialog.getOpenFileName(
            self, "בחר תמונה גדולה", "", "Images (*.png *.jpg)"
        )
        if fn:
            self.large_path = fn
            self.lbl_large.setPixmap(
                QPixmap(fn).scaled(300, 300, Qt.KeepAspectRatio)
            )

    def change_small(self):
        """
        Open file dialog to select a new small image,
        then update the preview.
        """
        fn, _ = QFileDialog.getOpenFileName(
            self, "בחר תמונה קטנה", "", "Images (*.png *.jpg)"
        )
        if fn:
            self.small_path = fn
            self.lbl_small.setPixmap(
                QPixmap(fn).scaled(300, 300, Qt.KeepAspectRatio)
            )

    # Screen 2: Classification step to choose detection thresholds
    def classify_screen(self):
        """
        Display original and processed images side by side,
        with controls to accept or reject current threshold.
        """
        w = QWidget()
        v = QVBoxLayout(w)

        self.lbl_question = QLabel("האם הסיווג הזה מספק אותך או לחלופין להחליף?")
        self.lbl_question.setAlignment(Qt.AlignCenter)
        self.lbl_question.setFont(QFont("Arial", 14))

        h_imgs = QHBoxLayout()
        self.orig_img = QLabel()
        self.orig_img.setAlignment(Qt.AlignCenter)
        self.class_img = QLabel()
        self.class_img.setAlignment(Qt.AlignCenter)
        h_imgs.addWidget(self.orig_img)
        h_imgs.addWidget(self.class_img)

        h_btn = QHBoxLayout()
        btn_x = QPushButton("✗")
        btn_x.clicked.connect(self.next_class_option)
        btn_v = QPushButton("✓")
        btn_v.clicked.connect(self.accept_class)
        h_btn.addStretch()
        h_btn.addWidget(btn_x)
        h_btn.addWidget(btn_v)
        h_btn.addStretch()

        v.addWidget(self.lbl_question)
        v.addLayout(h_imgs)
        v.addLayout(h_btn)
        self.stack.addWidget(w)

        # Precompute threshold options for large and small detection
        orig_large, step_large = 0.8, 0.2
        self.large_opts = [orig_large + (i - 2) * step_large for i in range(5)]
        orig_small, step_small = 5.0, 1.0
        self.small_opts = [orig_small + (i - 2) * step_small for i in range(5)]
        self.is_large_phase = True
        self.current_idx = 2  # start at the original threshold

    # Screen 3: Display matched results and CSV download buttons
    def match_screen(self):
        """
        Show the final matched small and large images,
        and provide buttons to open the CSV files.
        """
        w = QWidget()
        v = QVBoxLayout(w)
        lbl = QLabel("תוצאות התאמת כוכבים")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("Arial", 14))

        h = QHBoxLayout()
        # Small matched image column
        col_s = QVBoxLayout()
        t_s = QLabel("תמונה קטנה מותאמת")
        t_s.setAlignment(Qt.AlignCenter)
        self.res_small = QLabel()
        self.res_small.setAlignment(Qt.AlignCenter)
        col_s.addWidget(t_s)
        col_s.addWidget(self.res_small)
        # Large matched image column
        col_l = QVBoxLayout()
        t_l = QLabel("תמונה גדולה מותאמת")
        t_l.setAlignment(Qt.AlignCenter)
        self.res_large = QLabel()
        self.res_large.setAlignment(Qt.AlignCenter)
        col_l.addWidget(t_l)
        col_l.addWidget(self.res_large)
        h.addLayout(col_s)
        h.addLayout(col_l)

        # CSV download buttons
        btn1 = QPushButton("הורד CSV – small")
        btn1.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(SMALL_CSV)))
        btn2 = QPushButton("הורד CSV – large")
        btn2.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(LARGE_CSV)))
        btn3 = QPushButton("הורד CSV – matches")
        btn3.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(MATCHES_CSV)))

        v.addWidget(lbl)
        v.addLayout(h)
        v.addWidget(btn1)
        v.addWidget(btn2)
        v.addWidget(btn3)
        self.stack.addWidget(w)

    # Screen 4: Loading/progress bar while processing
    def loading_screen(self):
        w = QWidget()
        v = QVBoxLayout(w)
        lbl = QLabel("אנא המתן, מנתח את הנתונים...")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("Arial", 14))
        self.pb = QProgressBar()
        self.pb.setRange(0, 0)  # Indeterminate marquee mode
        v.addStretch()
        v.addWidget(lbl)
        v.addWidget(self.pb)
        v.addStretch()
        self.stack.addWidget(w)

    # Transition from image select to classification
    def go_to_classify(self):
        """
        Prepare for classification by showing the loading screen
        and then updating to the first threshold choice.
        """
        self.is_large_phase = True
        self.current_idx = 2
        self.update_class_images()

    def update_class_images(self):
        """
        Trigger a brief loading delay, then process
        the current threshold choice and update the images.
        """
        self.stack.setCurrentIndex(4)
        QApplication.processEvents()
        QTimer.singleShot(50, self._run_classification)

    def _run_classification(self):
        """
        Perform star detection with the current threshold,
        annotate the result, and update the classification screen.
        """
        if self.is_large_phase:
            val = self.large_opts[self.current_idx]
            stars = detect_cc(self.large_path, thresh_std=val)
            annotate_all(self.large_path, stars, LARGE_DETECTED)
            src, dst = self.large_path, LARGE_DETECTED
        else:
            val = self.small_opts[self.current_idx]
            stars = detect_sep(self.small_path, thresh_sigma=val)
            annotate_all(self.small_path, stars, SMALL_DETECTED)
            src, dst = self.small_path, SMALL_DETECTED

        pix1 = QPixmap(src).scaled(400, 400, Qt.KeepAspectRatio)
        pix2 = QPixmap(dst).scaled(400, 400, Qt.KeepAspectRatio)
        self.orig_img.setPixmap(pix1)
        self.class_img.setPixmap(pix2)
        self.stack.setCurrentIndex(2)

    def next_class_option(self):
        """
        Cycle through threshold options and update images.
        """
        self.current_idx = (self.current_idx + 1) % len(self.large_opts)
        self.update_class_images()

    def accept_class(self):
        """
        Accept the current phase threshold:
        - First accept the large detection, then proceed to small.
        """
        if self.is_large_phase:
            self.large_detection = detect_cc(self.large_path,
                                             thresh_std=self.large_opts[self.current_idx])
            self.is_large_phase = False
            self.current_idx = 2
            self.update_class_images()
        else:
            self.small_detection = detect_sep(self.small_path,
                                              thresh_sigma=self.small_opts[self.current_idx])
            self.perform_matching()

    def perform_matching(self):
        """
        Start the matching process with a brief loading delay.
        """
        self.stack.setCurrentIndex(4)
        QApplication.processEvents()
        QTimer.singleShot(50, self._run_matching)

    def _run_matching(self):
        """
        Write detection CSVs, compute matches, write match CSV,
        annotate matched images, and display results.
        """
        write_coords(SMALL_CSV, self.small_detection)
        write_coords(LARGE_CSV, self.large_detection)
        self.matches = match_stars(self.small_detection, self.large_detection)
        write_matches(MATCHES_CSV, self.matches,
                      self.small_detection, self.large_detection)
        annotate_matched_small(self.small_path,
                               self.small_detection, self.matches, SMALL_MATCHED)
        annotate_matched_large(self.large_path,
                               self.large_detection, self.matches, LARGE_MATCHED)
        self.show_matching_images()
        self.stack.setCurrentIndex(3)

    def show_matching_images(self):
        """
        Load and display the final matched images in the results screen.
        """
        pix_s = QPixmap(SMALL_MATCHED).scaled(350, 350, Qt.KeepAspectRatio)
        pix_l = QPixmap(LARGE_MATCHED).scaled(350, 350, Qt.KeepAspectRatio)
        self.res_small.setPixmap(pix_s)
        self.res_large.setPixmap(pix_l)

if __name__ == "__main__":
    """
    Entry point: launch the Qt application.
    """
    app = QApplication(sys.argv)
    ui = StarMatcherUI()
    ui.show()
    sys.exit(app.exec())
