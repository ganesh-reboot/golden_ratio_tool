#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InsightFace batch exporter with MediaPipe fallback (and optional YOLO anime-face).

Multi-scale detection with fresh detector per size (default):
  --det-sizes auto|480,640,1024
  --det-policy first-hit|best
  --det-reuse  (opt-in to reuse per-size detectors across images for speed)
  --min-face-area-pct 0..1

Computed per image:
- Ratios (length/width, thirds, phi proxy), IPD (inner/outer), eye→edge, canthal tilt
- Jaw angle & symmetry (when 68 is available), shape heuristic
- Head pose: yaw / pitch / roll
- Quality: face_area_pct, sharpness_laplacian, brightness_mean
- Overlays + landmark JSON dumps

Outputs:
- Master per-image CSV
- Legacy per-person summary (means)
- NEW consolidated per-person CSV (robust medians, counts, best sample, golden score)
"""

import argparse, csv, json, math, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2
from tqdm import tqdm

# Quiet ORT logs
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except Exception:
    pass

from insightface.app import FaceAnalysis

# --------- Optional YOLO (only imported if --anime-weights is given) ----------
_YOLO = None
def _lazy_load_yolo(weights_path: str):
    global _YOLO
    if _YOLO is None:
        try:
            from ultralytics import YOLO
            _YOLO = YOLO(weights_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO weights: {e}")

# --------- MediaPipe FaceMesh (fallback landmarks) ----------
try:
    import mediapipe as mp
    _mp_fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, refine_landmarks=True, max_num_faces=1
    )
except Exception:
    _mp_fm = None  # only used if installed

PHI = 1.618033988749895
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ------------------------ filesystem helpers ------------------------

def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS]

def people_under(root: Path) -> List[Tuple[str, List[Path]]]:
    people: List[Tuple[str, List[Path]]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = [p for p in sub.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if imgs:
            people.append((sub.name, imgs))
    if not people:
        top_imgs = list_images(root)
        if top_imgs:
            people.append((root.name, top_imgs))
    return people

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ------------------------ small math helpers ------------------------

def dist(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))

def angle(p1, p2, p3) -> float:
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p2)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

def midpoint(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return (a + b) / 2.0

def phi_closeness(x: float) -> float:
    return 1.0 - abs(x - PHI) / PHI

# ------------------------ quality metrics ------------------------

def face_area_pct(bbox, img_shape) -> Optional[float]:
    if bbox is None: return None
    x1,y1,x2,y2 = bbox
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    area = w * h
    H, W = img_shape[:2]
    total = float(W*H) if W>0 and H>0 else 0.0
    return float(area/total) if total>0 else None

def sharpness_laplacian(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def brightness_mean(img_bgr) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[...,2]))

# ------------------------ metrics: 68 landmarks ------------------------

def metrics_from_68(lm68: np.ndarray) -> Dict[str, Any]:
    pts = np.asarray(lm68, dtype=float)
    if pts.shape[1] == 3:
        pts = pts[:, :2]

    jaw_left, jaw_right = pts[0], pts[16]
    chin = pts[8]
    brow_left_o, brow_right_o = pts[17], pts[26]
    nose_top, nose_base = pts[27], pts[33]
    nose_tip = pts[30]
    nose_left, nose_right = pts[31], pts[35]
    mouth_left, mouth_right = pts[48], pts[54]
    leye_outer, leye_inner = pts[36], pts[39]
    reye_inner, reye_outer = pts[42], pts[45]

    face_width = dist(jaw_left, jaw_right)
    upper = dist(midpoint(brow_left_o, brow_right_o), nose_top)
    mid   = dist(nose_top, nose_base)
    lower = dist(nose_base, chin)
    face_h = upper + mid + lower

    eye_wL = dist(leye_outer, leye_inner)
    eye_wR = dist(reye_outer, reye_inner)
    ipd_outer = dist(leye_outer, reye_outer)
    ipd_inner = dist(leye_inner, reye_inner)
    nose_w = dist(nose_left, nose_right)
    mouth_w = dist(mouth_left, mouth_right)
    jaw_w = dist(pts[5], pts[11])

    jaw_angle_L = angle(pts[3], pts[5], pts[7])
    jaw_angle_R = angle(pts[13], pts[11], pts[9])
    jaw_angle = (jaw_angle_L + jaw_angle_R) / 2.0

    def tilt(a, b):
        dx = a[0] - b[0]; dy = a[1] - b[1]
        return -math.degrees(math.atan2(dy, dx))
    canthal_tilt = (tilt(leye_outer, leye_inner) + tilt(reye_outer, reye_inner)) / 2.0

    midx = midpoint(jaw_left, jaw_right)[0]
    left_pairs  = [0,1,2,3,4,36,37,38,39,31,48,49,50]
    right_pairs = [16,15,14,13,12,45,44,43,42,35,54,53,52]
    sym_err = 0.0
    for li, ri in zip(left_pairs, right_pairs):
        lx, ly = pts[li]; rx, ry = pts[ri]
        lx_ref = 2*midx - lx
        sym_err += abs(lx_ref - rx) + abs(ly - ry)
    sym_err /= len(left_pairs)

    ratios = {
        "length_to_width": face_h / (face_width + 1e-9),
        "upper_frac": upper / (face_h + 1e-9),
        "mid_frac":   mid   / (face_h + 1e-9),
        "lower_frac": lower / (face_h + 1e-9),
        "ipd_outer_to_face_width": ipd_outer / (face_width + 1e-9),
        "ipd_inner_to_face_width": ipd_inner / (face_width + 1e-9),
        "nose_to_face_width": nose_w / (face_width + 1e-9),
        "mouth_to_face_width": mouth_w / (face_width + 1e-9),
        "eyeL_to_ipd": eye_wL / (ipd_outer + 1e-9),
        "eyeR_to_ipd": eye_wR / (ipd_outer + 1e-9),
        "jaw_to_face_width": jaw_w / (face_width + 1e-9),
        "phi_closeness_length_width": phi_closeness(face_h / (face_width + 1e-9)),
    }

    xmin = float(np.min(pts[:,0])); xmax = float(np.max(pts[:,0]))
    left_eye_to_left_edge = leye_outer[0] - xmin
    right_eye_to_right_edge = xmax - reye_outer[0]

    lw = ratios["length_to_width"]
    brow_w = dist(brow_left_o, brow_right_o)
    shape = "oval"
    if lw >= 1.55 and jaw_angle < 130: shape = "oblong"
    elif lw <= 1.20 and jaw_angle < 130: shape = "round"
    elif lw <= 1.40 and 0.95 <= (brow_w/(jaw_w+1e-9)) <= 1.05: shape = "square"
    elif brow_w > jaw_w * 1.10 and lw >= 1.3: shape = "heart"

    return {
        "face_width": face_width,
        "face_height_proxy": face_h,
        "jaw_angle_deg": jaw_angle,
        "canthal_tilt_deg": canthal_tilt,
        "symmetry_error_px": sym_err,
        "ipd_outer_px": ipd_outer,
        "ipd_inner_px": ipd_inner,
        "left_eye_to_left_edge_px": left_eye_to_left_edge,
        "right_eye_to_right_edge_px": right_eye_to_right_edge,
        "left_eye_to_left_edge_norm": left_eye_to_left_edge/(face_width+1e-9),
        "right_eye_to_right_edge_norm": right_eye_to_right_edge/(face_width+1e-9),
        "ratios": ratios,
        "shape_guess": shape,
        "eye_points": {
            "le_outer": leye_outer.tolist(), "le_inner": leye_inner.tolist(),
            "re_inner": reye_inner.tolist(), "re_outer": reye_outer.tolist()
        },
        "pose_points_2d": {
            "nose_tip": nose_tip.tolist(),
            "chin": pts[8].tolist(),
            "le_outer": leye_outer.tolist(),
            "re_outer": reye_outer.tolist(),
            "mouth_left": mouth_left.tolist(),
            "mouth_right": mouth_right.tolist(),
        },
    }

# ------------------------ metrics: 106 landmarks ------------------------

def _extrema_face_width(pts: np.ndarray) -> Tuple[float, float, float]:
    xs = pts[:,0]
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    return xmin, xmax, xmax - xmin

def _cluster_eye_points(lm106: np.ndarray, kps_eye: np.ndarray, face_width: float) -> Optional[np.ndarray]:
    pts = lm106[:, :2]; center = kps_eye[:2]
    r = max(8.0, 0.22 * face_width)
    d = np.linalg.norm(pts - center, axis=1)
    cluster = pts[d <= r]
    if len(cluster) < 3:
        r2 = max(12.0, 0.28 * face_width)
        cluster = pts[np.linalg.norm(pts - center, axis=1) <= r2]
    return cluster if len(cluster) >= 2 else None

def metrics_from_106(lm106: np.ndarray, kps5: Optional[np.ndarray]) -> Dict[str, Any]:
    pts = np.asarray(lm106, dtype=float)
    if pts.shape[1] == 3: pts = pts[:, :2]

    xmin, xmax, face_width = _extrema_face_width(pts)
    ymin, ymax = float(np.min(pts[:,1])), float(np.max(pts[:,1]))
    face_height = ymax - ymin

    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1,2)
    y_upper = ymin + 0.30 * face_height
    y_lower = ymin + 0.75 * face_height
    upper_band = hull[hull[:,1] <= y_upper]
    lower_band = hull[hull[:,1] >= y_lower]

    def farthest_pair(arr: np.ndarray) -> Optional[float]:
        if arr is None or len(arr) < 2: return None
        maxd = 0.0
        for i in range(len(arr)-1):
            d = np.max(np.linalg.norm(arr[i] - arr[i+1:], axis=1)) if i+1 < len(arr) else 0.0
            if d > maxd: maxd = float(d)
        return maxd if maxd > 0 else None

    brow_w = farthest_pair(upper_band)
    jaw_w  = farthest_pair(lower_band)

    if upper_band is not None and len(upper_band) >= 2:
        browL = upper_band[np.argmin(upper_band[:,0])]
        browR = upper_band[np.argmax(upper_band[:,0])]
        brow_mid = midpoint(browL, browR)
    else:
        brow_mid = np.array([ (xmin+xmax)/2.0, y_upper ])

    xmed = np.median(pts[:,0])
    near_mid = pts[np.argsort(np.abs(pts[:,0]-xmed))[:30]]
    nose_base = near_mid[np.argmax(near_mid[:,1])]
    above = pts[pts[:,1] < nose_base[1]]
    nose_top = above[np.argmin(np.abs(above[:,0]-xmed))] if len(above) else np.array([xmed, ymin + 0.4*face_height])
    chin = pts[np.argmax(pts[:,1])]

    upper = dist(brow_mid, nose_top)
    mid   = dist(nose_top, nose_base)
    lower = dist(nose_base, chin)
    face_h_proxy = upper + mid + lower

    le_outer = re_outer = le_inner = re_inner = None
    ipd_outer = ipd_inner = None
    canthal_tilt = None

    if kps5 is not None and kps5.shape[0] >= 2:
        eyeL_center = kps5[0, :2]; eyeR_center = kps5[1, :2]
        left_cluster  = _cluster_eye_points(pts, eyeL_center, face_width)
        right_cluster = _cluster_eye_points(pts, eyeR_center, face_width)
        if left_cluster is not None:
            le_outer = left_cluster[np.argmin(left_cluster[:,0])]
            le_inner = left_cluster[np.argmax(left_cluster[:,0])]
        else:
            le_outer = eyeL_center; le_inner = eyeL_center
        if right_cluster is not None:
            re_inner = right_cluster[np.argmin(right_cluster[:,0])]
            re_outer = right_cluster[np.argmax(right_cluster[:,0])]
        else:
            re_outer = eyeR_center; re_inner = eyeR_center

        ipd_outer = dist(le_outer, re_outer) if (le_outer is not None and re_outer is not None) else None
        ipd_inner = dist(le_inner, re_inner) if (le_inner is not None and re_inner is not None) else None

        def tilt(a, b):
            dx = a[0]-b[0]; dy = a[1]-b[1]
            return -math.degrees(math.atan2(dy, dx))
        if (le_outer is not None and le_inner is not None and
            re_outer is not None and re_inner is not None):
            canthal_tilt = (tilt(le_outer, le_inner) + tilt(re_outer, re_inner)) / 2.0

    left_eye_to_left_edge  = (le_outer[0] - xmin) if le_outer is not None else None
    right_eye_to_right_edge = (xmax - re_outer[0]) if re_outer is not None else None

    ratios = {
        "length_to_width": face_h_proxy / (face_width + 1e-9) if face_width > 0 else None,
        "upper_frac": upper / (face_h_proxy + 1e-9) if face_h_proxy > 0 else None,
        "mid_frac":   mid   / (face_h_proxy + 1e-9) if face_h_proxy > 0 else None,
        "lower_frac": lower / (face_h_proxy + 1e-9) if face_h_proxy > 0 else None,
        "phi_closeness_length_width": phi_closeness(face_h_proxy / (face_width + 1e-9)) if face_width > 0 else None,
    }

    shape = None
    if ratios["length_to_width"] is not None and brow_w is not None and jaw_w is not None:
        lw = ratios["length_to_width"]; bj = brow_w/(jaw_w+1e-9)
        if bj >= 1.10 and lw >= 1.3: shape = "heart"
        elif lw <= 1.20 and bj >= 1.0: shape = "round"
        elif 1.20 < lw <= 1.40 and 0.95 <= bj <= 1.05: shape = "square"
        elif lw >= 1.50: shape = "oblong"
        else: shape = "oval"

    nose_tip = kps5[2, :2].tolist() if kps5 is not None and kps5.shape[0] >= 3 else [xmed, (ymin+ymax)/2]
    mouth_left = kps5[3, :2].tolist() if kps5 is not None and kps5.shape[0] >= 5 else [xmin, ymax-0.2*face_height]
    mouth_right = kps5[4, :2].tolist() if kps5 is not None and kps5.shape[0] >= 5 else [xmax, ymax-0.2*face_height]
    chin_pt = pts[np.argmax(pts[:,1])].tolist()

    return {
        "face_width": face_width,
        "face_height_proxy": face_h_proxy,
        "jaw_angle_deg": None,
        "canthal_tilt_deg": canthal_tilt,
        "symmetry_error_px": None,
        "ipd_outer_px": ipd_outer,
        "ipd_inner_px": ipd_inner,
        "left_eye_to_left_edge_px": left_eye_to_left_edge,
        "right_eye_to_right_edge_px": right_eye_to_right_edge,
        "left_eye_to_left_edge_norm": (left_eye_to_left_edge/(face_width+1e-9)) if (left_eye_to_left_edge is not None and face_width>0) else None,
        "right_eye_to_right_edge_norm": (right_eye_to_right_edge/(face_width+1e-9)) if (right_eye_to_right_edge is not None and face_width>0) else None,
        "ratios": ratios,
        "shape_guess": shape,
        "eye_points": {
            "le_outer": le_outer.tolist() if le_outer is not None else None,
            "le_inner": le_inner.tolist() if le_inner is not None else None,
            "re_inner": re_inner.tolist() if re_inner is not None else None,
            "re_outer": re_outer.tolist() if re_outer is not None else None,
        },
        "face_bbox_proxy": [xmin, ymin, xmax, ymax],
        "brow_band_width_px": brow_w,
        "jaw_band_width_px": jaw_w,
        "pose_points_2d": {
            "nose_tip": nose_tip,
            "chin": chin_pt,
            "le_outer": (le_outer.tolist() if le_outer is not None else [xmin, ymin]),
            "re_outer": (re_outer.tolist() if re_outer is not None else [xmax, ymin]),
            "mouth_left": mouth_left,
            "mouth_right": mouth_right,
        },
    }

# ------------------------ metrics: MediaPipe 468 landmarks ------------------------

MP_IDX = dict(nose_tip=1, chin=152, le_outer=33, re_outer=263, mouth_left=61, mouth_right=291)

def metrics_from_468(pts468: np.ndarray) -> Dict[str, Any]:
    pts = np.asarray(pts468, dtype=float)
    if pts.shape[1] == 3:
        pts = pts[:, :2]

    xs, ys = pts[:,0], pts[:,1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    face_width = xmax - xmin
    face_height = ymax - ymin

    le_outer = pts[MP_IDX["le_outer"]]
    re_outer = pts[MP_IDX["re_outer"]]
    nose_tip = pts[MP_IDX["nose_tip"]]
    chin = pts[MP_IDX["chin"]]
    mouth_left = pts[MP_IDX["mouth_left"]]
    mouth_right = pts[MP_IDX["mouth_right"]]

    ipd_outer = dist(le_outer, re_outer)

    brow_mid_y = ymin + 0.30 * face_height
    brow_mid = np.array([ (xmin+xmax)/2.0, brow_mid_y ])
    nose_top = nose_tip - np.array([0.0, 0.07*face_height])  # approx
    nose_base = np.array([(mouth_left[0]+mouth_right[0])/2.0, (mouth_left[1]+mouth_right[1])/2.0])

    upper = dist(brow_mid, nose_top)
    mid   = dist(nose_top, nose_base)
    lower = dist(nose_base, chin)
    face_h_proxy = upper + mid + lower

    def tilt(a, b):
        dx = a[0] - b[0]; dy = a[1] - b[1]
        return -math.degrees(math.atan2(dy, dx))
    canthal_tilt = tilt(le_outer, re_outer)

    left_eye_to_left_edge = le_outer[0] - xmin
    right_eye_to_right_edge = xmax - re_outer[0]

    ratios = {
        "length_to_width": face_h_proxy/(face_width+1e-9) if face_width>0 else None,
        "upper_frac": upper/(face_h_proxy+1e-9) if face_h_proxy>0 else None,
        "mid_frac": mid/(face_h_proxy+1e-9) if face_h_proxy>0 else None,
        "lower_frac": lower/(face_h_proxy+1e-9) if face_h_proxy>0 else None,
        "phi_closeness_length_width": phi_closeness(face_h_proxy/(face_width+1e-9)) if face_width>0 else None,
    }

    shape = None
    lw = ratios["length_to_width"] if ratios["length_to_width"] is not None else 0
    if lw >= 1.55: shape = "oblong"
    elif lw <= 1.20: shape = "round"
    else: shape = "oval"

    return {
        "face_width": face_width,
        "face_height_proxy": face_h_proxy,
        "jaw_angle_deg": None,
        "canthal_tilt_deg": canthal_tilt,
        "symmetry_error_px": None,
        "ipd_outer_px": ipd_outer,
        "ipd_inner_px": None,
        "left_eye_to_left_edge_px": left_eye_to_left_edge,
        "right_eye_to_right_edge_px": right_eye_to_right_edge,
        "left_eye_to_left_edge_norm": (left_eye_to_left_edge/(face_width+1e-9)) if face_width>0 else None,
        "right_eye_to_right_edge_norm": (right_eye_to_right_edge/(face_width+1e-9)) if face_width>0 else None,
        "ratios": ratios,
        "shape_guess": shape,
        "eye_points": {
            "le_outer": le_outer.tolist(),
            "re_outer": re_outer.tolist(),
        },
        "pose_points_2d": {
            "nose_tip": nose_tip.tolist(),
            "chin": chin.tolist(),
            "le_outer": le_outer.tolist(),
            "re_outer": re_outer.tolist(),
            "mouth_left": mouth_left.tolist(),
            "mouth_right": mouth_right.tolist(),
        },
        "face_bbox_proxy": [xmin, ymin, xmax, ymax],
    }

def mediapipe_landmarks(img_bgr) -> Optional[Dict[str, Any]]:
    if _mp_fm is None: return None
    H, W = img_bgr.shape[:2]
    res = _mp_fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks: return None
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x*W, p.y*H] for p in lm], dtype=np.float32)
    return {"landmark_468": pts}

# ------------------------ head pose (PnP) ------------------------

MODEL_POINTS_3D = np.array([
    (0.0,   0.0,    0.0),    # nose tip
    (0.0,  -330.0, -65.0),   # chin
    (-225.0, 170.0, -135.0), # left eye outer
    (225.0, 170.0, -135.0),  # right eye outer
    (-150.0,-150.0, -125.0), # left mouth corner
    (150.0, -150.0, -125.0), # right mouth corner
], dtype=np.float64)

def estimate_pose(img_shape, pose_points_2d: Dict[str, Any]) -> Optional[Tuple[float,float,float]]:
    try:
        h, w = img_shape[:2]
        image_points = np.array([
            pose_points_2d["nose_tip"],
            pose_points_2d["chin"],
            pose_points_2d["le_outer"],
            pose_points_2d["re_outer"],
            pose_points_2d["mouth_left"],
            pose_points_2d["mouth_right"],
        ], dtype=np.float64)

        focal_length = w
        center = (w/2.0, h/2.0)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,1))
        ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return None
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        return float(yaw), float(pitch), float(roll)
    except Exception:
        return None

# ------------------------ overlays ------------------------

def draw_overlay(img: np.ndarray, face, metrics: Dict[str, Any], draw_468=False) -> np.ndarray:
    vis = img.copy()
    if hasattr(face, "bbox"):
        x1,y1,x2,y2 = face.bbox.astype(int)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
    lm = None
    if getattr(face, "landmark_3d_68", None) is not None:
        lm = face.landmark_3d_68[:, :2]
    elif getattr(face, "landmark_2d_106", None) is not None:
        lm = face.landmark_2d_106[:, :2]
    if lm is not None:
        for (x,y) in lm:
            cv2.circle(vis, (int(x), int(y)), 1, (255,0,0), -1)
    if draw_468 and isinstance(draw_468, np.ndarray):
        for (x,y) in draw_468:
            cv2.circle(vis, (int(x), int(y)), 1, (200,200,0), -1)
    ep = (metrics or {}).get("eye_points") or {}
    for k in ["le_outer","le_inner","re_inner","re_outer"]:
        p = ep.get(k)
        if p is not None:
            cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0,255,255), -1)
            cv2.putText(vis, k, (int(p[0])+4, int(p[1])-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
    if metrics.get("pose_yaw_deg") is not None:
        text = f"yaw:{metrics['pose_yaw_deg']:.1f}  pitch:{metrics['pose_pitch_deg']:.1f}  roll:{metrics['pose_roll_deg']:.1f}"
        cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return vis

# ------------------------ detector pool ------------------------

class DetectorPool:
    """
    Builds FaceAnalysis instances at requested sizes.
    By default we create a FRESH instance for each size attempt (per image),
    avoiding 'det_size already set' issues.

    If --det-reuse is given, we cache one instance per size across images (faster).
    """
    def __init__(self, ctx_id: int, allowed_modules: List[str], reuse: bool = False):
        self.ctx_id = ctx_id
        self.allowed = allowed_modules[:]
        self.reuse = reuse
        self._cache: Dict[int, FaceAnalysis] = {}

    def get_for_size(self, s: int) -> FaceAnalysis:
        if self.reuse:
            app = self._cache.get(s)
            if app is None:
                app = FaceAnalysis(allowed_modules=self.allowed)
                app.prepare(ctx_id=self.ctx_id, det_size=(s, s))
                self._cache[s] = app
            return app
        app = FaceAnalysis(allowed_modules=self.allowed)
        app.prepare(ctx_id=self.ctx_id, det_size=(s, s))
        return app

# ------------------------ detection helpers ------------------------

def parse_det_sizes(det_sizes_arg: str, img_shape) -> List[int]:
    if det_sizes_arg and det_sizes_arg.lower() != "auto":
        vals = []
        for tok in det_sizes_arg.split(","):
            try:
                s = int(tok.strip())
                if s >= 256: vals.append(s)
            except: pass
        return sorted(list(set(vals)))
    H, W = img_shape[:2]
    m = min(H, W)
    if m <= 600:
        return [384, 480, 640]
    elif m <= 1200:
        return [480, 640, 800, 1024]
    else:
        return [640, 800, 1024, 1280]

def pick_largest(faces):
    if not faces: return None
    def area(f):
        x1,y1,x2,y2 = f.bbox.astype(int)
        return max(0, x2-x1) * max(0, y2-y1)
    return max(faces, key=area)

def detect_multiscale(det_pool: DetectorPool, img: np.ndarray,
                      sizes: List[int], policy: str, min_det_score: float):
    best = None
    best_size = None
    best_faces = None
    best_score = -1.0
    for s in sizes:
        app = det_pool.get_for_size(s)
        faces = app.get(img)
        faces = [f for f in faces if float(getattr(f, "det_score", 0.0)) >= min_det_score]
        if not faces:
            continue
        cand = pick_largest(faces)
        if policy == "first-hit":
            return cand, s, faces
        x1,y1,x2,y2 = cand.bbox.astype(int)
        area = max(0, x2-x1) * max(0, y2-y1)
        score = float(getattr(cand, "det_score", 0.0)) * (area + 1e-6)
        if score > best_score:
            best_score = score
            best = cand
            best_size = s
            best_faces = faces
    if best is not None:
        return best, best_size, best_faces
    return None, None, None

# ------------------------ main per-image analysis ------------------------

def analyze_image(det_pool: DetectorPool, img_path: Path, prefer68: bool,
                  save_overlays: bool, overlays_dir: Path,
                  save_landmarks: bool, landmarks_dir: Path,
                  yaw_max: float, pitch_max: float, roll_max: float,
                  use_anime: bool, anime_weights: Optional[str],
                  det_margin: float = 0.12, args: Optional[argparse.Namespace]=None) -> Dict[str, Any]:

    rec: Dict[str, Any] = {"image": str(img_path), "error": None}
    img = cv2.imread(str(img_path))
    if img is None:
        rec["error"] = "read_error"; return rec

    H, W = img.shape[:2]

    sizes = parse_det_sizes(getattr(args, "det_sizes", "auto"), img.shape)
    face, used_size, _faces = detect_multiscale(
        det_pool, img, sizes, args.det_policy, args.min_det_score
    )
    if used_size is not None:
        rec["det_size_used"] = int(used_size)

    lm68 = lm106 = kps5 = None
    if face is not None:
        rec["det_score"] = float(getattr(face, "det_score", 0.0))
        bbox = getattr(face, "bbox", np.array([0,0,0,0], dtype=float)).astype(float)
        x1,y1,x2,y2 = bbox
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        bw, bh = (x2-x1), (y2-y1)
        x1e = max(0, int(cx - (1+det_margin)*bw/2))
        x2e = min(W, int(cx + (1+det_margin)*bw/2))
        y1e = max(0, int(cy - (1+det_margin)*bh/2))
        y2e = min(H, int(cy + (1+det_margin)*bh/2))
        rec["bbox"] = [float(x1e), float(y1e), float(x2e), float(y2e)]

        if args.min_face_area_pct > 0:
            fap = face_area_pct(rec["bbox"], img.shape)
            if fap is None or fap < args.min_face_area_pct:
                rec["error"] = f"too_small(area_pct<{args.min_face_area_pct})"
                return rec

        lm68 = getattr(face, "landmark_3d_68", None) if prefer68 else None
        lm106 = getattr(face, "landmark_2d_106", None)
        kps5  = getattr(face, "kps", None)
    else:
        rec["det_score"] = 0.0
        rec["bbox"] = None

    metrics = None
    mp468 = None

    if lm68 is not None:
        metrics = metrics_from_68(lm68.copy())
    elif lm106 is not None:
        metrics = metrics_from_106(lm106.copy(), kps5.copy() if kps5 is not None else None)
    else:
        mp_pack = mediapipe_landmarks(img)
        if mp_pack is not None:
            mp468 = mp_pack["landmark_468"]
            metrics = metrics_from_468(mp468)

    if metrics is None and use_anime and anime_weights:
        _lazy_load_yolo(anime_weights)
        try:
            results = _YOLO(img[..., ::-1], verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                b = results.boxes
                areas = (b.xyxy[:,2]-b.xyxy[:,0]) * (b.xyxy[:,3]-b.xyxy[:,1])
                idx = int(np.argmax(areas.cpu().numpy()))
                x1,y1,x2,y2 = map(int, b.xyxy[idx].cpu().numpy())
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(W,x2), min(H,y2)
                crop = img[y1:y2, x1:x2]
                mp_pack = mediapipe_landmarks(crop)
                if mp_pack is not None:
                    mp468 = mp_pack["landmark_468"]
                    mp468_full = mp468.copy()
                    mp468_full[:,0] += x1
                    mp468_full[:,1] += y1
                    metrics = metrics_from_468(mp468_full)
                    rec["bbox"] = [float(x1), float(y1), float(x2), float(y2)]
        except Exception as e:
            rec["anime_fallback_error"] = f"{type(e).__name__}:{e}"

    if metrics is None:
        rec["error"] = "no_landmarks"
        rec["quality_face_area_pct"] = face_area_pct(rec.get("bbox"), img.shape)
        rec["quality_sharpness_lap"] = sharpness_laplacian(img)
        rec["quality_brightness_mean"] = brightness_mean(img)
        return rec

    # Pose
    yaw = pitch = roll = None
    pp2d = metrics.get("pose_points_2d")
    if pp2d:
        pose = estimate_pose(img.shape, pp2d)
        if pose is not None:
            yaw, pitch, roll = pose
            rec["pose_yaw_deg"] = yaw
            rec["pose_pitch_deg"] = pitch
            rec["pose_roll_deg"] = roll
            rec["is_frontal"] = (abs(yaw) <= yaw_max and abs(pitch) <= pitch_max and abs(roll) <= roll_max)
        else:
            rec["pose_yaw_deg"] = rec["pose_pitch_deg"] = rec["pose_roll_deg"] = None
            rec["is_frontal"] = None
    else:
        rec["pose_yaw_deg"] = rec["pose_pitch_deg"] = rec["pose_roll_deg"] = None
        rec["is_frontal"] = None

    rec.update(metrics)

    # Quality
    rec["quality_face_area_pct"] = face_area_pct(rec.get("bbox"), img.shape)
    rec["quality_sharpness_lap"] = sharpness_laplacian(img)
    rec["quality_brightness_mean"] = brightness_mean(img)

    # dumps
    if save_landmarks:
        try:
            dump = {
                "bbox": rec.get("bbox"),
                "landmark_kind": "68" if lm68 is not None else ("106" if lm106 is not None else ("468" if mp468 is not None else None)),
                "landmark": (lm68[:,:2].tolist() if lm68 is not None else
                             (lm106[:,:2].tolist() if lm106 is not None else
                              (mp468.tolist() if mp468 is not None else None))),
                "kps5": (kps5[:,:2].tolist() if kps5 is not None else None),
                "pose": {"yaw": yaw, "pitch": pitch, "roll": roll}
            }
            out_json = landmarks_dir / f"{img_path.stem}_landmarks.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(dump, f)
            rec["landmarks_json"] = str(out_json)
        except Exception as e:
            rec["landmarks_json_error"] = f"{type(e).__name__}:{e}"

    if save_overlays:
        try:
            fake_face = type("F", (), {"bbox": np.array([0,0,0,0])})()
            vis = draw_overlay(img, face if face is not None else fake_face, rec,
                               draw_468=(mp468 if mp468 is not None else False))
            out_img = overlays_dir / f"{img_path.stem}_overlay.jpg"
            cv2.imwrite(str(out_img), vis)
            rec["overlay_path"] = str(out_img)
        except Exception as e:
            rec["overlay_error"] = f"{type(e).__name__}:{e}"

    return rec

# ------------------------ CSV utils & consolidation ------------------------

def avg(vals: List[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in vals if isinstance(v, (int,float))]
    return float(sum(nums)/len(nums)) if nums else None

def median(vals: List[Optional[float]]) -> Optional[float]:
    nums = sorted(float(v) for v in vals if isinstance(v, (int,float)))
    n = len(nums)
    if n == 0: return None
    mid = n//2
    if n % 2 == 1: return nums[mid]
    return (nums[mid-1] + nums[mid]) / 2.0

def mode_nonnull(vals: List[Any]) -> Optional[Any]:
    xs = [v for v in vals if v is not None]
    return max(set(xs), key=xs.count) if xs else None

def choose_best_sample(recs: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    """
    Pick one representative image:
    score = det_score * (face_area_pct+1e-6) * yaw/pitch/roll penalty
    """
    best = None
    best_score = -1.0
    for r in recs:
        if r.get("error"): continue
        det = float(r.get("det_score") or 0.0)
        area = float(r.get("quality_face_area_pct") or 0.0)
        yaw = abs(float(r.get("pose_yaw_deg")) ) if r.get("pose_yaw_deg") is not None else 0.0
        pitch = abs(float(r.get("pose_pitch_deg"))) if r.get("pose_pitch_deg") is not None else 0.0
        roll = abs(float(r.get("pose_roll_deg")) ) if r.get("pose_roll_deg") is not None else 0.0
        pose_pen = 1.0 / (1.0 + 0.01*(yaw+pitch+roll))
        s = det * (area + 1e-6) * pose_pen
        if s > best_score:
            best_score = s
            best = r
    return best

# ------------------------ CLI ------------------------

def run_pipeline(root_path: str, outdir: str, extra_options: dict | None = None):
    import argparse

    if extra_options is None:
        extra_options = {}

    args = argparse.Namespace(
        root=root_path,
        out=extra_options.get("out", "if_master.csv"),
        summary=extra_options.get("summary", "if_summary_by_person.csv"),
        consolidated=extra_options.get("consolidated", "if_consolidated.csv"),
        per_person_only=extra_options.get("per_person_only", False),
        golden_score_scale=extra_options.get("golden_score_scale", 100.0),
        outdir=outdir,
        landmarks=extra_options.get("landmarks", "auto"),
        det_sizes=extra_options.get("det_sizes", "auto"),
        det_policy=extra_options.get("det_policy", "first-hit"),
        det_reuse=extra_options.get("det_reuse", False),
        min_face_area_pct=extra_options.get("min_face_area_pct", 0.0),
        ctx_id=extra_options.get("ctx_id", -1),  # CPU default
        save_overlays=extra_options.get("save_overlays", False),
        save_landmarks=extra_options.get("save_landmarks", False),
        print_found=False,
        frontality_filter=extra_options.get("frontality_filter", False),
        yaw_max=extra_options.get("yaw_max", 50.0),
        pitch_max=extra_options.get("pitch_max", 40.0),
        roll_max=extra_options.get("roll_max", 45.0),
        min_det_score=extra_options.get("min_det_score", 0.45),
        keep_rejected=extra_options.get("keep_rejected", False),
        anime_weights=extra_options.get("anime_weights", None),
    )

    return run_core_logic(args)

def run_core_logic(args):
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr); sys.exit(2)

    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)
    overlays_dir = outdir / "overlays"
    landmarks_dir = outdir / "landmarks"
    if args.save_overlays: ensure_dir(overlays_dir)
    if args.save_landmarks: ensure_dir(landmarks_dir)

    # Configure allowed modules (detector + landmarks)
    allowed = ['detection']
    prefer68 = True
    if args.landmarks == "68":
        allowed.append('landmark_3d_68'); prefer68 = True
    elif args.landmarks == "106":
        allowed.append('landmark_2d_106'); prefer68 = False
    else:
        allowed.extend(['landmark_3d_68','landmark_2d_106']); prefer68 = True

    det_pool = DetectorPool(ctx_id=args.ctx_id, allowed_modules=allowed, reuse=args.det_reuse)

    people = people_under(root)
    if args.print_found:
        print("Found:")
        for person, imgs in people:
            print(f"  {person}: {len(imgs)} images")
    if not people:
        print("No images found under root.", file=sys.stderr); sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for person, imgs in people:
        for img in tqdm(imgs, desc=person):
            rec = analyze_image(
                det_pool, img, prefer68,
                args.save_overlays, overlays_dir,
                args.save_landmarks, landmarks_dir,
                args.yaw_max, args.pitch_max, args.roll_max,
                use_anime=bool(args.anime_weights), anime_weights=args.anime_weights,
                det_margin=0.12, args=args
            )
            rec["person"] = person

            if args.print_found and rec.get("det_size_used") is not None:
                print(f"[{person}] {img.name} -> det_size {rec['det_size_used']}, score={rec.get('det_score')}")

            # Filtering (optional)
            reason = None
            if rec.get("error"):
                reason = rec["error"]
            elif rec.get("det_score", 0) < args.min_det_score:
                reason = f"low_det_score({rec.get('det_score'):.2f})"
            elif args.frontality_filter and rec.get("is_frontal") is False:
                y, p, r = rec.get("pose_yaw_deg"), rec.get("pose_pitch_deg"), rec.get("pose_roll_deg")
                reason = f"not_frontal(y={y:.1f},p={p:.1f},r={r:.1f})" if y is not None else "pose_unavailable"

            if reason:
                rec["filter_reason"] = reason
                if args.keep_rejected:
                    rows.append(rec)
                else:
                    continue
            else:
                rows.append(rec)

    # ---- Build headers (flatten ratios.* keys) ----
    ratio_keys = set()
    for r in rows:
        rk = r.get("ratios", {})
        if isinstance(rk, dict):
            ratio_keys.update(rk.keys())
    ratio_cols = [f"ratios.{k}" for k in sorted(ratio_keys)]

    preferred_cols = [
        "person","image","error","filter_reason","det_score","bbox","det_size_used",
        "pose_yaw_deg","pose_pitch_deg","pose_roll_deg","is_frontal",
        "face_width","face_height_proxy",
        "jaw_angle_deg","canthal_tilt_deg","symmetry_error_px",
        "ipd_outer_px","ipd_inner_px",
        "left_eye_to_left_edge_px","left_eye_to_left_edge_norm",
        "right_eye_to_right_edge_px","right_eye_to_right_edge_norm",
        "brow_band_width_px","jaw_band_width_px",
        "shape_guess",
        "quality_face_area_pct","quality_sharpness_lap","quality_brightness_mean",
        "overlay_path","overlay_error","landmarks_json","landmarks_json_error",
        "eye_points"
    ]
    other_keys = set()
    for r in rows:
        other_keys.update(r.keys())
    other_keys.discard("ratios")
    cols = preferred_cols + ratio_cols + sorted([k for k in other_keys if k not in preferred_cols])

    # ---- Master CSV ----
    if not args.per_person_only:
        out_csv = outdir / args.out
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                row = r.copy()
                rk = row.pop("ratios", {})
                if isinstance(rk, dict):
                    for k, v in rk.items():
                        row[f"ratios.{k}"] = v
                if "bbox" in row and isinstance(row["bbox"], (list,tuple)):
                    row["bbox"] = json.dumps(row["bbox"])
                if "eye_points" in row and isinstance(row["eye_points"], dict):
                    row["eye_points"] = json.dumps(row["eye_points"])
                w.writerow({c: row.get(c) for c in cols})

    # ---- Group by person ----
    by_person: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows: by_person.setdefault(r.get("person","unknown"), []).append(r)

    # ---- Legacy Summary (means) ----
    out_sum = outdir / args.summary
    with open(out_sum, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "person","n_images",
            "avg_phi_closeness_length_width","avg_length_to_width",
            "avg_ipd_outer_px",
            "avg_left_eye_to_left_edge_norm","avg_right_eye_to_right_edge_norm",
            "avg_canthal_tilt_deg","avg_jaw_angle_deg","avg_symmetry_error_px",
            "avg_yaw_deg","avg_pitch_deg","avg_roll_deg",
            "avg_face_area_pct","avg_sharpness_lap","avg_brightness_mean",
            "shape_guess_mode","n_errors"
        ])
        w.writeheader()
        for person, recs in by_person.items():
            def rget(r, key):
                rk = r.get("ratios", {})
                return rk.get(key) if isinstance(rk, dict) else None
            shapes = [r.get("shape_guess") for r in recs if r.get("shape_guess")]
            shape_mode = max(set(shapes), key=shapes.count) if shapes else None
            w.writerow({
                "person": person,
                "n_images": len(recs),
                "avg_phi_closeness_length_width": avg([rget(r,"phi_closeness_length_width") for r in recs]),
                "avg_length_to_width":            avg([rget(r,"length_to_width") for r in recs]),
                "avg_ipd_outer_px":               avg([r.get("ipd_outer_px") for r in recs]),
                "avg_left_eye_to_left_edge_norm": avg([r.get("left_eye_to_left_edge_norm") for r in recs]),
                "avg_right_eye_to_right_edge_norm": avg([r.get("right_eye_to_right_edge_norm") for r in recs]),
                "avg_canthal_tilt_deg":           avg([r.get("canthal_tilt_deg") for r in recs]),
                "avg_jaw_angle_deg":              avg([r.get("jaw_angle_deg") for r in recs]),
                "avg_symmetry_error_px":          avg([r.get("symmetry_error_px") for r in recs]),
                "avg_yaw_deg":                    avg([r.get("pose_yaw_deg") for r in recs]),
                "avg_pitch_deg":                  avg([r.get("pose_pitch_deg") for r in recs]),
                "avg_roll_deg":                   avg([r.get("pose_roll_deg") for r in recs]),
                "avg_face_area_pct":              avg([r.get("quality_face_area_pct") for r in recs]),
                "avg_sharpness_lap":              avg([r.get("quality_sharpness_lap") for r in recs]),
                "avg_brightness_mean":            avg([r.get("quality_brightness_mean") for r in recs]),
                "shape_guess_mode":               shape_mode,
                "n_errors":                       sum(1 for r in recs if r.get("error"))
            })

    # ---- NEW Consolidated (medians + extras) ----
    cons_fields = [
        "person","n_images","n_used","n_errors","n_rejected","frontal_ratio",
        "best_image","best_overlay","best_det_size_used",
        "median_yaw_deg","median_pitch_deg","median_roll_deg",
        "mean_yaw_deg","mean_pitch_deg","mean_roll_deg",
        "median_face_area_pct","median_sharpness_lap","median_brightness_mean",
        "median_jaw_angle_deg","median_canthal_tilt_deg","median_symmetry_error_px",
        "shape_guess_mode","golden_ratio_score_0_100"
    ]
    # add ratio medians
    cons_fields += [f"median.{rk}" for rk in sorted(ratio_keys)]

    out_cons = outdir / args.consolidated
    with open(out_cons, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cons_fields)
        w.writeheader()

        for person, recs in by_person.items():
            used = [r for r in recs if not r.get("filter_reason")]
            n_images = len(recs)
            n_used = len(used)
            n_errors = sum(1 for r in recs if r.get("error"))
            n_rejected = n_images - n_used
            frontal = [r.get("is_frontal") for r in used if r.get("is_frontal") is not None]
            frontal_ratio = (sum(1 for v in frontal if v) / len(frontal)) if frontal else None

            # medians
            med = lambda key: median([r.get(key) for r in used])
            mea = lambda key: avg([r.get(key) for r in used])

            med_yaw   = med("pose_yaw_deg")
            med_pitch = med("pose_pitch_deg")
            med_roll  = med("pose_roll_deg")

            med_face_area = med("quality_face_area_pct")
            med_sharp     = med("quality_sharpness_lap")
            med_bright    = med("quality_brightness_mean")

            med_jaw   = med("jaw_angle_deg")
            med_canth = med("canthal_tilt_deg")
            med_sym   = med("symmetry_error_px")

            # shape mode
            shape_mode = mode_nonnull([r.get("shape_guess") for r in used])

            # ratio medians
            ratio_meds: Dict[str, Optional[float]] = {}
            for rk in ratio_keys:
                ratio_meds[rk] = median([
                    (r.get("ratios", {}) or {}).get(rk) for r in used
                ])

            phi_med = ratio_meds.get("phi_closeness_length_width")
            golden_score = (phi_med * args.golden_score_scale) if (phi_med is not None) else None

            # best sample
            best = choose_best_sample(used) if used else None
            best_image   = best.get("image") if best else None
            best_overlay = best.get("overlay_path") if best else None
            best_det     = best.get("det_size_used") if best else None

            row = {
                "person": person,
                "n_images": n_images,
                "n_used": n_used,
                "n_errors": n_errors,
                "n_rejected": n_rejected,
                "frontal_ratio": frontal_ratio,
                "best_image": best_image,
                "best_overlay": best_overlay,
                "best_det_size_used": best_det,
                "median_yaw_deg": med_yaw,
                "median_pitch_deg": med_pitch,
                "median_roll_deg": med_roll,
                "mean_yaw_deg": mea("pose_yaw_deg"),
                "mean_pitch_deg": mea("pose_pitch_deg"),
                "mean_roll_deg": mea("pose_roll_deg"),
                "median_face_area_pct": med_face_area,
                "median_sharpness_lap": med_sharp,
                "median_brightness_mean": med_bright,
                "median_jaw_angle_deg": med_jaw,
                "median_canthal_tilt_deg": med_canth,
                "median_symmetry_error_px": med_sym,
                "shape_guess_mode": shape_mode,
                "golden_ratio_score_0_100": golden_score
            }
            for rk in sorted(ratio_keys):
                row[f"median.{rk}"] = ratio_meds[rk]
            w.writerow(row)

    # ---- Done ----
    if not args.per_person_only:
        print(f"Saved master CSV  -> {outdir / args.out}")
    print(f"Saved summary CSV -> {outdir / args.summary}")
    print(f"Saved consolidated CSV -> {outdir / args.consolidated}")
    if args.save_overlays:
        print(f"Overlays         -> {outdir / 'overlays'}")
    if args.save_landmarks:
        print(f"Landmark JSON    -> {outdir / 'landmarks'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Root with subfolders per person (or images directly)")
    ap.add_argument("--out", type=str, default="if_master.csv", help="Master per-image CSV filename (in --outdir)")
    ap.add_argument("--summary", type=str, default="if_summary_by_person.csv", help="Per-person summary CSV filename (in --outdir)")
    ap.add_argument("--consolidated", type=str, default="if_consolidated.csv", help="Consolidated one-row-per-person CSV filename (in --outdir)")
    ap.add_argument("--per-person-only", action="store_true", help="Write only consolidated & summary CSVs (skip per-image master)")
    ap.add_argument("--golden-score-scale", type=float, default=100.0, help="Scale factor for phi_closeness median → golden_ratio_score_0_100")
    ap.add_argument("--outdir", type=str, default="if_out", help="Directory for all outputs (will be created)")
    ap.add_argument("--landmarks", choices=["68","106","auto"], default="auto", help="Prefer 68 if available; else 106")

    # Multi-scale controls
    ap.add_argument("--det-sizes", type=str, default="auto", help="Comma list of sizes (e.g. '384,480,640,1024') or 'auto'.")
    ap.add_argument("--det-policy", choices=["first-hit","best"], default="first-hit", help="Stop on first good scale, or evaluate all and pick best.")
    ap.add_argument("--det-reuse", action="store_true", help="Reuse detector instances per size across images (faster).")
    ap.add_argument("--min-face-area-pct", type=float, default=0.0, help="Optional: require bbox area/img area >= this (0..1).")

    ap.add_argument("--ctx-id", type=int, default=0, help="GPU id; -1 to force CPU")
    ap.add_argument("--save-overlays", action="store_true", help="Save annotated overlays (JPG) under --outdir/overlays")
    ap.add_argument("--save-landmarks", action="store_true", help="Dump landmarks/kps JSON per image under --outdir/landmarks")
    ap.add_argument("--print-found", dest="print_found", action="store_true", help="Print discovered people and image counts; also per-image det size")

    # Frontality filter controls (off by default)
    ap.add_argument("--frontality-filter", action="store_true", help="Drop non-frontal faces by yaw/pitch/roll thresholds")
    ap.add_argument("--yaw-max", type=float, default=50.0, help="Max |yaw| to keep (deg)")
    ap.add_argument("--pitch-max", type=float, default=40.0, help="Max |pitch| to keep (deg)")
    ap.add_argument("--roll-max", type=float, default=45.0, help="Max |roll| to keep (deg)")
    ap.add_argument("--min-det-score", type=float, default=0.45, help="Minimum detection score to keep")

    ap.add_argument("--keep-rejected", action="store_true", help="Keep filtered rows in CSV with filter_reason instead of dropping")

    # Optional anime/cartoon fallback
    ap.add_argument("--anime-weights", type=str, default=None, help="Path to a YOLO anime-face weights .pt file (enables cartoon fallback)")

    args = ap.parse_args()
    run_core_logic(args)

if __name__ == "__main__":
    main()
