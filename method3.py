import cv2
import numpy as np
import os
import glob
import time
import csv
from collections import defaultdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CULANE_ROOT       = r"D:\LaneProject\driver_100_30frame"
RUN_MODE          = "images"
SINGLE_IMAGE_PATH = r"D:\LaneProject\driver_100_30frame\05250322_0271.MP4\00000.jpg"
MAX_IMAGES        = None   # set to e.g. 500 for quick test

CSV_OUTPUT = "method3_results.csv"
PROC_W, PROC_H = 160, 120

NEAR_TOP, NEAR_BOT = 0.50, 1.00
FAR_TOP,  FAR_BOT  = 0.00, 0.50
NEAR_WEIGHT        = 0.70
FAR_WEIGHT         = 0.30
DEAD_ZONE_FRAC = 0.12
LANE_LOST_PCT  = 0.5   # low threshold

ERODE_KERNEL  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

POINT_TOLERANCE_PX = 10
BRIGHTNESS_NIGHT  = 60
BRIGHTNESS_SHADOW = 100

# ==============================================================================
# HELPERS
# ==============================================================================
def find_culane_images(root):
    paths = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
    return paths

def load_culane_labels(root):
    labels = {}
    txt_files = glob.glob(os.path.join(root, "**", "*.lines.txt"), recursive=True)
    total = len(txt_files)
    print(f"[LABELS] .lines.txt files found: {total}")
    with_lanes = 0
    for idx, tf in enumerate(txt_files):
        if (idx + 1) % 1000 == 0:
            print(f"  ... processed {idx+1}/{total}")
        lane_points = []
        try:
            with open(tf, "r") as f:
                for line in f:
                    vals = line.strip().split()
                    if len(vals) < 4:
                        continue
                    coords = [float(v) for v in vals]
                    pts = [(coords[i], coords[i+1]) for i in range(0, len(coords)-1, 2)]
                    if pts:
                        lane_points.append(pts)
        except Exception:
            continue
        img_path = ""
        for ext in (".jpg", ".png"):
            candidate = tf.replace(".lines.txt", ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            continue
        img_path = img_path.replace("\\", "/").lower()
        labels[img_path] = lane_points
        if lane_points:
            with_lanes += 1
    print(f"[LABELS] {with_lanes} label files had real lane points")
    print(f"[LABELS] {len(labels)} total label files loaded")
    return labels

def get_lighting_condition(original_frame):
    gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < BRIGHTNESS_NIGHT:
        return "night"
    elif brightness < BRIGHTNESS_SHADOW:
        return "shadow"
    else:
        return "day"

def build_roi_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    top_y     = int(h * 0.45)
    bot_y     = h
    top_left  = int(w * 0.20)
    top_right = int(w * 0.80)
    pts = np.array([[0, bot_y], [w, bot_y], [top_right, top_y], [top_left, top_y]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

_ROI_MASK = build_roi_mask(PROC_H, PROC_W)

def adaptive_thresholds(v_mean, condition):
    if condition == "night":
        lower_v = 10 + (v_mean / 255.0) * 50
        lower_v = float(np.clip(lower_v, 10, 80))
        upper_s = 100 - (v_mean / 255.0) * 30
        upper_s = float(np.clip(upper_s, 70, 100))
    else:
        lower_v = 20 + (v_mean / 255.0) * 100
        lower_v = float(np.clip(lower_v, 20, 140))
        upper_s = 80 - (v_mean / 255.0) * 30
        upper_s = float(np.clip(upper_s, 50, 80))
    return lower_v, upper_s

def estimate_lane_curvature(gt_lanes):
    """
    Estimate if lane is straight or curved from ground truth points.
    Returns: 'straight' or 'curve'
    Measures max horizontal deviation from straight line connecting endpoints.
    """
    if not gt_lanes:
        return 'unknown'
    
    max_curvature = 0
    
    for lane in gt_lanes:
        if len(lane) < 3:
            continue
        
        # Sort by y-coordinate (vertical)
        pts_sorted = sorted(lane, key=lambda p: p[1])
        
        # Get endpoints
        x1, y1 = pts_sorted[0]
        x2, y2 = pts_sorted[-1]
        
        if abs(y2 - y1) < 1e-6:
            continue
        
        # Measure deviation from straight line
        max_deviation = 0
        for x, y in pts_sorted:
            # Interpolate expected x on straight line
            t = (y - y1) / (y2 - y1)
            x_expected = x1 + t * (x2 - x1)
            deviation = abs(x - x_expected)
            max_deviation = max(max_deviation, deviation)
        
        max_curvature = max(max_curvature, max_deviation)
    
    # If max deviation > 15 pixels (in original image coords), consider it curved
    return 'curve' if max_curvature > 15 else 'straight'

# ==============================================================================
# DETECT FUNCTION (night: dense + thinning erosion + tiny‑component removal)
# ==============================================================================
def detect(frame, previous_x, condition):
    small = cv2.resize(frame, (PROC_W, PROC_H))
    blurred = cv2.GaussianBlur(small, (5, 5), 0)

    if condition == "night":
        # ---- CLAHE ----
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enh = clahe.apply(l)
        enhanced_lab = cv2.merge((l_enh, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # ---- HSV mask ----
        hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
        v_mean = float(hsv[:, :, 2].mean())
        lower_v, upper_s = adaptive_thresholds(v_mean, condition)
        lower = np.array([0, 0, lower_v], dtype=np.uint8)
        upper = np.array([180, upper_s, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower, upper)

        # ---- Gamma + vertical Top‑Hat ----
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        gamma = 1.8
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        gray_gamma = cv2.LUT(gray, table)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
        tophat = cv2.morphologyEx(gray_gamma, cv2.MORPH_TOPHAT, kernel_v)
        _, top_mask = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)

        # Combine
        mask = cv2.bitwise_or(mask_hsv, top_mask)

        # ---- Morphological closing + dilation ----
        kernel_close = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv2.dilate(mask, DILATE_KERNEL, iterations=1)

        # ---- Thin the mask with a small erosion (2x2) ----
        mask = cv2.erode(mask, np.ones((2,2), np.uint8), iterations=1)

        # ---- Remove tiny components (area < 10) ----
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filtered = np.zeros_like(mask)
            for cnt in contours:
                if cv2.contourArea(cnt) >= 10:
                    cv2.drawContours(filtered, [cnt], -1, 255, -1)
            mask = filtered

        # ---- Apply ROI ----
        mask = cv2.bitwise_and(mask, _ROI_MASK)

        # ---- Lane lost check ----
        white_pct = cv2.countNonZero(mask) / float(PROC_H * PROC_W) * 100.0
        if white_pct < LANE_LOST_PCT:
            return previous_x, True, white_pct, v_mean, lower_v, upper_s, mask

        # ---- Split zones ----
        near_zone = mask[int(PROC_H * NEAR_TOP):int(PROC_H * NEAR_BOT), :]
        far_zone  = mask[int(PROC_H * FAR_TOP) :int(PROC_H * FAR_BOT), :]

        # ---- Robust centroid: histogram mode ----
        def mode_centroid(zone):
            pts = cv2.findNonZero(zone)
            if pts is None or len(pts) < 5:
                return PROC_W / 2.0
            xs = pts[:, 0, 0]
            hist, _ = np.histogram(xs, bins=PROC_W, range=(0, PROC_W))
            return float(np.argmax(hist))

        near_cx = mode_centroid(near_zone)
        far_cx  = mode_centroid(far_zone)
        final_x = NEAR_WEIGHT * near_cx + FAR_WEIGHT * far_cx

        # ---- Dead zone ----
        center_x = PROC_W / 2.0
        dead_thresh = DEAD_ZONE_FRAC * PROC_W
        if abs(final_x - center_x) < dead_thresh:
            final_x = center_x

        return final_x, False, white_pct, v_mean, lower_v, upper_s, mask

    else:
        # ---- Day / Shadow ----
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        v_mean = float(hsv[:, :, 2].mean())
        lower_v, upper_s = adaptive_thresholds(v_mean, condition)
        lower = np.array([0, 0, lower_v], dtype=np.uint8)
        upper = np.array([180, upper_s, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, ERODE_KERNEL, iterations=1)
        mask = cv2.dilate(mask, DILATE_KERNEL, iterations=2)
        mask = cv2.bitwise_and(mask, _ROI_MASK)

        white_pct = cv2.countNonZero(mask) / float(PROC_H * PROC_W) * 100.0
        if white_pct < LANE_LOST_PCT:
            return previous_x, True, white_pct, v_mean, lower_v, upper_s, mask

        near_zone = mask[int(PROC_H * NEAR_TOP):int(PROC_H * NEAR_BOT), :]
        far_zone  = mask[int(PROC_H * FAR_TOP) :int(PROC_H * FAR_BOT), :]

        def mode_centroid(zone):
            pts = cv2.findNonZero(zone)
            if pts is None or len(pts) < 5:
                return PROC_W / 2.0
            xs = pts[:, 0, 0]
            hist, _ = np.histogram(xs, bins=PROC_W, range=(0, PROC_W))
            return float(np.argmax(hist))

        near_cx = mode_centroid(near_zone)
        far_cx  = mode_centroid(far_zone)
        final_x = NEAR_WEIGHT * near_cx + FAR_WEIGHT * far_cx

        center_x = PROC_W / 2.0
        dead_thresh = DEAD_ZONE_FRAC * PROC_W
        if abs(final_x - center_x) < dead_thresh:
            final_x = center_x

        return final_x, False, white_pct, v_mean, lower_v, upper_s, mask

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================
def evaluate(pred_mask, gt_lanes, orig_h, orig_w, small_h=PROC_H, small_w=PROC_W):
    if not gt_lanes:
        return None, None, None
    sx, sy = small_w / orig_w, small_h / orig_h
    tolerance = max(1, int(POINT_TOLERANCE_PX * sx))
    gt_small = [[(x * sx, y * sy) for x, y in lane] for lane in gt_lanes]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance*2+1, tolerance*2+1))
    pred_dilated = cv2.dilate(pred_mask, k)
    tp = fn = 0
    for lane in gt_small:
        for (x, y) in lane:
            xi, yi = int(x), int(y)
            if 0 <= xi < small_w and 0 <= yi < small_h:
                if pred_dilated[yi, xi] > 0:
                    tp += 1
                else:
                    fn += 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    gt_canvas = np.zeros((small_h, small_w), dtype=np.uint8)
    for lane in gt_small:
        for (x, y) in lane:
            cv2.circle(gt_canvas, (int(x), int(y)), tolerance, 255, -1)
    pred_bin  = (pred_mask  > 0).astype(np.uint8)
    gt_bin    = (gt_canvas  > 0).astype(np.uint8)
    tp_px     = int(np.sum(pred_bin & gt_bin))
    fp_px     = int(np.sum(pred_bin & (1 - gt_bin)))
    precision = tp_px / (tp_px + fp_px) if (tp_px + fp_px) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    gt_iou = np.zeros((small_h, small_w), dtype=np.uint8)
    for lane in gt_small:
        pts = [(int(x), int(y)) for x, y in lane]
        if len(pts) >= 2:
            for i in range(len(pts)-1):
                cv2.line(gt_iou, pts[i], pts[i+1], 255, 2)
        else:
            for (x, y) in pts:
                cv2.circle(gt_iou, (x, y), 2, 255, -1)
    gt_iou_bin = (gt_iou > 0).astype(np.uint8)
    inter = int(np.sum(pred_bin & gt_iou_bin))
    union = int(np.sum(pred_bin | gt_iou_bin))
    iou   = inter / union if union > 0 else 0.0
    return recall, f1, iou

# ==============================================================================
# NEW METRIC 1: LANE CONTINUITY SCORE
# ==============================================================================
def lane_continuity(pred_mask, gt_lanes, orig_h, orig_w, 
                    small_h=PROC_H, small_w=PROC_W):
    """
    Measures how continuous/unbroken the detected lane is.
    Returns: (avg_continuity, avg_gap_length, avg_fragments_per_lane)
    """
    if not gt_lanes:
        return None, None, None
    
    sx, sy = small_w / orig_w, small_h / orig_h
    
    all_continuities = []
    all_gap_lengths = []
    all_fragment_counts = []
    
    for lane in gt_lanes:
        pts = [(int(x * sx), int(y * sy)) for x, y in lane]
        pts.sort(key=lambda p: p[1])
        
        if len(pts) < 2:
            continue
        
        gaps = []
        in_gap = False
        gap_start = 0
        fragments = 1
        
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            num_samples = max(2, int(dist))
            
            hits = 0
            for t in range(num_samples):
                alpha = t / (num_samples - 1) if num_samples > 1 else 0
                x = int(x1 + alpha * (x2 - x1))
                y = int(y1 + alpha * (y2 - y1))
                
                if 0 <= x < small_w and 0 <= y < small_h:
                    if pred_mask[y, x] > 0:
                        hits += 1
            
            hit_ratio = hits / num_samples if num_samples > 0 else 0
            
            if hit_ratio < 0.3:
                if not in_gap:
                    in_gap = True
                    gap_start = i
                    fragments += 1
            else:
                if in_gap:
                    in_gap = False
                    gaps.append(i - gap_start)
        
        if gaps:
            all_gap_lengths.extend(gaps)
        
        total_segments = len(pts) - 1
        if total_segments > 0:
            continuity = 1.0 - (len(gaps) / total_segments)
            all_continuities.append(max(0, continuity))
        else:
            all_continuities.append(1.0)
        
        all_fragment_counts.append(fragments)
    
    return (np.mean(all_continuities) if all_continuities else 0.0,
            np.mean(all_gap_lengths) if all_gap_lengths else 0.0,
            np.mean(all_fragment_counts) if all_fragment_counts else 1.0)

# ==============================================================================
# NEW METRIC 2: FALSE POSITIVE RATE BY CONDITION
# ==============================================================================
def false_positive_analysis(pred_mask, gt_lanes, orig_h, orig_w,
                            small_h=PROC_H, small_w=PROC_W):
    """
    Analyzes false positive distribution.
    Returns: (fp_rate, fp_density_pct, noise_sources_dict)
    """
    if not gt_lanes:
        return None, None, None
    
    sx, sy = small_w / orig_w, small_h / orig_h
    tolerance = max(1, int(POINT_TOLERANCE_PX * sx))
    
    gt_mask = np.zeros((small_h, small_w), dtype=np.uint8)
    
    for lane in gt_lanes:
        pts = [(int(x * sx), int(y * sy)) for x, y in lane]
        for x, y in pts:
            cv2.circle(gt_mask, (x, y), tolerance, 255, -1)
    
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    
    tp = np.sum(pred_bin & gt_bin)
    fp = np.sum(pred_bin & (1 - gt_bin))
    total_pred = tp + fp
    
    fp_rate = fp / total_pred if total_pred > 0 else 0
    fp_density = fp / (small_h * small_w) * 100
    
    fp_mask = pred_bin & (1 - gt_bin)
    
    h_third = small_h // 3
    w_quarter = small_w // 4
    
    noise_sources = {
        'top_third': np.sum(fp_mask[:h_third, :]),
        'mid_third': np.sum(fp_mask[h_third:2*h_third, :]),
        'bottom_third': np.sum(fp_mask[2*h_third:, :]),
        'left_edge': np.sum(fp_mask[:, :w_quarter]),
        'right_edge': np.sum(fp_mask[:, 3*w_quarter:]),
        'center': np.sum(fp_mask[:, w_quarter:3*w_quarter]),
    }
    
    total_fp = fp if fp > 0 else 1
    noise_sources_pct = {k: (v / total_fp) * 100 for k, v in noise_sources.items()}
    
    return fp_rate, fp_density, noise_sources_pct

# ==============================================================================
# ENHANCED METRICS ACCUMULATOR
# ==============================================================================
class Metrics:
    def __init__(self):
        self.total = 0
        self.lane_lost = 0
        self.acc_vals = []
        self.f1_vals = []
        self.iou_vals = []
        self.latencies = []
        self.robustness = {"day": [], "shadow": [], "night": []}
        
        # New metric accumulators
        self.continuity_vals = []
        self.gap_length_vals = []
        self.fragment_vals = []
        self.fp_rate_vals = []
        self.fp_density_vals = []
        self.noise_sources = {"top_third": [], "mid_third": [], "bottom_third": [],
                              "left_edge": [], "right_edge": [], "center": []}
        
        # Condition-specific F1 breakdown
        self.condition_lane_breakdown = defaultdict(lambda: {
            'accuracy': [], 'f1': [], 'iou': [], 'continuity': []
        })
        
    def add(self, acc, f1, iou, condition, latency_ms, lane_lost,
            continuity=None, gap_length=None, fragments=None,
            fp_rate=None, fp_density=None, noise_sources_dict=None,
            lane_type='unknown'):
        self.total += 1
        if lane_lost:
            self.lane_lost += 1
        self.latencies.append(latency_ms)
        
        if acc is not None:
            self.acc_vals.append(acc)
            self.f1_vals.append(f1)
            self.robustness[condition].append(acc)
        
        if iou is not None:
            self.iou_vals.append(iou)
        
        # New metrics
        if continuity is not None:
            self.continuity_vals.append(continuity)
        if gap_length is not None:
            self.gap_length_vals.append(gap_length)
        if fragments is not None:
            self.fragment_vals.append(fragments)
        if fp_rate is not None:
            self.fp_rate_vals.append(fp_rate)
        if fp_density is not None:
            self.fp_density_vals.append(fp_density)
        if noise_sources_dict is not None:
            for key in self.noise_sources:
                self.noise_sources[key].append(noise_sources_dict.get(key, 0))
        
        # Condition + Lane Type breakdown
        key = f"{condition}_{lane_type}"
        self.condition_lane_breakdown[key]['accuracy'].append(acc if acc is not None else 0)
        self.condition_lane_breakdown[key]['f1'].append(f1 if f1 is not None else 0)
        self.condition_lane_breakdown[key]['iou'].append(iou if iou is not None else 0)
        self.condition_lane_breakdown[key]['continuity'].append(continuity if continuity is not None else 0)
    
    def summary(self):
        def mean_pct(vals):
            return sum(vals) / len(vals) * 100 if vals else float("nan")
        def mean_val(vals):
            return sum(vals) / len(vals) if vals else float("nan")
        def fmt(v):
            return f"{v:.1f}" if v == v else "N/A"
        
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 1
        avg_fps = 1000.0 / avg_latency
        lane_lost_rate = self.lane_lost / self.total * 100 if self.total else 0
        W = 60
        
        print("\n" + "=" * W)
        print("  RESULTS — Method 3: Enhanced Adaptive HSV  |  CULane")
        print("=" * W)
        print(f"  Total frames            : {self.total}")
        print(f"  Frames with GT labels   : {len(self.acc_vals)}")
        print(f"  Lane Lost Rate (%)      : {lane_lost_rate:.1f}")
        print(f"  Avg FPS                 : {avg_fps:.1f}")
        print("-" * W)
        
        # Core metrics
        print(f"  CORE METRICS:")
        print(f"    Accuracy (%)          : {fmt(mean_pct(self.acc_vals))}")
        print(f"    F1 Score (%)          : {fmt(mean_pct(self.f1_vals))}")
        print(f"    IoU      (%)          : {fmt(mean_pct(self.iou_vals))}")
        print("-" * W)
        
        # Lane Continuity
        print(f"  LANE CONTINUITY:")
        print(f"    Continuity Score      : {fmt(mean_val(self.continuity_vals) * 100)} %")
        print(f"    Avg Gap Length (pts)  : {fmt(mean_val(self.gap_length_vals))}")
        print(f"    Avg Fragments/Lane    : {fmt(mean_val(self.fragment_vals))}")
        print("-" * W)
        
        # False Positive Analysis
        print(f"  FALSE POSITIVE ANALYSIS:")
        print(f"    FP Rate               : {fmt(mean_val(self.fp_rate_vals) * 100)} %")
        print(f"    FP Density (area %)   : {fmt(mean_val(self.fp_density_vals))} %")
        print(f"    Noise Sources (spatial % of FPs):")
        print(f"      Top Third           : {fmt(mean_val(self.noise_sources['top_third']))} %")
        print(f"      Mid Third           : {fmt(mean_val(self.noise_sources['mid_third']))} %")
        print(f"      Bottom Third        : {fmt(mean_val(self.noise_sources['bottom_third']))} %")
        print(f"      Left Edge           : {fmt(mean_val(self.noise_sources['left_edge']))} %")
        print(f"      Right Edge          : {fmt(mean_val(self.noise_sources['right_edge']))} %")
        print(f"      Center              : {fmt(mean_val(self.noise_sources['center']))} %")
        print("-" * W)
        
        # Robustness by lighting
        print(f"  ROBUSTNESS (Accuracy by lighting):")
        print(f"    Day    ({len(self.robustness['day']):4d} frames)  : {fmt(mean_pct(self.robustness['day']))} %")
        print(f"    Shadow ({len(self.robustness['shadow']):4d} frames)  : {fmt(mean_pct(self.robustness['shadow']))} %")
        print(f"    Night  ({len(self.robustness['night']):4d} frames)  : {fmt(mean_pct(self.robustness['night']))} %")
        print("-" * W)
        
        # Condition-Specific F1 Breakdown
        print(f"  CONDITION + LANE TYPE BREAKDOWN:")
        print(f"    {'Scenario':<25s} {'Frames':>6s} {'Acc %':>8s} {'F1 %':>8s} {'IoU %':>8s} {'Cont %':>8s}")
        print(f"    {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        scenario_order = [
            'day_straight', 'day_curve',
            'shadow_straight', 'shadow_curve', 
            'night_straight', 'night_curve',
            'day_unknown', 'shadow_unknown', 'night_unknown'
        ]
        
        for scenario in scenario_order:
            if scenario in self.condition_lane_breakdown:
                data = self.condition_lane_breakdown[scenario]
                n = len(data['accuracy'])
                if n > 0:
                    print(f"    {scenario:<25s} {n:>6d} {mean_pct(data['accuracy']):>7.1f}% "
                          f"{mean_pct(data['f1']):>7.1f}% {mean_pct(data['iou']):>7.1f}% "
                          f"{mean_val(data['continuity'])*100:>7.1f}%")
        
        print("=" * W)

# ==============================================================================
# MAIN LOOPS
# ==============================================================================
def run_on_images(image_paths, labels):
    labelled_paths = [p for p in image_paths if p.replace("\\", "/").lower() in labels]
    if MAX_IMAGES:
        labelled_paths = labelled_paths[:MAX_IMAGES]
    total = len(labelled_paths)
    print(f"[RUN] {len(image_paths)} total images | {total} will be processed")
    if not labelled_paths:
        print("[ERROR] No GT matches found")
        return

    csv_file = open(CSV_OUTPUT, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "condition", "lane_type",
        "V_mean", "lower_V_used", "upper_S_used",
        "white_pixel_percent", "final_X", "lane_lost",
        "recall", "f1", "iou",
        "continuity", "gap_length", "fragments",
        "fp_rate", "fp_density_pct",
        "noise_top", "noise_mid", "noise_bottom",
        "noise_left", "noise_right", "noise_center",
        "processing_time_ms"
    ])

    metrics = Metrics()
    prev_x = PROC_W / 2.0
    t_start = time.time()

    for idx, img_path in enumerate(labelled_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        orig_h, orig_w = frame.shape[:2]
        gt_lanes = labels[img_path.replace("\\", "/").lower()]
        condition = get_lighting_condition(frame)
        lane_type = estimate_lane_curvature(gt_lanes)

        t0 = time.perf_counter()
        final_x, lane_lost, white_pct, v_mean, lower_v, upper_s, mask = detect(frame, prev_x, condition)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        if not lane_lost:
            prev_x = final_x

        acc, f1, iou = evaluate(mask, gt_lanes, orig_h, orig_w)
        
        # New metrics
        continuity, gap_length, fragments = lane_continuity(mask, gt_lanes, orig_h, orig_w)
        fp_rate, fp_density, noise_sources = false_positive_analysis(mask, gt_lanes, orig_h, orig_w)
        
        metrics.add(acc, f1, iou, condition, latency_ms, lane_lost,
                   continuity=continuity, gap_length=gap_length, fragments=fragments,
                   fp_rate=fp_rate, fp_density=fp_density, noise_sources_dict=noise_sources,
                   lane_type=lane_type)

        csv_writer.writerow([
            os.path.basename(img_path),
            condition,
            lane_type,
            f"{v_mean:.2f}", f"{lower_v:.2f}", f"{upper_s:.2f}",
            f"{white_pct:.2f}", f"{final_x:.2f}", lane_lost,
            f"{acc:.4f}" if acc is not None else "",
            f"{f1:.4f}" if f1 is not None else "",
            f"{iou:.4f}" if iou is not None else "",
            f"{continuity:.4f}" if continuity is not None else "",
            f"{gap_length:.2f}" if gap_length is not None else "",
            f"{fragments:.2f}" if fragments is not None else "",
            f"{fp_rate:.4f}" if fp_rate is not None else "",
            f"{fp_density:.2f}" if fp_density is not None else "",
            f"{noise_sources.get('top_third', 0):.1f}" if noise_sources else "",
            f"{noise_sources.get('mid_third', 0):.1f}" if noise_sources else "",
            f"{noise_sources.get('bottom_third', 0):.1f}" if noise_sources else "",
            f"{noise_sources.get('left_edge', 0):.1f}" if noise_sources else "",
            f"{noise_sources.get('right_edge', 0):.1f}" if noise_sources else "",
            f"{noise_sources.get('center', 0):.1f}" if noise_sources else "",
            f"{latency_ms:.3f}"
        ])

        if (idx+1) % 500 == 0:
            elapsed = time.time() - t_start
            fps = (idx+1) / elapsed
            acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A"
            cont_str = f"{continuity*100:.1f}%" if continuity is not None else "N/A"
            fp_str = f"{fp_rate*100:.1f}%" if fp_rate is not None else "N/A"
            
            print(f"[RUN] {idx+1}/{total}  |  FPS: {fps:.1f}  |  "
                  f"latency: {latency_ms:.2f} ms  |  "
                  f"V_mean: {v_mean:.1f}  white_pct: {white_pct:.2f}%  |  "
                  f"acc: {acc_str}  cont: {cont_str}  FP: {fp_str}  |  "
                  f"{condition}/{lane_type}")

    csv_file.close()
    print(f"[CSV] Saved to {CSV_OUTPUT}")
    metrics.summary()

def run_single_image(path, labels):
    frame = cv2.imread(path)
    if frame is None:
        print("[ERROR] Cannot read:", path)
        return
    orig_h, orig_w = frame.shape[:2]
    gt_lanes = labels.get(path.replace("\\", "/").lower(), [])
    condition = get_lighting_condition(frame)
    lane_type = estimate_lane_curvature(gt_lanes)
    
    prev_x = PROC_W / 2.0
    t0 = time.perf_counter()
    final_x, lane_lost, white_pct, v_mean, lower_v, upper_s, mask = detect(frame, prev_x, condition)
    t1 = time.perf_counter()
    
    print(f"Condition     : {condition}")
    print(f"Lane Type     : {lane_type}")
    print(f"V_mean (proc) : {v_mean:.1f}")
    print(f"lower_V_used  : {lower_v:.1f}")
    print(f"upper_S_used  : {upper_s:.1f}")
    print(f"white_pct     : {white_pct:.2f}%")
    print(f"final_X       : {final_x:.1f}")
    print(f"lane_lost     : {lane_lost}")
    print(f"Latency       : {(t1-t0)*1000:.2f} ms")
    
    acc, f1, iou = evaluate(mask, gt_lanes, orig_h, orig_w)
    continuity, gap_length, fragments = lane_continuity(mask, gt_lanes, orig_h, orig_w)
    fp_rate, fp_density, noise_sources = false_positive_analysis(mask, gt_lanes, orig_h, orig_w)
    
    if acc is not None:
        print(f"Accuracy      : {acc*100:.1f}%")
        print(f"F1            : {f1*100:.1f}%")
        print(f"IoU           : {iou*100:.1f}%")
    
    if continuity is not None:
        print(f"Continuity    : {continuity*100:.1f}%")
        print(f"Fragments     : {fragments:.1f}")
    
    if fp_rate is not None:
        print(f"FP Rate       : {fp_rate*100:.1f}%")
        print(f"FP Density    : {fp_density:.1f}%")
    
    cv2.imwrite("debug_mask.png", mask)
    print("[DEBUG] Saved debug_mask.png")

if __name__ == "__main__":
    print(f"[MAIN] Root: {CULANE_ROOT}  Mode: {RUN_MODE}")
    labels = load_culane_labels(CULANE_ROOT)
    try:
        if RUN_MODE == "images":
            paths = find_culane_images(CULANE_ROOT)
            if not paths:
                print("[ERROR] No images found.")
            else:
                print(f"[MAIN] {len(paths)} images found")
                run_on_images(paths, labels)
        elif RUN_MODE == "single":
            run_single_image(SINGLE_IMAGE_PATH, labels)
        else:
            print(f"[ERROR] Unknown RUN_MODE: {RUN_MODE}")
    except KeyboardInterrupt:
        print("\n[MAIN] Stopped.")
    finally:
        cv2.destroyAllWindows()
        print("[MAIN] Done.")