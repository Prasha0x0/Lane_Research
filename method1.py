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
MAX_IMAGES        = None   # Set to e.g. 500 for quick test

CSV_OUTPUT = "hough_results.csv"
PROC_W, PROC_H = 160, 120

# Hough parameters (tuned for better detection)
HOUGH_THRESHOLD = 30
MIN_LINE_LENGTH = 10
MAX_LINE_GAP = 30

# Metrics
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
    """Trapezoid ROI (same as CULane pipeline)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    top_y     = int(h * 0.45)
    bot_y     = h
    top_left  = int(w * 0.20)
    top_right = int(w * 0.80)
    pts = np.array([[0, bot_y], [w, bot_y], [top_right, top_y], [top_left, top_y]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

_ROI_MASK = build_roi_mask(PROC_H, PROC_W)

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
# HOUGH DETECTION (Improved with CLAHE)
# ==============================================================================
def detect_hough(frame):
    """
    Process frame with Hough lines, return binary mask.
    Uses CLAHE for contrast enhancement and tuned parameters.
    """
    # Resize + preprocessing
    small = cv2.resize(frame, (PROC_W, PROC_H))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for contrast enhancement (helps night detection)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)  # Lower thresholds for night

    # Apply ROI
    masked_edges = cv2.bitwise_and(edges, _ROI_MASK)

    # Hough lines (tuned for better detection)
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    # Create binary mask from Hough lines
    hough_mask = np.zeros((PROC_H, PROC_W), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)   # thickness 2

    # Apply ROI
    hough_mask = cv2.bitwise_and(hough_mask, _ROI_MASK)

    return hough_mask

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================
def evaluate(pred_mask, gt_lanes, orig_h, orig_w, small_h=PROC_H, small_w=PROC_W):
    if not gt_lanes:
        return None, None, None

    # Scale GT to small mask size
    sx, sy = small_w / orig_w, small_h / orig_h
    tolerance = max(1, int(POINT_TOLERANCE_PX * sx))
    gt_small = [[(x * sx, y * sy) for x, y in lane] for lane in gt_lanes]

    # --- Accuracy (recall) ---
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

    # --- Precision ---
    gt_canvas = np.zeros((small_h, small_w), dtype=np.uint8)
    for lane in gt_small:
        for (x, y) in lane:
            cv2.circle(gt_canvas, (int(x), int(y)), tolerance, 255, -1)
    pred_bin  = (pred_mask  > 0).astype(np.uint8)
    gt_bin    = (gt_canvas  > 0).astype(np.uint8)
    tp_px     = int(np.sum(pred_bin & gt_bin))
    fp_px     = int(np.sum(pred_bin & (1 - gt_bin)))
    precision = tp_px / (tp_px + fp_px) if (tp_px + fp_px) > 0 else 0.0

    # --- F1 ---
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # --- IoU ---
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
    - continuity: 0-1 where 1 = perfectly unbroken lane
    - gap_length: average number of GT points between breaks
    - fragments: number of disconnected prediction segments per GT lane
    """
    if not gt_lanes:
        return None, None, None
    
    sx, sy = small_w / orig_w, small_h / orig_h
    
    all_continuities = []
    all_gap_lengths = []
    all_fragment_counts = []
    
    for lane in gt_lanes:
        # Scale and sort by y-coordinate
        pts = [(int(x * sx), int(y * sy)) for x, y in lane]
        pts.sort(key=lambda p: p[1])
        
        if len(pts) < 2:
            continue
        
        # Trace along the lane and count breaks
        gaps = []
        in_gap = False
        gap_start = 0
        fragments = 1
        
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            
            # Sample points along the segment between GT points
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
            
            if hit_ratio < 0.3:  # Less than 30% of points hit = gap
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
    - fp_rate: false positives / total predictions (0-1)
    - fp_density_pct: false positive pixels as % of frame area
    - noise_sources: % of FPs from different spatial regions
    """
    if not gt_lanes:
        return None, None, None
    
    sx, sy = small_w / orig_w, small_h / orig_h
    tolerance = max(1, int(POINT_TOLERANCE_PX * sx))
    
    # Create ground truth mask
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
    fp_density = fp / (small_h * small_w) * 100  # percentage of frame
    
    # Analyze where FPs come from (spatial regions)
    fp_mask = pred_bin & (1 - gt_bin)
    
    # Define regions
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
# NEW METRIC 3: CONDITION-SPECIFIC F1 BREAKDOWN
# ==============================================================================
def condition_specific_f1_breakdown(acc, f1, iou, continuity, condition, lane_type):
    """
    Returns structured dict for condition+lane_type stratification.
    """
    return {
        'condition': condition,
        'lane_type': lane_type,
        'accuracy': acc if acc is not None else 0.0,
        'f1': f1 if f1 is not None else 0.0,
        'iou': iou if iou is not None else 0.0,
        'continuity': continuity if continuity is not None else 0.0,
    }

# ==============================================================================
# ENHANCED METRICS ACCUMULATOR
# ==============================================================================
class Metrics:
    def __init__(self):
        self.total = 0
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
        
        # Condition-specific F1 breakdown (condition + lane_type)
        self.condition_lane_breakdown = defaultdict(lambda: {
            'accuracy': [], 'f1': [], 'iou': [], 'continuity': []
        })
        
    def add(self, acc, f1, iou, condition, latency_ms, 
            continuity=None, gap_length=None, fragments=None,
            fp_rate=None, fp_density=None, noise_sources_dict=None,
            lane_type='unknown'):
        self.total += 1
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
        W = 60
        
        print("\n" + "=" * W)
        print("  RESULTS — Hough Line Detection  |  CULane")
        print("=" * W)
        print(f"  Total frames            : {self.total}")
        print(f"  Frames with GT labels   : {len(self.acc_vals)}")
        print(f"  Avg FPS                 : {avg_fps:.1f}")
        print("-" * W)
        
        # Core metrics
        print(f"  CORE METRICS:")
        print(f"    Accuracy (%)          : {fmt(mean_pct(self.acc_vals))}")
        print(f"    F1 Score (%)          : {fmt(mean_pct(self.f1_vals))}")
        print(f"    IoU      (%)          : {fmt(mean_pct(self.iou_vals))}")
        print("-" * W)
        
        # New: Lane Continuity
        print(f"  LANE CONTINUITY:")
        print(f"    Continuity Score      : {fmt(mean_val(self.continuity_vals) * 100)} %")
        print(f"    Avg Gap Length (pts)  : {fmt(mean_val(self.gap_length_vals))}")
        print(f"    Avg Fragments/Lane    : {fmt(mean_val(self.fragment_vals))}")
        print("-" * W)
        
        # New: False Positive Analysis
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
        
        # New: Condition-Specific F1 Breakdown
        print(f"  CONDITION + LANE TYPE BREAKDOWN:")
        print(f"    {'Scenario':<25s} {'Frames':>6s} {'Acc %':>8s} {'F1 %':>8s} {'IoU %':>8s} {'Cont %':>8s}")
        print(f"    {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        # Display in logical order
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
# MAIN LOOP
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
        "recall", "f1", "iou",
        "continuity", "gap_length", "fragments",
        "fp_rate", "fp_density_pct",
        "noise_top", "noise_mid", "noise_bottom",
        "noise_left", "noise_right", "noise_center",
        "time_ms"
    ])

    metrics = Metrics()
    t_start = time.time()

    for idx, img_path in enumerate(labelled_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        orig_h, orig_w = frame.shape[:2]
        gt_lanes = labels[img_path.replace("\\", "/").lower()]
        condition = get_lighting_condition(frame)
        lane_type = estimate_lane_curvature(gt_lanes)

        # ---- Detect with Hough ----
        t0 = time.perf_counter()
        mask = detect_hough(frame)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # ---- Evaluate ----
        acc, f1, iou = evaluate(mask, gt_lanes, orig_h, orig_w)
        
        # ---- New Metrics ----
        continuity, gap_length, fragments = lane_continuity(mask, gt_lanes, orig_h, orig_w)
        fp_rate, fp_density, noise_sources = false_positive_analysis(mask, gt_lanes, orig_h, orig_w)
        
        # ---- Accumulate ----
        metrics.add(acc, f1, iou, condition, latency_ms,
                   continuity=continuity, gap_length=gap_length, fragments=fragments,
                   fp_rate=fp_rate, fp_density=fp_density, noise_sources_dict=noise_sources,
                   lane_type=lane_type)

        # ---- CSV Row ----
        csv_writer.writerow([
            os.path.basename(img_path),
            condition,
            lane_type,
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
                  f"acc: {acc_str}  cont: {cont_str}  "
                  f"FP: {fp_str}  |  {condition}/{lane_type}")

    csv_file.close()
    print(f"[CSV] Saved to {CSV_OUTPUT}")
    metrics.summary()

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print(f"[MAIN] Root: {CULANE_ROOT}")
    labels = load_culane_labels(CULANE_ROOT)
    paths = find_culane_images(CULANE_ROOT)
    if not paths:
        print("[ERROR] No images found.")
    else:
        print(f"[MAIN] {len(paths)} images found")
        run_on_images(paths, labels)
    print("[MAIN] Done.")