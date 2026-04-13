import cv2
import numpy as np
import time
import math
import io
from datetime import datetime
from core.state import camera_metrics, camera_health, analytics_state, crowd_history
import core.state
from config.settings import *
from services.alert_engine import _maybe_alert

def _apply_orientation(frame, cam_name):
    """Apply flip and rotation from cam_orientation settings."""
    o = cam_orientation.get(cam_name, {})
    if o.get("flip_h"):
        frame = cv2.flip(frame, 1)
    if o.get("flip_v"):
        frame = cv2.flip(frame, 0)
    r = o.get("rotate", 0)
    if r == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif r == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif r == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _auto_orient(frame, cam_name):
    """Auto-rotate portrait→landscape then apply manual orientation."""
    h, w = frame.shape[:2]
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return _apply_orientation(frame, cam_name)


def _point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def _density_heatmap(frame, centers, radius=40):
    """Overlay a Gaussian density heat-map on the frame."""
    if not centers:
        return frame
    heat = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
    for cx, cy in centers:
        x1 = max(0, cx - radius); x2 = min(FRAME_WIDTH, cx + radius)
        y1 = max(0, cy - radius); y2 = min(FRAME_HEIGHT, cy + radius)
        for gy in range(y1, y2):
            for gx in range(x1, x2):
                d = math.hypot(gx - cx, gy - cy)
                if d < radius:
                    heat[gy, gx] += math.exp(-0.5 * (d / (radius / 2.5)) ** 2)
    heat = np.clip(heat / (heat.max() + 1e-6), 0, 1)
    heat_u8  = (heat * 255).astype(np.uint8)
    heat_col = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    frame    = cv2.addWeighted(frame, 0.65, heat_col, 0.35, 0)
    return frame


def _draw_flow_vectors(frame, prev_pts, next_pts, status):
    """Draw optical-flow vectors for tracked points."""
    if prev_pts is None or next_pts is None:
        return frame
    for i, (p, n) in enumerate(zip(prev_pts, next_pts)):
        if status[i]:
            px, py = map(int, p.ravel())
            nx, ny = map(int, n.ravel())
            cv2.arrowedLine(frame, (px, py), (nx, ny), (0, 255, 255), 1,
                            tipLength=0.4)
    return frame


def _compute_turbulence(flow_vecs):
    """
    Turbulence index: variance of flow directions.
    High variance = chaotic, pushing → stampede precursor.
    Returns value in [0, 1].
    """
    if len(flow_vecs) < 4:
        return 0.0
    angles = [math.atan2(dy, dx) for dx, dy in flow_vecs if dx != 0 or dy != 0]
    if not angles:
        return 0.0
    # circular variance proxy
    sin_m = np.mean(np.sin(angles))
    cos_m = np.mean(np.cos(angles))
    R     = math.sqrt(sin_m**2 + cos_m**2)   # resultant length; 1=uniform, 0=chaos
    return float(np.clip(1.0 - R, 0, 1))


def _compute_risk(density_n, motion_n, flow_n, turb_n, accel_n, is_bottleneck):
    """
    Weighted composite risk score in [0, 1].
    Weights tuned for small-area stampede scenarios (classroom demo).
    """
    if is_bottleneck:
        w = {"density": 0.40, "motion": 0.15, "flow": 0.20,
             "turb": 0.15, "accel": 0.10}
    else:
        w = {"density": 0.35, "motion": 0.20, "flow": 0.20,
             "turb": 0.15, "accel": 0.10}
    score = (density_n * w["density"] + motion_n * w["motion"] +
             flow_n    * w["flow"]    + turb_n  * w["turb"]   +
             accel_n   * w["accel"])
    return float(np.clip(score, 0, 1))


def _risk_level(score):
    if score < 0.25: return "Low"
    if score < 0.50: return "Medium"
    if score < 0.75: return "High"
    return "Critical"


def _risk_color_bgr(level):
    return {"Low": (50, 205, 50), "Medium": (0, 200, 200),
            "High": (0, 140, 255), "Critical": (0, 0, 255)}.get(level, (255, 255, 255))


def generate_frames(camera_name):
    import core.state
    # entry_count is core.state.entry_count

    source = CAMERAS[camera_name]
    cap    = cv2.VideoCapture(source)
    state  = analytics_state[camera_name]
    skip   = 2       # increased from 2 to 5 to prevent Python thread-locking and timeout on Cam2/Cam5

    frame_idx  = 0
    last_t     = time.time()
    cached_frame = None   # last encoded frame (re-sent on skipped frames)

    while True:
        ok, frame = cap.read()
        if not ok:
            # Loop video files; reconnect live sources
            if isinstance(source, str) and any(
                source.lower().endswith(e) for e in (".mp4", ".avi", ".mov", ".mkv")
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    camera_health[camera_name]["status"] = "offline"
                    break
            else:
                camera_health[camera_name]["status"] = "offline"
                camera_health[camera_name]["notes"]  = "No signal"
                # Try reconnect after 2s
                time.sleep(2)
                cap = cv2.VideoCapture(source)
                continue

        frame_idx += 1
        now = time.time()
        state["frame_times"].append(now)
        if len(state["frame_times"]) > 1:
            fps = len(state["frame_times"]) / (state["frame_times"][-1] - state["frame_times"][0] + 1e-9)
            camera_health[camera_name]["fps"] = round(fps, 1)

        camera_health[camera_name]["status"]          = "ok"
        camera_health[camera_name]["last_frame_time"] = now

        if frame_idx % skip != 0 and cached_frame is not None:
            yield cached_frame
            continue

        # ── Orientation ──────────────────────────────────────────────────────
        frame = _auto_orient(frame, camera_name)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── YOLO detection ───────────────────────────────────────────────────
        # Using conf=0.25 (lowered from 0.40) to force the AI to detect partially occluded faces/shoulders in dense crowds
        results    = model(frame, conf=0.25, iou=YOLO_IOU, verbose=False, classes=[0])
        detections = results[0].boxes.data
        centers    = []
        boxes      = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centers.append((cx, cy))
            boxes.append((x1, y1, x2, y2, float(conf)))

        count = len(centers)

        # ── Entry camera ─────────────────────────────────────────────────────
        if camera_name == "EntryCam":
            line_y = int(FRAME_HEIGHT * 0.55)
            cv2.line(frame, (0, line_y), (FRAME_WIDTH, line_y), (0, 255, 255), 2)
            cv2.putText(frame, "ENTRY LINE", (10, line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            prev_c = state["prev_centers"]
            for (px, py), (cx, cy) in zip(prev_c, centers):
                if py < line_y <= cy:
                    core.state.entry_count += 1
            state["prev_centers"] = centers
            # Draw boxes
            for x1, y1, x2, y2, conf in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ENTRIES: {core.state.entry_count}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            cached_frame = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            yield cached_frame
            state["prev_gray"] = gray
            continue

        # ── ROI ──────────────────────────────────────────────────────────────
        roi_cfg = CAMERA_ROIS.get(camera_name, [{"rect": (0.05, 0.05, 0.95, 0.95),
                                                   "role": "general"}])[0]
        r       = roi_cfg["rect"]
        rx1 = int(r[0] * FRAME_WIDTH);  ry1 = int(r[1] * FRAME_HEIGHT)
        rx2 = int(r[2] * FRAME_WIDTH);  ry2 = int(r[3] * FRAME_HEIGHT)
        roi_rect = (rx1, ry1, rx2, ry2)

        in_roi   = [(cx, cy) for (cx, cy) in centers
                    if _point_in_rect(cx, cy, roi_rect)]
        n_roi    = len(in_roi)

        # Apply Architectural Occlusion Multiplier
        occ_mult = OCCLUSION_MULTIPLIERS.get(camera_name, 1.0)
        estimated_n_roi = n_roi * occ_mult

        real_area   = ROI_REAL_AREA_M2.get(camera_name, 30.0)
        density_raw = estimated_n_roi / real_area if real_area > 0 else 0.0

        is_bn       = roi_cfg["role"] == "bottleneck"
        crit_thresh = BOTTLENECK_DENSITY_CRITICAL if is_bn else OPEN_CRITICAL_DENSITY
        density_n   = float(np.clip(density_raw / crit_thresh, 0, 1))

        # ── Optical flow (Lucas-Kanade sparse) ───────────────────────────────
        prev_gray = state["prev_gray"]
        flow_vecs = []
        next_pts  = None
        status    = []
        prev_pts  = state["prev_points"]

        if prev_gray is not None and len(in_roi) > 0:
            pts_arr = np.array([[float(cx), float(cy)] for cx, cy in in_roi],
                               dtype=np.float32).reshape(-1, 1, 2)
            try:
                next_pts, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, pts_arr, None, **LK_PARAMS
                )
                status = st.ravel()
                for i, (p, n) in enumerate(zip(pts_arr, next_pts)):
                    if status[i]:
                        dx = float(n[0, 0] - p[0, 0])
                        dy = float(n[0, 1] - p[0, 1])
                        flow_vecs.append((dx, dy))
            except cv2.error:
                pass

        speeds = [math.hypot(dx, dy) for dx, dy in flow_vecs]
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        motion_n  = float(np.clip(avg_speed / S_REF, 0, 1))

        state["speed_history"].append(avg_speed)

        # Acceleration: rate of speed change
        sh = list(state["speed_history"])
        if len(sh) >= 6:
            accel = abs(np.mean(sh[-3:]) - np.mean(sh[-6:-3]))
            accel_n = float(np.clip(accel / (S_REF * 0.5), 0, 1))
        else:
            accel_n = 0.0

        # Flow conflict: opposing vertical movement
        up   = sum(1 for _, dy in flow_vecs if dy < -1.5)
        down = sum(1 for _, dy in flow_vecs if dy >  1.5)
        conflict = 1.0 if (up >= MIN_FLOW_DIRS and down >= MIN_FLOW_DIRS) else \
                   0.5 if ((up + down) >= MIN_FLOW_DIRS and abs(up - down) <= 1) else 0.0
        flow_n = float(conflict)

        # Turbulence
        turb_n = _compute_turbulence(flow_vecs)

        # ── Density-based override ───────────────────────────────────────────
        if is_bn:
            if density_raw >= BOTTLENECK_DENSITY_CRITICAL:
                raw_score = _compute_risk(density_n, motion_n, flow_n, turb_n, accel_n, True)
                risk_score = float(np.clip(max(0.80, raw_score), 0, 1))
            elif density_raw >= BOTTLENECK_DENSITY_HIGH:
                raw_score  = _compute_risk(density_n, motion_n, flow_n, turb_n, accel_n, True)
                risk_score = float(np.clip(raw_score * 1.3, 0, 1))
            else:
                risk_score = _compute_risk(density_n, motion_n, flow_n, turb_n, accel_n, True)
        else:
            if density_raw >= OPEN_CRITICAL_DENSITY:
                risk_score = 1.0
            elif density_raw >= OPEN_DENSITY_HIGH:
                raw_score  = _compute_risk(density_n, motion_n, flow_n, turb_n, accel_n, False)
                risk_score = float(np.clip(raw_score * 1.2, 0, 1))
            else:
                risk_score = _compute_risk(density_n, motion_n, flow_n, turb_n, accel_n, False)

        level = _risk_level(risk_score)

        # Smooth with recent history (exponential moving average)
        hist = list(state["risk_history"])
        if hist:
            prev_r = hist[-1][1]
            risk_score = 0.7 * risk_score + 0.3 * prev_r

        state["risk_history"].append((now, risk_score))
        level = _risk_level(risk_score)

        crowd_history[camera_name].append((now, count))

        camera_metrics[camera_name] = {
            "risk_score":    round(risk_score, 4),
            "risk_level":    level,
            "density":       round(density_raw, 4),
            "motion":        round(motion_n, 4),
            "flow_conflict": round(flow_n, 4),
            "turbulence":    round(turb_n, 4),
            "acceleration":  round(accel_n, 4),
            "count":         count,
            "fps":           camera_health[camera_name]["fps"],
        }

        # Alert logic
        _maybe_alert(camera_name, level, risk_score,
                     density_raw, motion_n, flow_n, turb_n, accel_n,
                     frame if level == "Critical" else None)

        # ── VISUAL OVERLAYS ──────────────────────────────────────────────────

        # Heat-map (only for High/Critical to save CPU)
        if level in ("High", "Critical"):
            frame = _density_heatmap(frame, in_roi)

        # Flow vectors
        if next_pts is not None and len(status) > 0:
            frame = _draw_flow_vectors(
                frame,
                np.array([[float(x), float(y)] for x, y in in_roi],
                         dtype=np.float32).reshape(-1, 1, 2),
                next_pts, status
            )

        # ROI rectangle
        roi_col = _risk_color_bgr(level)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_col, 2)
        cv2.putText(frame, roi_cfg["name"], (rx1 + 4, ry1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, roi_col, 1)

        # Person bounding boxes
        for x1, y1, x2, y2, conf in boxes:
            col = (0, 255, 0) if level == "Low" else roi_col
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

        # HUD panel (semi-transparent box)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (240, 140), (10, 10, 20), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        lvl_c = roi_col
        fps_  = camera_health[camera_name]["fps"]
        lines = [
            (f"People : {count} (ROI:{n_roi})", (255, 255, 255)),
            (f"Density: {density_raw:.2f} p/m2",  (200, 200, 200)),
            (f"Risk   : {level} ({risk_score*100:.0f}%)", lvl_c),
            (f"Motion : {motion_n*100:.0f}%  Turb:{turb_n*100:.0f}%", (180, 180, 255)),
            (f"Flow   : {'CONFLICT' if flow_n>=0.5 else 'stable'}", (255, 200, 100)),
            (f"FPS    : {fps_:.1f}", (120, 180, 120)),
        ]
        for i, (txt, col) in enumerate(lines):
            cv2.putText(frame, txt, (8, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

        # Critical warning banner
        if level == "Critical":
            t_crit = core.state.critical_start_time.get(camera_name)
            secs   = int(now - t_crit) if t_crit else 0
            banner = f"!! CRITICAL RISK  {secs}s !!"
            bw, bh = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            bx = (FRAME_WIDTH - bw) // 2
            cv2.rectangle(frame, (bx - 6, FRAME_HEIGHT - 42),
                          (bx + bw + 6, FRAME_HEIGHT - 14), (0, 0, 180), -1)
            cv2.putText(frame, banner, (bx, FRAME_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Update state
        state["prev_gray"]    = gray
        state["prev_points"]  = np.array(
            [[float(cx), float(cy)] for cx, cy in in_roi],
            dtype=np.float32
        ).reshape(-1, 1, 2) if in_roi else None

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        cached_frame = (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        yield cached_frame


# ═════════════════════════════════════════════════════════════════════════════
