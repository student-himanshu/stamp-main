"""
StampedeGuard Pro — Advanced Real-Time Stampede Detection System
================================================================
Features:
  • YOLOv8 person detection with confidence-based filtering
  • Optical flow (Lucas-Kanade) for precise velocity vectors per person
  • Turbulence / chaos index from flow divergence
  • Density heat-map overlay using Gaussian kernel estimation
  • Entry/exit line-crossing with direction tracking
  • Adaptive ROI per camera with role-aware thresholds
  • Multi-level alerting: In-app SSE push + Twilio SMS/Voice
  • Incident log with screenshots (base64) embedded
  • Camera health watchdog with auto-reconnect
  • Live FPS counter per camera
  • Acceleration detection (rapid speed change → panic signal)
  • Head-count trend + rolling 30s average for smoothing
  • Camera orientation controls: flip H/V + rotation for WiFi cam
  • Classroom demo mode: 4-person squeeze detection in ~1m² zone
"""

from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_file
import cv2
import numpy as np
import time
import math
import os
import io
import base64
import threading
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from dotenv import load_dotenv
from twilio_service import send_sms_alert

load_dotenv()

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")          # swap to yolov8s.pt for higher accuracy
YOLO_CONF = 0.40                    # detection confidence threshold
YOLO_IOU  = 0.45                    # NMS IOU threshold

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA SOURCES  (editable via dashboard)
# ─────────────────────────────────────────────────────────────────────────────
CAMERAS = {
    "Cam1":     "http://10.71.221.24:8080/video",  # WiFi phone (IP Webcam)
    "Cam2":     "videos/sample5.mp4",
    "Cam3":     0,                                   # Laptop webcam
    "Cam4":     "videos/sample3.mp4",
    "Cam5":     "videos/sample2.mp4",
    "EntryCam": "videos/sample5.mp4",
}

# ─────────────────────────────────────────────────────────────────────────────
# PER-CAMERA ORIENTATION CONTROLS  (set via dashboard or startup)
# ─────────────────────────────────────────────────────────────────────────────
# flip_h: mirror left/right   flip_v: mirror up/down   rotate: 0/90/180/270
cam_orientation = {
    cam: {"flip_h": False, "flip_v": False, "rotate": 0}
    for cam in CAMERAS
}
cam_orientation["Cam1"]["flip_h"] = True   # phone mirror-flip default

# ─────────────────────────────────────────────────────────────────────────────
# REAL-WORLD ROI AREAS (m²)
# ─────────────────────────────────────────────────────────────────────────────
ROI_REAL_AREA_M2 = {
    "Cam1": 2.0,    # classroom demo: ~1 m² squeeze zone (4 people = CRITICAL)
    "Cam2": 4.0,
    "Cam3": 6.0,
    "Cam4": 40.0,
    "Cam5": 30.0,
    "EntryCam": 999,
}

# ─────────────────────────────────────────────────────────────────────────────
# ROI DEFINITIONS (normalised 0-1 coords, x1 y1 x2 y2)
# Role: 'bottleneck' | 'general'
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_ROIS = {
    "Cam1": [{"name": "SqueezZone", "role": "bottleneck", "rect": (0.25, 0.20, 0.75, 0.80)}],
    "Cam2": [{"name": "CrowdArea", "role": "general",    "rect": (0.10, 0.20, 0.90, 0.88)}],
    "Cam3": [{"name": "LiveZone",  "role": "bottleneck", "rect": (0.15, 0.10, 0.85, 0.90)}],
    "Cam4": [{"name": "OpenFloor", "role": "general",    "rect": (0.05, 0.05, 0.95, 0.95)}],
    "Cam5": [{"name": "MainArea",  "role": "general",    "rect": (0.05, 0.05, 0.95, 0.95)}],
}

# ─────────────────────────────────────────────────────────────────────────────
# DENSITY THRESHOLDS (people / m²)
# ─────────────────────────────────────────────────────────────────────────────
# Classroom demo mode: for 1m² zone with 4 people —
#   density = 4.0 people/m² → we want CRITICAL at ~3.5 people/m²
BOTTLENECK_DENSITY_WARN      = 1.5
BOTTLENECK_DENSITY_HIGH      = 2.5
BOTTLENECK_DENSITY_CRITICAL  = 3.5   # dashboard-editable
OPEN_DENSITY_WARN            = 2.0
OPEN_DENSITY_HIGH            = 3.5
OPEN_CRITICAL_DENSITY        = 5.0   # dashboard-editable

# ─────────────────────────────────────────────────────────────────────────────
# MOTION & FLOW PARAMS
# ─────────────────────────────────────────────────────────────────────────────
S_REF          = 12.0   # reference pixel speed per frame (for normalisation)
MIN_FLOW_DIRS  = 2      # min detections needed to compute flow conflict
LK_PARAMS = dict(
    winSize=(15, 15), maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# ─────────────────────────────────────────────────────────────────────────────
# ALERTING
# ─────────────────────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS       = 30
CRITICAL_PERSISTENCE_SECONDS = 15   # seconds Critical must persist before SMS/call


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
camera_metrics = {
    cam: {
        "risk_score":    0.0,
        "risk_level":    "Low",
        "density":       0.0,
        "motion":        0.0,
        "flow_conflict": 0.0,
        "turbulence":    0.0,
        "acceleration":  0.0,
        "count":         0,
        "fps":           0.0,
    }
    for cam in CAMERAS if cam != "EntryCam"
}

crowd_history = {cam: deque(maxlen=1800) for cam in CAMERAS if cam != "EntryCam"}
alerts        = {cam: "" for cam in CAMERAS if cam != "EntryCam"}
alert_sent    = {cam: False for cam in CAMERAS if cam != "EntryCam"}
camera_health = {
    cam: {"status": "ok", "last_frame_time": 0.0, "notes": "", "fps": 0.0}
    for cam in CAMERAS
}

analytics_state = {
    cam: {
        "prev_gray":       None,
        "prev_points":     None,
        "prev_centers":    [],
        "speed_history":   deque(maxlen=30),
        "risk_history":    deque(maxlen=600),
        "frame_times":     deque(maxlen=30),
    }
    for cam in CAMERAS
}

incident_log       = []
entry_count        = 0
last_alert_time    = {cam: 0 for cam in CAMERAS if cam != "EntryCam"}
critical_start_time= {cam: None for cam in CAMERAS if cam != "EntryCam"}

# SSE broadcast queue
sse_clients = []
sse_lock    = threading.Lock()


# print("TWILIO SID:", os.getenv("TWILIO_ACCOUNT_SID"))
# print("FROM:", TWILIO_FROM)
# print("TO:", ALERT_TO)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

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


def _sse_broadcast(data: str):
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.append(data)
            except Exception:
                dead.append(q)
        for d in dead:
            sse_clients.remove(d)



def _log_incident(cam, risk_level, risk_score, density, motion, flow, turb, accel, frame=None):
    entry = {
        "timestamp":  datetime.utcnow().isoformat(timespec="seconds"),

        "camera":     cam,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "density":    density,
        "motion":     motion,
        "flow_conflict": flow,
        "turbulence": turb,
        "acceleration": accel,
        "snapshot":   None,
    }
    if frame is not None:
        _, buf   = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        entry["snapshot"] = base64.b64encode(buf.tobytes()).decode()
    incident_log.append(entry)
    _sse_broadcast(f"incident:{cam}:{risk_level}")


def _maybe_alert(cam, risk_level, risk_score, density, motion, flow, turb, accel, frame=None):
    now = time.time()
    if risk_level == "Medium":
        if critical_start_time[cam] is None:
            critical_start_time[cam] = now
        elapsed = now - critical_start_time[cam]
        if elapsed >= CRITICAL_PERSISTENCE_SECONDS:
            if now - last_alert_time.get(cam, 0) >= ALERT_COOLDOWN_SECONDS:
                last_alert_time[cam] = now
                msg = (f"Density:{density:.2f} p/m²  Motion:{motion:.2f}  "
                       f"Flow:{flow:.2f}  Turb:{turb:.2f}  "
                       f"Sustained:{int(elapsed)}s")
                _log_incident(cam, risk_level, risk_score,
                              density, motion, flow, turb, accel, frame)
                send_sms_alert(cam, risk_level, msg)

    elif risk_level in ("High",):
        # Log High events but no phone call
        if now - last_alert_time.get(cam, 0) >= ALERT_COOLDOWN_SECONDS * 2:
            last_alert_time[cam] = now
            _log_incident(cam, risk_level, risk_score,
                          density, motion, flow, turb, accel, frame)
    else:
        critical_start_time[cam] = None


# ═════════════════════════════════════════════════════════════════════════════
# FRAME GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def generate_frames(camera_name):
    global entry_count

    source = CAMERAS[camera_name]
    cap    = cv2.VideoCapture(source)
    state  = analytics_state[camera_name]
    skip   = 2       # process every Nth frame for perf

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
        results    = model(frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False, classes=[0])
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
                    entry_count += 1
            state["prev_centers"] = centers
            # Draw boxes
            for x1, y1, x2, y2, conf in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ENTRIES: {entry_count}", (10, 35),
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

        real_area   = ROI_REAL_AREA_M2.get(camera_name, 30.0)
        density_raw = n_roi / real_area if real_area > 0 else 0.0

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
            t_crit = critical_start_time.get(camera_name)
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
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET", "POST"])
def index():
    global BOTTLENECK_DENSITY_CRITICAL, OPEN_CRITICAL_DENSITY

    if request.method == "POST":
        # WiFi camera URL
        cam1_ip = request.form.get("cam1_ip", "").strip()
        if cam1_ip:
            if cam1_ip.startswith("http://") or cam1_ip.startswith("https://"):
                CAMERAS["Cam1"] = cam1_ip
            elif ":" in cam1_ip and "/video" in cam1_ip:
                CAMERAS["Cam1"] = f"http://{cam1_ip}"
            else:
                CAMERAS["Cam1"] = f"http://{cam1_ip}:8080/video"

        # Orientation controls for Cam1 (WiFi)
        if "flip_h_Cam1" in request.form:
            cam_orientation["Cam1"]["flip_h"] = request.form.get("flip_h_Cam1") == "true"
        if "flip_v_Cam1" in request.form:
            cam_orientation["Cam1"]["flip_v"] = request.form.get("flip_v_Cam1") == "true"
        rot = request.form.get("rotate_Cam1")
        if rot:
            cam_orientation["Cam1"]["rotate"] = int(rot)

        # Density thresholds
        try:
            v = request.form.get("bottleneck_density_critical")
            if v: BOTTLENECK_DENSITY_CRITICAL = float(v)
        except ValueError: pass
        try:
            v = request.form.get("open_density_critical")
            if v: OPEN_CRITICAL_DENSITY = float(v)
        except ValueError: pass

        # ROI real-area update
        for cam in ROI_REAL_AREA_M2:
            v = request.form.get(f"roi_area_{cam}")
            try:
                if v: ROI_REAL_AREA_M2[cam] = max(0.1, float(v))
            except ValueError: pass

        return redirect(url_for("index"))

    cam1_src = CAMERAS.get("Cam1", "")
    cam1_disp = cam1_src.replace("http://", "").replace("/video", "")

    return render_template(
        "index.html",
        cameras=CAMERAS,
        metrics=camera_metrics,
        health=camera_health,
        cam1_ip=cam1_disp,
        bottleneck_density_critical=BOTTLENECK_DENSITY_CRITICAL,
        open_density_critical=OPEN_CRITICAL_DENSITY,
        roi_areas=ROI_REAL_AREA_M2,
        cam_orientation=cam_orientation,
        incidents=incident_log,
    )


@app.route("/video_feed/<camera_name>")
def video_feed(camera_name):
    return Response(
        generate_frames(camera_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/graph/<camera_name>")
def graph(camera_name):
    if camera_name not in CAMERAS or camera_name == "EntryCam":
        return "Camera not found", 404
    
    data   = list(crowd_history.get(camera_name, []))
    times  = [time.strftime("%H:%M:%S", time.localtime(t)) for t, _ in data]
    counts = [c for _, c in data]

    risk_hist = list(analytics_state.get(camera_name, {}).get("risk_history", []))
    rtimes    = [time.strftime("%H:%M:%S", time.localtime(t)) for t, _ in risk_hist]
    rscores   = [round(s * 100, 1) for _, s in risk_hist]

    return render_template(
        "graph.html", times=times, counts=counts,
        rtimes=rtimes, rscores=rscores, cam=camera_name
    )


@app.route("/incidents")
def incidents():
    return render_template("incidents.html", incidents=incident_log)


@app.route("/entry_count")
def entry():
    return jsonify(count=entry_count)


@app.route("/metrics")
def metrics():
    return jsonify(camera_metrics)


@app.route("/camera_health")
def camera_health_api():
    return jsonify(camera_health)


@app.route("/force_alert/<camera_name>")
def force_alert(camera_name):
    """Force a critical alert for testing/demo purposes."""
    if camera_name not in CAMERAS or camera_name == "EntryCam":
        return "Camera not found", 404
    
    if camera_name in camera_metrics:
        camera_metrics[camera_name].update(
            {"risk_score": 1.0, "risk_level": "Critical"}
        )
        critical_start_time[camera_name] = time.time() - CRITICAL_PERSISTENCE_SECONDS - 1
        _maybe_alert(camera_name, "Critical", 1.0,
                     camera_metrics[camera_name].get("density", 0),
                     camera_metrics[camera_name].get("motion", 0),
                     camera_metrics[camera_name].get("flow_conflict", 0),
                     camera_metrics[camera_name].get("turbulence", 0),
                     camera_metrics[camera_name].get("acceleration", 0))
    return redirect(url_for("index"))


@app.route("/reset_alert/<camera_name>")
def reset_alert(camera_name):
    """Reset alert state for a camera."""
    if camera_name not in CAMERAS or camera_name == "EntryCam":
        return "Camera not found", 404
    
    if camera_name in camera_metrics:
        camera_metrics[camera_name].update({"risk_score": 0.0, "risk_level": "Low"})
    if camera_name in alerts:
        alerts[camera_name] = ""
    if camera_name in critical_start_time:
        critical_start_time[camera_name] = None
    return redirect(url_for("index"))


@app.route("/orientation/<camera_name>", methods=["POST"])
def set_orientation(camera_name):
    """AJAX endpoint to update orientation live."""
    if camera_name not in cam_orientation:
        return jsonify(ok=False), 404
    data = request.get_json(force=True)
    if "flip_h"  in data: cam_orientation[camera_name]["flip_h"]  = bool(data["flip_h"])
    if "flip_v"  in data: cam_orientation[camera_name]["flip_v"]  = bool(data["flip_v"])
    if "rotate"  in data: cam_orientation[camera_name]["rotate"]  = int(data["rotate"])
    return jsonify(ok=True, orientation=cam_orientation[camera_name])


@app.route("/snapshot/<camera_name>")
def snapshot(camera_name):
    """Return latest frame as JPEG download."""
    # Re-run one frame from the source
    src = CAMERAS.get(camera_name)
    if src is None:
        return "Camera not found", 404
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return "Unable to capture frame", 503
    frame = _auto_orient(frame, camera_name)
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    _, buf = cv2.imencode(".jpg", frame)
    img_io = io.BytesIO(buf.tobytes())
    img_io.seek(0)
    filename = f"{camera_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
    return send_file(img_io, mimetype="image/jpeg",
                     as_attachment=True, download_name=filename)


@app.route("/sse")
def sse():
    """Server-Sent Events for real-time push notifications."""
    q = deque(maxlen=20)
    with sse_lock:
        sse_clients.append(q)

    def stream():
        yield "data: connected\n\n"
        while True:
            if q:
                yield f"data: {q.popleft()}\n\n"
            time.sleep(0.3)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(debug=True, threaded=True)