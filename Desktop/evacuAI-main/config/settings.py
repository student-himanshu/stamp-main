import os
from ultralytics import YOLO
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
model = YOLO("yolov8s.pt")          # swap to yolov8s.pt for higher accuracy
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
    "Cam1":     "http://10.111.105.24:8080/video",  # WiFi phone (IP Webcam)
    "Cam2":     "videos/sample5.mp4",
    "Cam3":     0,          # 0                         # Laptop webcam
    "Cam4":     "videos/sample3.mp4",
    "Cam5":     "videos/sample6.mp4",
    "EntryCam": "videos/sample4.mp4",
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
    "Cam1": 5.0,    # classroom demo: ~1 m² squeeze zone (4 people = CRITICAL)
    "Cam2": 3.0,
    "Cam3": 6.0,
    "Cam4": 40.0,
    "Cam5": 4.0,
    "EntryCam": 100,
}

# ─────────────────────────────────────────────────────────────────────────────
# OCCLUSION COMPENSATION (For severe horizontal camera angles)
# ─────────────────────────────────────────────────────────────────────────────
# If a camera is mounted horizontally (eye-level), 1 person in front blocks 2 people behind them.
# This multiplier artificially inflates the count to estimate the invisible crowd.
OCCLUSION_MULTIPLIERS = {
    "Cam1": 0.5,
    "Cam2": 1.0,
    "Cam3": 1.0,
    "Cam4": 1.0,
    "Cam5": 1.0, # Assume every 1 person detected is hiding 1.5 people behind them
}

# ─────────────────────────────────────────────────────────────────────────────
# ROI DEFINITIONS (normalised 0-1 coords, x1 y1 x2 y2)
# Role: 'bottleneck' | 'general'
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_ROIS = {
    "Cam1": [{"name": "SqueezZone", "role": "bottleneck", "rect": (0.0, 0.0, 1.0, 1.0)}],
    "Cam2": [{"name": "CrowdArea", "role": "general",    "rect": (0.0, 0.0, 1.0, 1.0)}],
    "Cam3": [{"name": "LiveZone",  "role": "bottleneck", "rect": (0.0, 0.0, 1.0, 1.0)}],
    "Cam4": [{"name": "OpenFloor", "role": "general",    "rect": (0.0, 0.0, 1.0, 1.0)}],
    "Cam5": [{"name": "MainArea",  "role": "general",    "rect": (0.0, 0.0, 1.0, 1.0)}],
}

# ─────────────────────────────────────────────────────────────────────────────
# DENSITY THRESHOLDS (people / m²)
# ─────────────────────────────────────────────────────────────────────────────
# Classroom demo mode: for 1m² zone with 4 people —
#   density = 4.0 people/m² → we want CRITICAL at ~3.5 people/m²
BOTTLENECK_DENSITY_WARN      = 1.5
BOTTLENECK_DENSITY_HIGH      = 2.5
BOTTLENECK_DENSITY_CRITICAL  = 4.0  # 3.5  dashboard-editable
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


