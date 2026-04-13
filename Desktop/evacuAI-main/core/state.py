import threading
from collections import deque
from config.settings import CAMERAS

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

# Mute toggles
alert_mute_all = False
alert_mute_cam = {cam: False for cam in CAMERAS if cam != "EntryCam"}

# SSE broadcast queue
sse_clients = []
sse_lock    = threading.Lock()


# print("TWILIO SID:", os.getenv("TWILIO_ACCOUNT_SID"))
# print("FROM:", TWILIO_FROM)
# print("TO:", ALERT_TO)

