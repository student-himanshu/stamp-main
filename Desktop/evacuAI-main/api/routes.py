from flask import Blueprint, render_template, Response, request, redirect, url_for, jsonify, send_file
import time
import cv2
import io
import math
from datetime import datetime
from collections import deque
from core.state import camera_metrics, crowd_history, incident_log, camera_health, analytics_state, sse_clients, sse_lock
import core.state
from config.settings import *
from services.vision_engine import generate_frames, _auto_orient
from dotenv import load_dotenv

load_dotenv()

bp = Blueprint('api', __name__)

# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@bp.route("/", methods=["GET", "POST"])
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

        # Process mutes
        core.state.alert_mute_all = request.form.get("mute_all_alerts") == "true"
        for cam in CAMERAS:
            if cam != "EntryCam":
                core.state.alert_mute_cam[cam] = request.form.get(f"mute_{cam}") == "true"

        return redirect(url_for("api.index"))

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
        alert_mute_all=core.state.alert_mute_all,
        alert_mute_cam=core.state.alert_mute_cam,
    )


@bp.route("/video_feed/<camera_name>")
def video_feed(camera_name):
    return Response(
        generate_frames(camera_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@bp.route("/graph/<camera_name>")
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


@bp.route("/incidents")
def incidents():
    return render_template("incidents.html", incidents=incident_log)


@bp.route("/entry_count")
def entry():
    return jsonify(count=core.state.entry_count)


@bp.route("/metrics")
def metrics():
    return jsonify(camera_metrics)


@bp.route("/camera_health")
def camera_health_api():
    return jsonify(camera_health)


@bp.route("/force_alert/<camera_name>")
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
    return redirect(url_for("api.index"))


@bp.route("/reset_alert/<camera_name>")
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
    return redirect(url_for("api.index"))


@bp.route("/orientation/<camera_name>", methods=["POST"])
def set_orientation(camera_name):
    """AJAX endpoint to update orientation live."""
    if camera_name not in cam_orientation:
        return jsonify(ok=False), 404
    data = request.get_json(force=True)
    if "flip_h"  in data: cam_orientation[camera_name]["flip_h"]  = bool(data["flip_h"])
    if "flip_v"  in data: cam_orientation[camera_name]["flip_v"]  = bool(data["flip_v"])
    if "rotate"  in data: cam_orientation[camera_name]["rotate"]  = int(data["rotate"])
    return jsonify(ok=True, orientation=cam_orientation[camera_name])


@bp.route("/snapshot/<camera_name>")
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


@bp.route("/sse")
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


