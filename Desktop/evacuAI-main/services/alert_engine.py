import time
import os
import cv2
import threading
from datetime import datetime
from core.state import incident_log, sse_clients, sse_lock, critical_start_time, last_alert_time
from config.settings import ALERT_COOLDOWN_SECONDS, CRITICAL_PERSISTENCE_SECONDS
from twilio_service import send_sms_alert
from telegram_service import send_telegram_alert

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
        import base64
        _, buf   = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        entry["snapshot"] = base64.b64encode(buf.tobytes()).decode()
    incident_log.append(entry)
    _sse_broadcast(f"incident:{cam}:{risk_level}")

def _maybe_alert(cam, risk_level, risk_score, density, motion, flow, turb, accel, frame=None):
    now = time.time()
    if risk_level == "Critical":
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
                              
                import core.state
                if not core.state.alert_mute_all and not core.state.alert_mute_cam.get(cam, False):
                    def blast_alarm():
                        try:
                            import winsound
                            import time
                            for _ in range(3):
                                winsound.Beep(2500, 500)
                                time.sleep(0.1)
                        except:
                            pass
                    threading.Thread(target=blast_alarm, daemon=True).start()
                                  
                    img_path = None
                    if frame is not None:
                        img_path = f"alert_{cam}.jpg"
                        cv2.imwrite(img_path, frame)
                    send_telegram_alert(cam, risk_level, msg, img_path)
                    if img_path and os.path.exists(img_path):
                        os.remove(img_path)
                        
                    send_sms_alert(cam, risk_level, msg)
                else:
                    print(f"[{cam}] Critical alert recorded but muted.")
    elif risk_level in ("High",):
        if now - last_alert_time.get(cam, 0) >= ALERT_COOLDOWN_SECONDS * 2:
            last_alert_time[cam] = now
            _log_incident(cam, risk_level, risk_score,
                          density, motion, flow, turb, accel, frame)
    else:
        critical_start_time[cam] = None
