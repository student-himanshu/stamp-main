# Full Setup Guide – StampedeGuard Pro

Follow these absolute steps perfectly to configure your local Windows machine to host and run the highly scalable backend instance of StampedeGuard Pro.

---

## 1. Prerequisites
- **Python 3.11** (Recommended, works broadly well on 3.8+)
- **NVIDIA GPU** (Optional but extremely highly recommended to run YOLOv8 natively without thread starvation). CPU inference works natively out of the box but drops FPS drastically.

---

## 2. Directory Map Check
If you downloaded all the repository files successfully, your `c:\Users\Saura\OneDrive\Desktop\stamp-main-main` directory layout must match this perfectly:
- `api/` – Contains `routes.py`
- `config/` – Contains `settings.py` (your camera source variables)
- `core/` – Contains `state.py` (thread-safe global dictionaries)
- `services/` – Contains `vision_engine.py` & `alert_engine.py`
- `templates/` – Contains your UI Dashboard HTML layouts.
- `app.py` – Highly lightweight 14-line script to spawn the server.
- `yolov8s.pt` – Neural net weights.

---

## 3. Activate Virtual Environment
Open **PowerShell** as an administrator inside the project repository root and execute:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
(If it throws an execution error, first run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then try activating `Activate.ps1` again).

---

## 4. Install Core Packages
Run our massive dependency library:
```powershell
pip install -r requirements.txt
```
*(This grabs `flask`, `opencv-python`, `ultralytics`, `twilio`, and `python-dotenv`).*

---

## 5. Build Environment Tokens
You MUST execute the application strictly alongside a `.env` file!
Create `.env` at the root of the folder and configure your Twilio/Telegram bot:
```env
TELEGRAM_BOT_TOKEN="secret_token"
TELEGRAM_CHAT_ID="123456789"
```
*(If you want Twilio SMS texts securely delivered natively, you MUST append `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, and `ALERT_TO_NUMBER` into the `.env` here too).*

---

## 6. Video Configurations
Open `config/settings.py`. Observe the `CAMERAS` dictionary. 
- You can route real physical webcams using integer values `0`, `1`, etc.
- You can link your phone natively using an `IP Webcam` app (like `http://192.168.1.100:8080/video`).
- You can loop local videos via `"videos/sample_video.mp4"`.

---

## 7. Run The Platform 🚀
Spin the central logic threads gracefully:
```powershell
python app.py
```
Open a Google Chrome / Edge Chromium browser on your network and strictly type **`http://127.0.0.1:5000`** into your URL bar!
