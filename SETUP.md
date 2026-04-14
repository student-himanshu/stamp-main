# Full Setup Guide – Stampede Detection

Step-by-step instructions to run the Stampede Detection app on your machine.

---

## 1. Prerequisites

- **Python 3.8 or newer**  
  Check: `python --version` or `python3 --version`
- **pip** (comes with Python)  
  Check: `pip --version`

---

## 2. Clone or Extract the Project

If you already have the folder:

```text
c:\Users\Saura\Downloads\Stampede_detection-main\Stampede_detection-main
```

Make sure it contains:

- `app.py` – main Flask app  
- `yolov5su.pt` – YOLO model  
- `templates/` – HTML templates  
- `videos/` – sample videos  

---

## 3. Create a Virtual Environment (Recommended)

Open **PowerShell** or **Command Prompt** and go to the project folder:

```powershell
cd "c:\Users\Saura\Downloads\Stampede_detection-main\Stampede_detection-main"
```

Create and activate a virtual environment:

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

You should see `(venv)` in the prompt.

---

## 4. Install Dependencies

With the virtual environment **activated**:

```powershell
pip install -r requirements.txt
```

This installs:

- **Flask** – web server and dashboard  
- **opencv-python** – video and image processing  
- **ultralytics** – YOLO model for person detection  

---

## 5. Verify the Model File

The app expects a YOLO model file named **`yolov5su.pt`** in the project root (same folder as `app.py`).

- If it’s missing, copy it into the project folder or download it from wherever you obtained the project.
- The first run may download extra model data; allow it to finish.

---

## 6. Run the Application

Still in the project folder, with the venv activated:

```powershell
python app.py
```

You should see something like:

```text
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

---

## 7. Open the Dashboard

In your browser go to:

**http://127.0.0.1:5000**

You should see:

- **Crowd Monitoring Dashboard** with camera feeds  
- Cameras: Cam1 (webcam), Cam2–Cam4 and EntryCam (sample videos)  
- Threshold settings and “View Graph” for each camera  
- Entry count for the Entry Camera  

---

## 8. Camera / Video Configuration

In `app.py`, cameras are defined as:

| Camera   | Default source              | Description        |
|----------|-----------------------------|--------------------|
| Cam1     | `0` (default webcam)        | Live camera        |
| Cam2     | `videos/sample4.mp4`        | Sample video       |
| Cam3     | `videos/sample2.mp4`        | Sample video       |
| Cam4     | `videos/sample3.mp4`        | Sample video       |
| EntryCam | `videos/sample2.mp4`        | Entry counting     |

To use **your own webcam**: keep Cam1 as `0` (or use another camera index, e.g. `1`).  
To use **your own videos**: put the file in `videos/` and set the path in `CAMERAS`, e.g. `"Cam2": "videos/your_video.mp4"`.

---

## 9. Troubleshooting

| Issue | What to do |
|-------|------------|
| **`python` not found** | Use `py` or `python3`, or install Python and add it to PATH. |
| **`Activate.ps1` blocked** | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell, then activate again. |
| **No module named 'cv2'** | Activate venv and run `pip install -r requirements.txt` again. |
| **No module named 'ultralytics'** | Same: `pip install -r requirements.txt`. |
| **Cam1 shows nothing** | No webcam or wrong index; in `app.py` change `"Cam1": 0` to `"Cam1": 1` (or use a video path for testing). |
| **Model file missing** | Ensure `yolov5su.pt` is in the same folder as `app.py`. |
| **Port 5000 in use** | In `app.py` change the last line to e.g. `app.run(debug=True, port=5001)` and open `http://127.0.0.1:5001`. |

---

## 10. Quick Reference

```powershell
# One-time setup
cd "c:\Users\Saura\Downloads\Stampede_detection-main\Stampede_detection-main"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Every time you want to run the app
cd "c:\Users\Saura\Downloads\Stampede_detection-main\Stampede_detection-main"
.\venv\Scripts\Activate.ps1
python app.py
# Then open http://127.0.0.1:5000 in the browser
```

---

After this, you have the full setup: venv, dependencies, model, and running app. If a step fails, use the troubleshooting section and the exact error message to narrow it down.
