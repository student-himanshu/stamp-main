# StampedeGuard Pro - Advanced Crowd Density Analytics

StampedeGuard Pro is a real-time computer vision platform designed to calculate crowd density, detect turbulence, and safely push multi-channel alerts (Telegram, SMS, and Audible Siren) strictly before a dangerous stampede occurs. 

## 🏗️ Architecture Stack
The application has been modularized into a clean MVC API pattern:
- **`app.py`**: A micro-entry point utilizing Flask blueprints.
- **`config/settings.py`**: Central repository for IP/webcam feeds, threshold calibrations, and environment tokens.
- **`core/state.py`**: Thread-safe memory structure capturing live risk metrics and incident logs natively.
- **`services/vision_engine.py`**: The heavy-lifting Computer Vision AI. Runs `ultralytics` YOLOv8 inference combined with Lucas-Kanade optical flow. 
- **`services/alert_engine.py`**: Manages all dispatching protocols utilizing `twilio` and `requests` for the Telegram API.
- **`api/routes.py`**: Exposes the data exclusively natively for the frontend UI dashboard.

## 🚀 Features
- **YOLOv8 & Optical Flow**: Precisely detects localized crowd density alongside physical flow divergence logic to predict panic movements.
- **Multi-Zone Analysis**: Supports variable Regions Of Interest (ROI) with fully customizable occlusion multipliers (per-camera).
- **Control Center Mutes**: Instantly mute specific global/camera warning sounds and text messages securely from the Web Dashboard.

## 🔑 Environment Secrets
Copy `.env.example` to `.env` to deploy your own keys:
```env
TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
TELEGRAM_CHAT_ID="your_telegram_chat_id"

TWILIO_ACCOUNT_SID="your_twilio_sid"
TWILIO_AUTH_TOKEN="your_twilio_token"
TWILIO_FROM_NUMBER="+1234567890"
ALERT_TO_NUMBER="+1098765432"
```

## 📜 License
Licensed under MIT. Please submit pull requests to master!