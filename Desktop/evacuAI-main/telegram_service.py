import os
import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(camera, level, message_text, image_path=None):
    if not BOT_TOKEN or not CHAT_ID:
        print("[Telegram Disabled] Missing credentials")
        return

    from datetime import datetime
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    detailed_text = (
        f"🚨 STAMPEDE ALERT 🚨\n"
        f"📷 Camera: {camera}\n"
        f"⚠️ Level: {level}\n"
        f"⏰ Time: {time_str}\n\n"
        f"📊 Details:\n{message_text}"
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto" if image_path else f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        if image_path:
            with open(image_path, "rb") as img:
                res = requests.post(url, data={
                    "chat_id": CHAT_ID,
                    "caption": detailed_text
                }, files={"photo": img})
        else:
            res = requests.post(url, json={
                "chat_id": CHAT_ID,
                "text": detailed_text
            })
        
        if res.status_code == 200:
            print("[SUCCESS] Telegram Sent Successfully!")
        else:
            print(f"[ERROR] Telegram Failed! Status: {res.status_code}, Reason: {res.text}")
            
    except Exception as e:
        print("Telegram Error/Exception:", e)

    # ------------------ VOICE ALERT SYSTEM ------------------
    try:
        from gtts import gTTS
        # Speak the exact camera name in the alert so you can hear it!
        voice_text = f"Critical Alert! Stampede risk detected on {camera}! Please review immediately."
        tts = gTTS(text=voice_text, lang='en', slow=False)
        voice_path = f"voice_alert_{camera}.mp3"
        tts.save(voice_path)
        
        url_voice = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVoice"
        with open(voice_path, "rb") as voice_file:
            res_voice = requests.post(url_voice, data={"chat_id": CHAT_ID}, files={"voice": voice_file})
            
        if os.path.exists(voice_path):
            os.remove(voice_path)
            
        if res_voice.status_code == 200:
            print("[SUCCESS] Telegram Voice Note Sent Successfully!")
    except ImportError:
        print("[WARNING] gTTS not installed. Run 'pip install gTTS' to hear voice alerts.")
    except Exception as e:
        print("Voice Audio Error:", e)
