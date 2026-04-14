# twilio_service.py

import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TO_NUMBER   = os.getenv("ALERT_TO_NUMBER")

client = None

if ACCOUNT_SID and AUTH_TOKEN:
    client = Client(ACCOUNT_SID, AUTH_TOKEN)


def send_sms_alert(camera, level, message_text):
    """
    Send SMS alert via Twilio
    """
    if not client or not FROM_NUMBER or not TO_NUMBER:
        print("[Twilio Disabled] Missing credentials")
        return

    full_message = f"[STAMPEDE ALERT]\nCamera: {camera}\nLevel: {level}\n{message_text}"

    try:
        message = client.messages.create(
            body=full_message,
            from_=FROM_NUMBER,
            to=TO_NUMBER
        )
        print("✅ SMS Sent. SID:", message.sid)
    except Exception as e:
        print("❌ Twilio Error:", e)
