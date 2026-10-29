import os


def sms_configured():
    return all(os.getenv(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"))


def send_sms(phone, text):
    """Send an SMS through Twilio. Returns (ok, info)."""
    if not sms_configured():
        return False, "SMS not configured"
    try:
        from twilio.rest import Client
        client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        client.messages.create(body=text, from_=os.getenv("TWILIO_FROM"), to=phone)
        return True, "sent"
    except Exception as exc:
        return False, str(exc)
