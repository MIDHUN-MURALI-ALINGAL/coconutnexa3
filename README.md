# CocoNexa 🥥

Coconut maturity detection with a YOLO model and a Streamlit front end.
Stages: Young, Semi-mature, Mature, Over mature.

## Features
- Phone-number login with an SMS code
- Camera or photo upload, then detection with your best.pt model
- Scan history saved per user
- Reminders sent by SMS at the time you choose
- Languages: English, Malayalam, Hindi, Tamil (add more in translations.py)
- Voice output in the chosen language
- Demo mode when best.pt is missing

## Files
- app.py – main app
- db.py – SQLite storage for scans and reminders
- sms.py – SMS sending through Twilio
- translations.py – languages and stage descriptions
- best.pt – your trained model (copy it here)
- requirements.txt, render.yaml

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## SMS setup (Twilio)
Set these environment variables on Render (or in your shell):
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- TWILIO_FROM (your Twilio phone number, e.g. +1...)

Without these, login shows the code on screen for testing, and reminders wait until SMS is set up.
In India, SMS senders must be registered under DLT rules. Twilio can guide you through the registration.

## Deploy on Render
Build: pip install -r requirements.txt
Start: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true

Data is stored in coconexa.db. On Render's free plan this file is erased on restart.
For permanent history, attach a persistent disk or move the data to Render Postgres.
The reminder checker runs inside the web service, so the service must stay awake for reminders to send.
