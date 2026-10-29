import hashlib
import json
import os
import secrets
import threading
import time
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from ultralytics import YOLO

import db
from sms import send_sms
from translations import LANGS, T, CLASS_TEXT, SPEECH_LANG

st.set_page_config(page_title="CocoNexa", page_icon="🥥", layout="wide")
db.init()

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
TZ = ZoneInfo(os.getenv("TZ_NAME", "Asia/Kolkata"))

CLASS_NAMES = ["Young", "Semi-mature", "Mature", "Over mature"]
CLASS_EMOJI = {"Young": "🟢", "Semi-mature": "🟡", "Mature": "🟤", "Over mature": "⚫"}
CLASS_COLOR = {"Young": "#2e7d32", "Semi-mature": "#9e9d24", "Mature": "#8d6e63", "Over mature": "#4e342e"}


def tr(key, **kw):
    lang = st.session_state.get("lang", "en")
    text = T.get(lang, {}).get(key) or T["en"][key]
    return text.format(**kw) if kw else text


def cls_text(label):
    lang = st.session_state.get("lang", "en")
    return CLASS_TEXT.get(lang, {}).get(label) or CLASS_TEXT["en"][label]


def normalize(raw):
    """Map a label from best.pt to one of the four classes, or None if unknown."""
    compact = str(raw).lower().replace("_", "").replace("-", "").replace(" ", "")
    for name in CLASS_NAMES:
        if compact == name.lower().replace("-", "").replace(" ", ""):
            return name
    return None


def speak(text, auto=False):
    """Read text aloud in the chosen language. Automatic speech respects the Voice switch."""
    if auto and not st.session_state.get("voice_on", True):
        return
    lang = SPEECH_LANG.get(st.session_state.get("lang", "en"), "en-US")
    components.html(
        f"""
        <script>
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance({json.dumps(text)});
          u.lang = {json.dumps(lang)};
          u.rate = 0.95;
          window.speechSynthesis.speak(u);
        </script>
        """,
        height=0,
    )


def save_lang():
    if "user" in st.session_state:
        db.set_pref(st.session_state.user, lang=st.session_state.lang)


def save_voice():
    if "user" in st.session_state:
        db.set_pref(st.session_state.user, voice=st.session_state.voice_on)


@st.cache_resource(show_spinner="Loading CocoNexa model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None  # demo mode
    return YOLO(MODEL_PATH)


def demo_analyze(image):
    """Demo mode without best.pt: estimates the stage from the coconut's colour.
    It is a rough guide only, not the trained model."""
    hsv = np.array(image.convert("HSV")).astype(np.float32)
    hue = hsv[..., 0] * 360 / 255
    sat = hsv[..., 1]
    val = hsv[..., 2]
    mask = (sat > 40) & (val > 40)  # coconut-coloured pixels, ignores plain background
    annotated = image.copy()
    if mask.mean() < 0.02:
        return [], annotated

    h_med = float(np.median(hue[mask]))
    v_med = float(np.median(val[mask]))
    if 75 <= h_med < 160:
        label = "Young"
    elif 50 <= h_med < 75:
        label = "Semi-mature"
    elif h_med < 50 or h_med >= 330:
        label = "Mature" if v_med >= 100 else "Over mature"
    else:
        label = "Over mature"

    ys, xs = np.where(mask)
    x1, x2 = np.percentile(xs, [5, 95])
    y1, y2 = np.percentile(ys, [5, 95])
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=4)
    draw.text((x1 + 6, max(0, y1 - 14)), f"{label} (demo)", fill=(255, 255, 0))

    conf = min(0.9, 0.5 + float(mask.mean()))
    return [(label, conf, f"colour hue {h_med:.0f}, brightness {v_med:.0f}")], annotated


def run_detection(model, image, conf):
    """Returns a list of (class or None, confidence, raw label) and an annotated image."""
    if model is None:
        return demo_analyze(image)
    bgr = np.array(image)[:, :, ::-1]  # Ultralytics expects BGR
    result = model.predict(bgr, conf=conf, verbose=False)[0]
    found = []
    for box in result.boxes:
        raw = result.names[int(box.cls[0])]
        found.append((normalize(raw), float(box.conf[0]), raw))
    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    return found, annotated


def finish_login():
    """Runs when the code is submitted, before the page reloads."""
    pending = st.session_state.get("pending")
    typed = st.session_state.get("code_input", "").strip()
    if pending and time.time() < pending["expires"] and typed == pending["code"]:
        pref = db.get_pref(pending["phone"])
        st.session_state.user = pending["phone"]
        st.session_state.lang = pref["lang"]
        st.session_state.voice_on = bool(pref["voice"])
        st.session_state.pop("pending", None)
    else:
        st.session_state.login_error = True


def login_screen():
    st.subheader(tr("login_title"))
    pending = st.session_state.get("pending")
    if pending is None:
        with st.form("phone_form"):
            phone = st.text_input(tr("phone_label"), placeholder="+91XXXXXXXXXX")
            send = st.form_submit_button(tr("send_code"))
        if send:
            phone = phone.strip().replace(" ", "")
            if not (phone.startswith("+") and phone[1:].isdigit() and len(phone) >= 10):
                st.error(tr("bad_phone"))
                return
            code = f"{secrets.randbelow(10**6):06d}"
            ok, _ = send_sms(phone, f"CocoNexa login code: {code}")
            st.session_state.pending = {"phone": phone, "code": code,
                                        "expires": time.time() + 300, "sent": ok}
            st.rerun()
        return

    if pending["sent"]:
        st.info(tr("sms_sent"))
    else:
        st.info(tr("dev_code", code=pending["code"]))
    with st.form("code_form"):
        st.text_input(tr("code_label"), key="code_input")
        st.form_submit_button(tr("verify"), on_click=finish_login)
    if st.session_state.pop("login_error", False):
        st.error(tr("wrong_code"))
    if st.button(tr("back")):
        st.session_state.pop("pending", None)
        st.rerun()


@st.cache_resource
def start_reminder_worker():
    """Checks every minute for due reminders and sends them by SMS."""
    def loop():
        while True:
            try:
                for r in db.due_reminders(time.time()):
                    ok, _ = send_sms(r["phone"], r["message"])
                    if ok:
                        db.mark_sent(r["id"])
            except Exception as exc:
                print("reminder worker error:", exc)
            time.sleep(60)
    threading.Thread(target=loop, daemon=True).start()
    return True


# ---- Page setup -------------------------------------------------------
st.sidebar.selectbox("🌐", options=list(LANGS), format_func=lambda k: LANGS[k],
                     key="lang", on_change=save_lang)

if "user" not in st.session_state:
    st.title("🥥 CocoNexa")
    login_screen()
    st.stop()

phone = st.session_state.user
start_reminder_worker()
model = load_model()
st.session_state.setdefault("voice_on", True)

st.sidebar.caption(f"📱 {phone}")
st.sidebar.toggle(tr("voice_on"), key="voice_on", on_change=save_voice)
if st.sidebar.button(tr("logout")):
    for key in ("user", "last_sig", "latest"):
        st.session_state.pop(key, None)
    st.rerun()
conf = st.sidebar.slider(tr("threshold"), 0.10, 0.90, 0.40, 0.05)
sidebar = st.sidebar.container()

st.title("🥥 CocoNexa")
if model is None:
    st.warning(tr("demo_warning"))

tab_scan, tab_hist, tab_rem = st.tabs([tr("tab_scan"), tr("tab_hist"), tr("tab_rem")])

# ---- Scan tab ---------------------------------------------------------
with tab_scan:
    total_box = st.container()
    cam_tab, up_tab = st.tabs([tr("take_photo"), tr("upload_photo")])
    with cam_tab:
        camera_photo = st.camera_input(tr("take_photo"))
    with up_tab:
        uploaded = st.file_uploader(tr("upload_photo"), type=["jpg", "jpeg", "png"])

    source = uploaded or camera_photo
    if source is None:
        st.info(tr("start_hint"))
    else:
        sig = hashlib.md5(source.getvalue()).hexdigest()
        image = Image.open(source).convert("RGB")
        with st.spinner(tr("detecting")):
            found, annotated = run_detection(model, image, conf)
        valid = [d for d in found if d[0] is not None]

        # Count and speak only once per new photo, not on every rerun
        if sig != st.session_state.get("last_sig"):
            st.session_state.last_sig = sig
            if valid:
                for label, p, _ in valid:
                    db.add_scan(phone, label, p)
                best = max(valid, key=lambda d: d[1])
                st.session_state.latest = best[0]
                info = cls_text(best[0])
                spoken = (f"{tr('photo_count')}: {len(valid)}. "
                          f"{tr('result_label')}: {info['name']}. {info['desc']} {info['use']}. "
                          f"{tr('count_spoken', n=db.total(phone))}")
                speak(spoken, auto=True)
            else:
                st.session_state.latest = None
                speak(tr("no_detect"), auto=True)

        latest = st.session_state.get("latest")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.image(annotated, use_container_width=True)
        with c2:
            st.metric(tr("photo_count"), len(valid))
            if latest is None:
                st.warning(tr("no_detect"))
            else:
                info = cls_text(latest)
                st.success(f"{CLASS_EMOJI[latest]} **{info['name']}**")
                st.write(info["desc"])
                st.write(f"**{tr('use_label')}:** {info['use']}")
            if st.button(tr("read_aloud")) and latest:
                info = cls_text(latest)
                speak(f"{info['name']}. {info['desc']} {info['use']}")

        with st.expander(tr("debug_title")):
            names = list(model.names.values()) if model is not None else "demo"
            st.write(f"{tr('model_classes')}: {names}")
            st.dataframe(
                pd.DataFrame([{"raw label": r, "class": c or "-", "confidence": round(p, 3)}
                              for c, p, r in found]),
                hide_index=True,
            )

    st.divider()
    if st.button(tr("explain_all")):
        speak(" ".join(f"{cls_text(n)['name']}. {cls_text(n)['desc']}" for n in CLASS_NAMES))

# Total count shown at the top of the scan tab
with total_box:
    st.metric(tr("total_count"), db.total(phone))

# ---- History tab ------------------------------------------------------
with tab_hist:
    st.subheader(tr("history_title"))
    rows = db.history(phone)
    if not rows:
        st.info(tr("history_empty"))
    else:
        st.dataframe(
            pd.DataFrame([{
                tr("col_time"): dt.datetime.fromtimestamp(r["created_at"], TZ).strftime("%d %b %Y %H:%M"),
                tr("col_stage"): cls_text(r["label"])["name"],
                tr("col_conf"): f"{r['confidence'] * 100:.1f}%",
            } for r in rows]),
            use_container_width=True,
            hide_index=True,
        )

# ---- Reminders tab ----------------------------------------------------
with tab_rem:
    st.subheader(tr("rem_title"))
    with st.form("rem_form"):
        msg = st.text_input(tr("rem_message"), value=tr("rem_default"))
        day = st.date_input(tr("rem_date"), value=dt.datetime.now(TZ).date())
        at = st.time_input(tr("rem_time"), value=dt.time(9, 0))
        save = st.form_submit_button(tr("rem_save"))
    if save and msg.strip():
        when = dt.datetime.combine(day, at, tzinfo=TZ).timestamp()
        db.add_reminder(phone, msg.strip(), when)
        st.success(tr("rem_saved"))
    st.caption(tr("rem_sms_note"))

    st.markdown(f"**{tr('rem_list')}**")
    rems = db.reminders(phone)
    if not rems:
        st.info(tr("rem_empty"))
    else:
        st.dataframe(
            pd.DataFrame([{
                tr("rem_date"): dt.datetime.fromtimestamp(r["remind_at"], TZ).strftime("%d %b %Y %H:%M"),
                tr("rem_message"): r["message"],
                tr("rem_pending"): "" if r["sent"] else "⏳",
                tr("rem_sent"): "✅" if r["sent"] else "",
            } for r in rems]),
            use_container_width=True,
            hide_index=True,
        )

# ---- Sidebar dashboard (filled last so it shows the newest scan) -----
with sidebar:
    st.header(tr("dashboard"))
    st.metric(tr("total_count"), db.total(phone))
    counts = {c: 0 for c in CLASS_NAMES}
    counts.update(db.counts(phone))
    latest = st.session_state.get("latest")
    for name in CLASS_NAMES:
        info = cls_text(name)
        border = f"3px solid {CLASS_COLOR[name]}" if name == latest else "1px solid #ddd"
        st.markdown(
            f"""
            <div style="border:{border};border-radius:10px;padding:8px 10px;margin:6px 0">
              <b>{CLASS_EMOJI[name]} {info['name']}</b>
              <span style="float:right">{counts[name]}</span><br>
              <small>{info['desc']}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.bar_chart(pd.DataFrame(
        {tr("counted"): [counts[c] for c in CLASS_NAMES]},
        index=[cls_text(c)["name"] for c in CLASS_NAMES],
    ))
    if st.button(tr("reset")):
        db.clear_scans(phone)
        st.session_state.pop("latest", None)
        st.rerun()
