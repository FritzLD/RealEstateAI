"""
src/auth.py
───────────
Access-code gate and lead-capture email for RealEstateAI.

Flow
────
1. render_access_gate() is called at the top of main() in app.py.
2. If st.session_state["authenticated"] is already True, returns True immediately.
3. Otherwise renders a centered form (name / email / phone / access code).
4. On submit: validate_code() checks the code against st.secrets["access_codes"].
5. On success: session state is set, send_lead_email() fires (silently), returns True.
6. On failure: shows an error and returns False → caller does st.stop().

Local development
─────────────────
If st.secrets has no "access_codes" section (i.e. no secrets.toml), the gate
is bypassed automatically so development works without any configuration.
"""

from __future__ import annotations

import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.text import MIMEText

import streamlit as st


# ── Code validation ────────────────────────────────────────────────────────────

def validate_code(code: str) -> str | None:
    """
    Check *code* (case-insensitive) against st.secrets["access_codes"].

    Returns the referrer/realtor name if valid, None if invalid.
    Returns the special string "DEV_BYPASS" if no access_codes section exists
    (local development without secrets.toml).
    """
    code = code.strip().upper()
    if not code:
        return None

    try:
        codes = st.secrets["access_codes"]
    except (KeyError, FileNotFoundError):
        # No secrets configured → bypass for local dev
        return "DEV_BYPASS"

    # TOML keys are case-sensitive; normalise both sides to upper
    codes_upper = {k.upper(): v for k, v in codes.items()}
    return codes_upper.get(code)          # None if not found


# ── Lead email ─────────────────────────────────────────────────────────────────

def send_lead_email(user_info: dict) -> bool:
    """
    Send a lead-notification email to Frederick via Gmail SMTP.

    Requires st.secrets["smtp"]:
        sender_email    – Gmail address used to send
        sender_password – Gmail App Password (not the account password)
        recipient_email – Frederick's address

    Returns True on success, False on any failure.
    Never raises – email failure must not block user access.
    """
    try:
        smtp_cfg = st.secrets["smtp"]
    except (KeyError, FileNotFoundError):
        return False          # SMTP not configured – silently skip

    try:
        sender    = smtp_cfg["sender_email"]
        password  = smtp_cfg["sender_password"]
        recipient = smtp_cfg["recipient_email"]
    except KeyError:
        return False

    referrer  = user_info.get("referrer", "Unknown")
    name      = user_info.get("name", "Unknown")
    timestamp = user_info.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    subject = f"[RealEstateAI] New Lead: {name} via {user_info.get('code', '?')}"
    body = (
        "New lead captured via RealEstateAI access gate:\n\n"
        f"  Name:        {name}\n"
        f"  Email:       {user_info.get('email', '')}\n"
        f"  Phone:       {user_info.get('phone', '')}\n"
        f"  Access Code: {user_info.get('code', '')}\n"
        f"  Referred by: {referrer}\n"
        f"  Time:        {timestamp}\n\n"
        "--\nSent automatically by RealEstateAI"
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        return True
    except Exception:
        return False          # Never raise – email failure must not block access


# ── Gate page ──────────────────────────────────────────────────────────────────

def render_access_gate() -> bool:
    """
    Render the full-page access gate.

    Returns True immediately if the user is already authenticated this session.
    Otherwise renders the form, validates the code, stores session state, and
    returns True on success or False if not yet authenticated (caller stops).
    """
    # Already authenticated this session → nothing to render
    if st.session_state.get("authenticated", False):
        return True

    # Check if access codes are even configured; if not, bypass (local dev)
    try:
        st.secrets["access_codes"]
        gate_active = True
    except (KeyError, FileNotFoundError):
        st.session_state["authenticated"] = True
        st.session_state["user_info"]     = {"name": "Dev", "referrer": "DEV_BYPASS"}
        return True

    if not gate_active:
        return True

    # ── Gate UI ───────────────────────────────────────────────────────────────
    # (st.set_page_config is already called in app.py — don't call it again)

    # Centre the form with padding columns
    _, col, _ = st.columns([1, 2, 1])

    with col:
        st.markdown("## 🏠 RealEstateAI")
        st.markdown("**Dayton MSA Market Intelligence**")
        st.divider()
        st.markdown(
            "To access this application, please enter your "
            "contact information and the access code provided by you."
        )
        st.write("")

        with st.form("access_gate_form", clear_on_submit=False):
            name  = st.text_input("Full Name",    placeholder="Jane Smith")
            email = st.text_input("Email",        placeholder="jane@example.com")
            phone = st.text_input("Phone Number", placeholder="(513) 555-0100")
            code  = st.text_input("Access Code",  placeholder="DUFF-001")
            submitted = st.form_submit_button("🔓  Request Access", use_container_width=True)

        if submitted:
            # Basic presence checks
            if not name.strip():
                st.error("Please enter your full name.")
            elif not email.strip():
                st.error("Please enter your email address.")
            elif not code.strip():
                st.error("Please enter your access code.")
            else:
                referrer = validate_code(code)
                if referrer is None:
                    st.error(
                        "❌ That access code is not recognised.  "
                        "Please double-check the code or contact Frederick Duff for assistance."
                    )
                else:
                    # Success — store session state
                    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    user_info = {
                        "name":      name.strip(),
                        "email":     email.strip(),
                        "phone":     phone.strip(),
                        "code":      code.strip().upper(),
                        "referrer":  referrer,
                        "timestamp": timestamp,
                    }
                    st.session_state["authenticated"] = True
                    st.session_state["user_info"]     = user_info

                    # Fire-and-forget lead email (silent on failure)
                    send_lead_email(user_info)

                    st.rerun()   # reload → gate returns True → main app renders

        # Contact footer
        st.divider()
        st.markdown(
            "Don't have a code? Contact:\n\n"
            "**Frederick Duff MBA**  \n"
            "📞 (502) 345-0682  \n"
            "📧 FDuff@QueenCitymortgage.net"
        )

    return False   # form shown but not yet authenticated
